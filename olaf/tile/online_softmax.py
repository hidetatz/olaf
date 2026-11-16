"""Online softmax fusion on Tile IR: attention-shaped task chains become FlashAttention-style tasks.

Matched forward chain (each step one task of the fused program, single writer, single use):

    S = contract(A, B)              one row label x, one column label y, one summed label
    X = f(S)                         pointwise (optional): scale, mask, any function of S
    m = rowmax_y(X)
    E = exp(X - F(m))                pointwise, F a function of m only
    l = rowsum_y(E)
    P = R(E / G(l, m))               pointwise, G a function of l and m, R an optional cast
    O = contract(P, V)               summing over y

becomes one task per (batch..., tile of x) that loops over tiles of y keeping a running max and
sum (the online softmax):

    m' = max(m, rowmax X_tile);  c = exp(m - F(m'))   (the exp task replayed on m)
    l  = l * c + rowsum E_tile;  acc = acc * c + R(E_tile) V_tile;  O = acc / G(l, m)

Every pointwise step is the matched task's own body, replayed on register tiles with the loads of
the chain's intermediates replaced by those tiles (`_replay`), so any scale or mask function and
any F, G work unchanged. What the pass needs from the algebra is only that the exp task is
exp(x - F(m)) (so the running rescale is exp of the difference) and that the last step divides
by a row constant (so the division moves after the sum over y). R moves before the division:
P is rounded before it is normalized, as FlashAttention does; this changes results at the
rounding level of R only (bf16: ~4e-3 relative per element).

When P also feeds a backward (P's other readers below, as PyTorch's attention backward and
olaf's composite VJP produce them), the backward is fused too, or nothing is changed:

    dP = contract(dO, V)  (the same V)      D = rowsum_y(dP * P)
    dS = pointwise(P, dP, D)                dQ = contract(dS, B'), dK = contract(dS, A'), dV = contract(P, dO)

D becomes rowsum(dO * O) (the sums exchange: sum_y P dP = sum_y P sum_e dO V = sum_e dO O),
and two tasks (FlashAttention-2's split) recompute P from m and l exactly as the unfused chain
does: one per tile of y for dK and dV (looping over x), one per tile of x for dQ (looping over y).

Limits: one row, one column and one summed label per contract; static head sizes that are
multiples of 16 up to 128; reductions over the last axis of plain loads.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from olaf.graph.dim import Dim
from olaf.graph.ops import parse_contract
from olaf.graph.types import DType
from olaf.tile.builder import INDEX, TileBuilder
from olaf.tile.fuse import _pointwise
from olaf.tile.ir import Arg, Body, Buffer, BufferKind, Launch, Program, Task, TileType, TOp, TValue, View, full_view

BLOCK = 64  # rows of x and of y per tile; with 64-wide heads the five dots stay within registers
WARPS = 4


@dataclass
class _Contract:
    index: int
    launch: Launch
    terms: tuple[str, str]
    out: str
    args: tuple[int, int]  # task arg of operand 0 and 1
    out_arg: int

    def buffer(self, operand: int) -> str:
        return self.launch.buffers[self.args[operand]]

    def view(self, operand: int) -> View:
        return self.launch.task.args[self.args[operand]].view

    def dtype(self, operand: int) -> DType:
        return self.launch.task.args[self.args[operand]].dtype

    @property
    def output(self) -> str:
        return self.launch.buffers[self.out_arg]


def _contract(index: int, launch: Launch) -> _Contract | None:
    origin = launch.task.origin
    if not origin or origin[0] != "contract":
        return None
    terms, out = parse_contract(origin[1])
    if len(terms) != 2:
        return None
    ops = _walk(launch.task.body)
    dots = [op for op in ops if op.kind == "dot"]
    stores = [op for op in ops if op.kind == "store"]
    if len(dots) != 1 or len(stores) != 1:
        return None
    producers = {v: op for op in ops for v in op.outputs}
    args = []
    for value in dots[0].inputs[:2]:
        while value in producers and producers[value].kind in ("reshape", "permute", "cast"):
            value = producers[value].inputs[0]
        load = producers.get(value)
        if load is None or load.kind != "load":
            return None
        args.append(load.attr("arg"))
    return _Contract(index, launch, (terms[0], terms[1]), out, (args[0], args[1]), stores[0].attr("arg"))


def _walk(body: Body):
    out = []
    for op in body.ops:
        out.append(op)
        for r in op.regions:
            out.extend(_walk(r))
    return out


def _reduce(launch: Launch, fn: str, rank: int) -> bool:
    origin = launch.task.origin
    return bool(origin) and origin[0] == "reduce" and origin[1] == fn and origin[2] == (rank - 1,)


class _Program:
    def __init__(self, program: Program) -> None:
        self.program = program
        self.buffers = {b.name: b for b in program.buffers}
        self.writers: dict[str, list[int]] = {}
        self.readers: dict[str, list[int]] = {}
        for i, launch in enumerate(program.launches):
            for name, arg in zip(launch.buffers, launch.task.args):
                (self.writers if arg.writes else self.readers).setdefault(name, []).append(i)

    def launch(self, i: int) -> Launch:
        return self.program.launches[i]

    def writer(self, name: str) -> int | None:
        w = self.writers.get(name, [])
        return w[0] if len(w) == 1 and self.buffers[name].kind is BufferKind.temp else None

    def only_readers(self, name: str) -> set[int]:
        return set(self.readers.get(name, []))

    def reads(self, i: int, name: str) -> list[int]:
        launch = self.launch(i)
        return [a for a, (n, arg) in enumerate(zip(launch.buffers, launch.task.args)) if n == name and not arg.writes]

    def written(self, i: int) -> str:
        launch = self.launch(i)
        outs = [n for n, a in zip(launch.buffers, launch.task.args) if a.writes]
        return outs[0] if len(outs) == 1 else ""


@dataclass
class _Forward:
    c1: _Contract
    t1: int | None
    rmax: int
    t3: int
    rsum: int
    t5: int
    c2: _Contract
    x_arg_t3: int  # T3's arg reading X
    m_arg_t3: list[int]  # T3's args reading m
    e_arg_t5: int
    l_arg_t5: list[int]
    m_arg_t5: list[int]
    s_arg_t1: int | None
    labels: dict[str, str]  # roles -> C1 label: row, col, sum; and "value" (C2's output column label)
    p_term: str  # C2's term for P
    v_operand: int  # which C2 operand is V


def _identity_read(p: _Program, i: int, name: str) -> int | None:
    reads = p.reads(i, name)
    if len(reads) != 1:
        return None
    arg = p.launch(i).task.args[reads[0]]
    return reads[0] if arg.view == full_view(p.buffers[name].type.shape) else None


def _row_reads(p: _Program, i: int, name: str, rank: int) -> list[int] | None:
    """Args of launch i reading a row statistic (rank-1) broadcast along the last of rank axes."""
    reads = p.reads(i, name)
    for a in reads:
        source = p.launch(i).task.args[a].view.source
        if source is None or [s for s in source] != [(j, 0) for j in range(rank - 1)]:
            return None
    return reads


def _value_of(task: Task, kind: str):
    store = next(op for op in task.body.ops if op.kind == "store")
    producers = {v: op for op in task.body.ops for v in op.outputs}
    return store.inputs[0], producers


def _depends_on(value: TValue, producers: dict, arg: int) -> bool:
    stack, seen = [value], set()
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        op = producers.get(v)
        if op is None:
            continue
        if op.kind == "load" and op.attr("arg") == arg:
            return True
        stack.extend(op.inputs)
    return False


def _match_forward(p: _Program, c2_index: int) -> _Forward | None:
    c2 = _contract(c2_index, p.launch(c2_index))
    if c2 is None:
        return None
    for v_operand in (1, 0):
        match = _match_forward_with(p, c2, v_operand)
        if match is not None:
            return match
    return None


def _match_forward_with(p: _Program, c2: _Contract, v_operand: int) -> _Forward | None:
    p_name = c2.buffer(1 - v_operand)
    t5 = p.writer(p_name)
    if t5 is None or _pointwise(p.launch(t5), p.buffers) is None:
        return None
    # E: the T5 input read with the identity view, produced by T3 and reduced by rsum.
    t5_task = p.launch(t5).task
    value, producers = _value_of(t5_task, "store")
    if producers.get(value) is not None and producers[value].kind == "cast":
        value = producers[value].inputs[0]
    division = producers.get(value)
    if division is None or division.kind != "div":
        return None
    numerator = producers.get(division.inputs[0])
    if numerator is None or numerator.kind != "load":
        return None
    e_name = p.launch(t5).buffers[numerator.attr("arg")]
    if _identity_read(p, t5, e_name) is None:
        return None
    rank = p.buffers[e_name].type.rank
    t3 = p.writer(e_name)
    if t3 is None or _pointwise(p.launch(t3), p.buffers) is None:
        return None
    readers_e = p.only_readers(e_name)
    rsum = next((i for i in readers_e if i != t5), None)
    if rsum is None or readers_e != {t5, rsum} or not _reduce(p.launch(rsum), "sum", rank) or _identity_read(p, rsum, e_name) is None:
        return None
    l_name = p.written(rsum)
    if p.writer(l_name) != rsum or p.only_readers(l_name) != {t5}:
        return None
    # T3 = exp(load X - F(m)).
    t3_task = p.launch(t3).task
    value, producers = _value_of(t3_task, "store")
    exp = producers.get(value)
    if exp is None or exp.kind != "exp":
        return None
    sub = producers.get(exp.inputs[0])
    if sub is None or sub.kind != "sub":
        return None
    x_load = producers.get(sub.inputs[0])
    if x_load is None or x_load.kind != "load":
        return None
    x_arg_t3 = x_load.attr("arg")
    x_name = p.launch(t3).buffers[x_arg_t3]
    if _depends_on(sub.inputs[1], producers, x_arg_t3) or _identity_read(p, t3, x_name) is None:
        return None
    readers_x = p.only_readers(x_name)
    rmax = next((i for i in readers_x if i != t3), None)
    if rmax is None or readers_x != {t3, rmax} or not _reduce(p.launch(rmax), "max", rank) or _identity_read(p, rmax, x_name) is None:
        return None
    m_name = p.written(rmax)
    if p.writer(m_name) != rmax or not p.only_readers(m_name) <= {t3, t5}:
        return None
    m_arg_t3 = _row_reads(p, t3, m_name, rank)
    m_arg_t5 = _row_reads(p, t5, m_name, rank)
    l_arg_t5 = _row_reads(p, t5, l_name, rank)
    if m_arg_t3 is None or m_arg_t5 is None or not l_arg_t5:
        return None
    # The rest of T3 and T5 may read only scalar constants (stride-0 views).
    for i, allowed in ((t3, {x_name, m_name}), (t5, {e_name, l_name, m_name})):
        launch = p.launch(i)
        for name, arg in zip(launch.buffers, launch.task.args):
            if not arg.writes and name not in allowed and any(s != 0 for s in arg.view.strides):
                return None
    # X = f(S) (T1) or X = S.
    t1 = p.writer(x_name)
    if t1 is None:
        return None
    if _contract(t1, p.launch(t1)) is not None:
        c1, t1, s_arg = _contract(t1, p.launch(t1)), None, None
    else:
        if _pointwise(p.launch(t1), p.buffers) is None:
            return None
        launch = p.launch(t1)
        candidates = [n for n, a in zip(launch.buffers, launch.task.args) if not a.writes and any(s != 0 for s in a.view.strides)]
        if len(candidates) != 1 or _identity_read(p, t1, candidates[0]) is None or p.only_readers(candidates[0]) != {t1}:
            return None
        s_name = candidates[0]
        s_arg = _identity_read(p, t1, s_name)
        w = p.writer(s_name)
        c1 = _contract(w, p.launch(w)) if w is not None else None
        if c1 is None:
            return None
    # Labels: C1 = (row, sum) x (col, sum) -> rows x cols; C2 sums over cols.
    ta, tb = c1.terms
    batch = [c for c in c1.out[:-2]]
    row, col = c1.out[-2], c1.out[-1]
    summed = [c for c in ta if c in tb and c not in c1.out]
    if len(summed) != 1 or row not in ta or row in tb or col not in tb or col in ta or any(c not in ta or c not in tb for c in batch):
        return None
    p_term = c2.terms[1 - v_operand]
    v_term = c2.terms[v_operand]
    rename = dict(zip(p_term, c1.out))
    if len(p_term) != rank:
        return None
    c2_sum = [c for c in p_term if c in v_term and c not in c2.out]
    if len(c2_sum) != 1 or rename[c2_sum[0]] != col or c2.out[:-1] != p_term[:-1] or rename.get(c2.out[-2]) != row:
        return None
    value_label = c2.out[-1]
    if value_label in p_term or value_label not in v_term:
        return None
    labels = {"row": row, "col": col, "sum": summed[0], "value": value_label, "batch": "".join(batch)}
    head, value_size = _head_sizes(c1, c2, v_operand, labels)
    if not all(isinstance(d, int) and d % 16 == 0 and 16 <= d <= 128 for d in (head, value_size)):
        return None
    return _Forward(c1, t1, rmax, t3, rsum, t5, c2, x_arg_t3, m_arg_t3, e_arg_t5=numerator.attr("arg"), l_arg_t5=l_arg_t5, m_arg_t5=m_arg_t5, s_arg_t1=s_arg, labels=labels, p_term=p_term, v_operand=v_operand)


def _head_sizes(c1: _Contract, c2: _Contract, v_operand: int, labels: dict) -> tuple[Dim, Dim]:
    """Sizes of the summed label of C1 and of V's value label (both become whole tiles)."""
    head = dict(zip(c1.terms[0], c1.view(0).shape))[labels["sum"]]
    value = dict(zip(c2.terms[v_operand], c2.view(v_operand).shape))[labels["value"]]
    return head, value


def c2_mapping(f) -> dict[str, str]:
    """C2's letters -> semantic names (C1's letters, and VALUE for the value axis)."""
    from olaf.tile.online_softmax_backward import VALUE

    mapping = dict(zip(f.p_term, f.c1.out))
    mapping[f.labels["value"]] = VALUE
    return mapping


# -- backward -------------------------------------------------------------------------------------


@dataclass
class _Backward:
    c3: _Contract  # dP = contract(dO, V)
    rd: int  # D = rowsum(dP * P)
    t9: int  # dS = pointwise(P, dP, D)
    dq: _Contract
    dk: _Contract
    dv: _Contract
    do_operand_c3: int
    p_arg_t9: int
    dp_arg_t9: int
    d_arg_t9: list[int]
    ds_operand_dq: int
    ds_operand_dk: int
    p_operand_dv: int


def _is_product(launch: Launch, args: list[int]) -> bool:
    """The task multiplies the loads of `args` (through casts) exactly once, and computes nothing else.

    Edge masks that fusion put around an inlined product (where over iota bounds, selecting the
    reduction's neutral value) are allowed; any other arithmetic is not.
    """
    ops = _walk(launch.task.body)
    producers = {v: op for op in ops for v in op.outputs}

    def source(value: TValue) -> int | None:
        while value in producers and producers[value].kind == "cast":
            value = producers[value].inputs[0]
        op = producers.get(value)
        return op.attr("arg") if op is not None and op.kind == "load" else None

    products = [op for op in ops if op.kind == "mul" and op.outputs[0].type.dtype.is_float]
    if len(products) != 1 or sorted(source(v) or -1 for v in products[0].inputs) != sorted(args):
        return False
    allowed = {"load", "cast", "mul", "full", "for", "add", "reduce", "store", "dim", "where", "cmp", "iota", "broadcast"}
    return {op.kind for op in ops} <= allowed


def _match_backward(p: _Program, f: _Forward) -> _Backward | None:
    p_name = f.c2.buffer(1 - f.v_operand)
    others = p.only_readers(p_name) - {f.c2.index}
    if len(others) != 3:
        return None
    contracts = {i: _contract(i, p.launch(i)) for i in others}
    dv_index = [i for i, c in contracts.items() if c is not None]
    if len(dv_index) != 1:
        return None
    dv = contracts[dv_index[0]]
    rest = others - {dv.index}
    rank = p.buffers[p_name].type.rank
    rd = next((i for i in rest if _reduce(p.launch(i), "sum", rank)), None)
    t9 = next((i for i in rest if i != rd), None)
    if rd is None or t9 is None or _pointwise(p.launch(t9), p.buffers) is None:
        return None
    # D = rowsum(dP * P): R_D reads P and dP and nothing else but constants.
    rd_launch = p.launch(rd)
    inputs = [n for n, a in zip(rd_launch.buffers, rd_launch.task.args) if not a.writes and any(s != 0 for s in a.view.strides)]
    if len(inputs) != 2 or p_name not in inputs:
        return None
    dp_name = next(n for n in inputs if n != p_name)
    if _identity_read(p, rd, p_name) is None or _identity_read(p, rd, dp_name) is None:
        return None
    if not _is_product(rd_launch, [_identity_read(p, rd, p_name), _identity_read(p, rd, dp_name)]):
        return None
    c3_index = p.writer(dp_name)
    c3 = _contract(c3_index, p.launch(c3_index)) if c3_index is not None else None
    if c3 is None or p.only_readers(dp_name) != {rd, t9}:
        return None
    v_name, v_view = f.c2.buffer(f.v_operand), f.c2.view(f.v_operand)
    v_side = [o for o in (0, 1) if c3.buffer(o) == v_name and c3.view(o) == v_view]
    if len(v_side) != 1:
        return None
    do_operand = 1 - v_side[0]
    d_name = p.written(rd)
    if p.writer(d_name) != rd or p.only_readers(d_name) != {t9}:
        return None
    # dS = T9(P, dP, D), read by exactly the dQ and dK contracts.
    p_arg = _identity_read(p, t9, p_name)
    dp_arg = _identity_read(p, t9, dp_name)
    d_args = _row_reads(p, t9, d_name, rank)
    if p_arg is None or dp_arg is None or not d_args:
        return None
    t9_launch = p.launch(t9)
    for name, arg in zip(t9_launch.buffers, t9_launch.task.args):
        if not arg.writes and name not in (p_name, dp_name, d_name) and any(s != 0 for s in arg.view.strides):
            return None
    ds_name = p.written(t9)
    ds_readers = p.only_readers(ds_name)
    if len(ds_readers) != 2:
        return None
    ds_contracts = [_contract(i, p.launch(i)) for i in ds_readers]
    if None in ds_contracts:
        return None
    a_name, a_view = f.c1.buffer(0), f.c1.view(0)
    b_name, b_view = f.c1.buffer(1), f.c1.view(1)
    dq = dk = None
    ds_dq = ds_dk = None
    for c in ds_contracts:
        for o in (0, 1):
            if c.buffer(o) != ds_name:
                continue
            other = 1 - o
            if c.buffer(other) == b_name and c.view(other) == b_view:
                dq, ds_dq = c, o
            elif c.buffer(other) == a_name and c.view(other) == a_view:
                dk, ds_dk = c, o
    if dq is None or dk is None:
        return None
    do_name, do_view = c3.buffer(do_operand), c3.view(do_operand)
    p_dv = [o for o in (0, 1) if dv.buffer(o) == p_name]
    if len(p_dv) != 1 or dv.buffer(1 - p_dv[0]) != do_name or dv.view(1 - p_dv[0]) != do_view:
        return None
    return _Backward(c3, rd, t9, dq, dk, dv, do_operand, p_arg, dp_arg, d_args, ds_dq, ds_dk, p_dv[0])


# -- building the fused tasks ------------------------------------------------------------------


class _Task:
    """Builds one task: argument bookkeeping over a TileBuilder."""

    def __init__(self) -> None:
        self.b = TileBuilder()
        self.args: list[tuple[str, Arg]] = []

    def arg(self, buffer: str, view: View, dtype: DType, writes: bool = False) -> int:
        key = (buffer, Arg(view, dtype, writes))
        if key not in self.args:
            self.args.append(key)
        return self.args.index(key)

    def load(self, buffer: str, view: View, dtype: DType, starts, shape, other=0) -> TValue:
        return self.b.op("load", *starts, arg=self.arg(buffer, view, dtype), shape=tuple(shape), dtype=dtype, other=other)

    def finish(self, name: str, grid, pids, origin) -> Launch:
        task = Task(name, tuple(grid), tuple(a for _, a in self.args), self.b.finish(pids), WARPS, origin)
        return Launch(task, tuple(n for n, _ in self.args))


def _operand(t: _Task, c: _Contract, operand: int, starts: dict, extents: dict, rows: str, cols: str) -> TValue:
    """C's operand as a 2-D f32 tile (rows, cols), labels in C's own naming."""
    b = t.b
    term = c.terms[operand]
    dtype = c.dtype(operand)
    tile = t.load(c.buffer(operand), c.view(operand), dtype, [starts[x] for x in term], [extents[x] for x in term])
    if dtype is not DType.f32:
        tile = b.op("cast", tile, dtype=DType.f32)
    order = [i for i, x in enumerate(term) if x not in (rows, cols)] + [term.index(rows), term.index(cols)]
    if order != list(range(len(term))):
        tile = b.op("permute", tile, perm=tuple(order))
    return b.op("reshape", tile, shape=(extents[rows], extents[cols]))


def _store(t: _Task, buffer: str, view: View, dtype: DType, term: str, value: TValue, rows: str, cols: str, starts: dict, extents: dict) -> None:
    b = t.b
    if value.type.dtype is not dtype:
        value = b.op("cast", value, dtype=dtype)
    unit = [x for x in term if x not in (rows, cols)]
    shaped = b.op("reshape", value, shape=tuple([1] * len(unit) + [extents[rows], extents[cols]]))
    current = unit + [rows, cols]
    perm = tuple(current.index(x) for x in term)
    if perm != tuple(range(len(term))):
        shaped = b.op("permute", shaped, perm=perm)
    b.emit("store", (shaped, *(starts[x] for x in term)), {"arg": t.arg(buffer, view, dtype, True)})


def _replay(t: _Task, p: _Program, index: int, subs: dict[int, TValue], tile: tuple[int, ...], starts: list[TValue], target: TValue | None = None) -> TValue:
    """Evaluate pointwise launch `index` on `tile` at `starts`, with loads of `subs` args replaced."""
    b = t.b
    launch = p.launch(index)
    info = _pointwise(launch, p.buffers)
    task = launch.task
    p_tile = info.store.inputs[0].type.shape
    mapped: dict[TValue, TValue] = {}
    for op in task.body.ops:
        if op.kind == "store" or (op.outputs and set(op.outputs) <= info.coordinate):
            continue
        ins = [mapped.get(v, v) for v in op.inputs]
        if op.kind == "load":
            a = op.attr("arg")
            if a in subs:
                out = subs[a]
                if out.type.shape != tile:
                    out = b.op("broadcast", out, shape=tile, dims=tuple(range(out.type.rank)) if out.type.rank == len(tile) else ())
            else:
                arg = task.args[a]
                out = t.load(launch.buffers[a], arg.view, arg.dtype, starts, tile, op.attr("other"))
        elif op.kind == "iota":
            axis = info.starts.index(op.inputs[0])
            out = b.op("iota", starts[axis], shape=tile, axis=op.attr("axis"))
        elif op.kind in ("full", "broadcast") and op.outputs[0].type.shape == p_tile and p_tile != ():
            out = b.op(op.kind, *ins, **{**dict(op.attrs), "shape": tile})
        else:
            out = b.op(op.kind, *ins, **dict(op.attrs))
        mapped[op.outputs[0]] = out
        if target is not None and op.outputs[0] is target:
            return out
    return mapped[info.store.inputs[0]]


def _rows(b: TileBuilder, value: TValue, rank: int, block: int) -> TValue:
    """A (block,) row statistic as the (1, ..., 1, block) tile the replayed tasks expect."""
    return b.op("reshape", value, shape=tuple([1] * (rank - 2) + [block]))


def _spread(b: TileBuilder, rows: TValue, tile: tuple[int, ...]) -> TValue:
    """(1,...,1,block) rows broadcast along the last axis of tile."""
    return b.op("broadcast", rows, shape=tile, dims=tuple(range(len(tile) - 1)))


def _masked(t: _Task, value: TValue, starts: list[TValue], limits: list[tuple[int, Dim]]) -> TValue:
    """value with -inf where a coordinate (axis, size) is out of range (tile edges)."""
    b = t.b
    tile = value.type.shape
    neg = b.full(tile, float("-inf"), value.type.dtype)
    for axis, size in limits:
        coord = b.op("iota", starts[axis], shape=tile, axis=axis)
        bound = b.op("broadcast", b.dim(size), shape=tile, dims=())
        value = b.op("where", b.op("cmp", coord, bound, fn="lt"), value, neg)
    return value


def _causal_offset(p: _Program, f: _Forward) -> int | None:
    """d when X is where(col <= row + d, ..., -inf) (a causal mask), else None.

    Read from T1's body: the stored value is a where whose false branch loads a -inf constant
    and whose condition compares iotas along the column and row axes (le: d = 0, lt: d = -1).
    Tiles entirely past that diagonal contribute exactly nothing, so loops can skip them.
    """
    if f.t1 is None:
        return None
    launch = p.launch(f.t1)
    value, producers = _value_of(launch.task, "store")
    where = producers.get(value)
    if where is None or where.kind != "where":
        return None
    other = producers.get(where.inputs[2])
    if other is None or other.kind != "load":
        return None
    constant = p.buffers[launch.buffers[other.attr("arg")]].constant
    if constant is None or constant.array.size != 1 or not (float(constant.array.reshape(())) == float("-inf")):
        return None
    cmp = producers.get(where.inputs[0])
    if cmp is None or cmp.kind != "cmp" or cmp.attr("fn") not in ("le", "lt"):
        return None
    axes = []
    for v in cmp.inputs:
        iota = producers.get(v)
        if iota is None or iota.kind != "iota":
            return None
        axes.append(iota.attr("axis"))
    rank = len(f.c1.out)
    if axes != [rank - 1, rank - 2]:
        return None
    return 0 if cmp.attr("fn") == "le" else -1


class _Builder:
    """Shared pieces of the fused forward and backward tasks."""

    def __init__(self, p: _Program, f: _Forward) -> None:
        self.p, self.f = p, f
        c1 = f.c1
        self.rank = len(c1.out)
        self.labels = f.labels
        self.batch = list(f.labels["batch"])
        self.shape = dict(zip(c1.out, p.buffers[c1.output].type.shape))
        self.head, self.value_size = _head_sizes(c1, f.c2, f.v_operand, f.labels)
        self.causal = _causal_offset(p, f)

    def col_end(self, b: TileBuilder, row_start: TValue) -> TValue:
        """End of the column loop for rows [row_start, row_start + BLOCK)."""
        end = b.dim(self.shape[self.labels["col"]])
        if self.causal is None:
            return end
        last = b.op("add", row_start, b.index(BLOCK + self.causal))
        return b.op("min", end, last)

    def row_begin(self, b: TileBuilder, col_start: TValue) -> TValue:
        """Start of the row loop for columns [col_start, col_start + BLOCK) (a multiple of BLOCK)."""
        if self.causal is None:
            return b.index(0)
        first = b.op("sub", col_start, b.index(self.causal))
        return b.op("mul", b.op("floordiv", first, b.index(BLOCK)), b.index(BLOCK))

    def scores(self, t: _Task, q_starts: dict, k_starts: dict, q_tile: TValue | None = None) -> tuple[TValue, TValue, list[TValue]]:
        """X tile (rank-d, 1..1 x BLOCK x BLOCK) for the query and key tiles at these starts."""
        b, f, lab = t.b, self.f, self.labels
        row, col, s = lab["row"], lab["col"], lab["sum"]
        ext = {x: 1 for x in self.batch} | {row: BLOCK, col: BLOCK, s: self.head}
        starts = {**q_starts, **k_starts, s: b.index(0)}
        a = q_tile if q_tile is not None else _operand(t, f.c1, 0, starts, ext, row, s)
        k = _operand(t, f.c1, 1, starts, ext, s, col)
        dot = b.op("dot", a, k, b.full((BLOCK, BLOCK), 0, DType.f32))
        tile = tuple([1] * len(self.batch) + [BLOCK, BLOCK])
        s4 = b.op("reshape", dot, shape=tile)
        coords = [starts[x] for x in f.c1.out]
        x = s4 if f.t1 is None else _replay(t, self.p, f.t1, {f.s_arg_t1: s4}, tile, coords)
        limits = [(self.rank - 2, self.shape[row]), (self.rank - 1, self.shape[col])]
        return _masked(t, x, coords, limits), a, coords

    def p_exact(self, t: _Task, x: TValue, coords, m_rows: TValue, l_rows: TValue) -> TValue:
        """P as the unfused chain computes it, from the final row statistics."""
        f, b = self.f, t.b
        tile = x.type.shape
        e = _replay(t, self.p, f.t3, {f.x_arg_t3: x, **{a: _spread(b, m_rows, tile) for a in f.m_arg_t3}}, tile, coords)
        subs = {f.e_arg_t5: e, **{a: _spread(b, m_rows, tile) for a in f.m_arg_t5}, **{a: _spread(b, l_rows, tile) for a in f.l_arg_t5}}
        return _replay(t, self.p, f.t5, subs, tile, coords)

    def pids(self, t: _Task, tiled: str) -> tuple[list[TValue], dict, list[Dim]]:
        b = t.b
        pids = [TValue(TileType((), INDEX)) for _ in range(len(self.batch) + 1)]
        starts = {x: pid for x, pid in zip(self.batch, pids)}
        starts[tiled] = b.op("mul", pids[-1], b.index(BLOCK))
        grid = [self.shape[x] for x in self.batch] + [_ceil(self.shape[tiled], BLOCK)]
        return pids, starts, grid


def _ceil(d: Dim, t: int) -> Dim:
    return -(-d // t) if isinstance(d, int) else (d + (t - 1)) // t


def _forward_task(p: _Program, f: _Forward, write_stats: bool) -> Launch:
    from olaf.tile.online_softmax_backward import VALUE, _letter, _local

    g = _Builder(p, f)
    t = _Task()
    b = t.b
    lab = f.labels
    row, col = lab["row"], lab["col"]
    m2 = c2_mapping(f)
    ext_v = {x: 1 for x in g.batch} | {col: BLOCK, VALUE: g.value_size}
    pids, q_starts, grid = g.pids(t, row)
    rows_tile = tuple([1] * len(g.batch) + [BLOCK])
    ext = {x: 1 for x in g.batch} | {row: BLOCK, col: BLOCK, lab["sum"]: g.head}
    q_tile = _operand(t, f.c1, 0, {**q_starts, lab["sum"]: b.index(0)}, ext, row, lab["sum"])
    m0 = b.full(rows_tile, float("-inf"), DType.f32)
    l0 = b.full(rows_tile, 0.0, DType.f32)
    acc0 = b.full((BLOCK, g.value_size), 0.0, DType.f32)
    rank = g.rank
    t5_value, t5_producers = _value_of(p.launch(f.t5).task, "store")
    rounding = t5_producers[t5_value] if t5_producers[t5_value].kind == "cast" else None
    division = t5_producers[rounding.inputs[0]] if rounding is not None else t5_producers[t5_value]

    def body(inner: TileBuilder, k0: TValue, carries):
        t.b = inner
        m, l, acc = carries
        x, _, coords = g.scores(t, q_starts, {col: k0}, q_tile)
        tile = x.type.shape
        m_new = inner.op("max", m, inner.op("reduce", x, fn="max", axes=(rank - 1,)))
        spread = {a: _spread(inner, m_new, tile) for a in f.m_arg_t3}
        e = _replay(t, p, f.t3, {f.x_arg_t3: x, **spread}, tile, coords)
        row_tile = rows_tile + (1,)
        old = inner.op("reshape", m, shape=row_tile)
        new = inner.op("reshape", m_new, shape=row_tile)
        c = _replay(t, p, f.t3, {f.x_arg_t3: old, **{a: new for a in f.m_arg_t3}}, row_tile, coords)
        c = inner.op("reshape", c, shape=rows_tile)
        l_new = inner.op("add", inner.op("mul", l, c), inner.op("reduce", e, fn="sum", axes=(rank - 1,)))
        pe = e if rounding is None else inner.op("cast", inner.op("cast", e, dtype=rounding.attr("dtype")), dtype=DType.f32)
        p2 = inner.op("reshape", pe, shape=(BLOCK, BLOCK))
        v_starts = _local(m2, {**q_starts, col: k0, VALUE: inner.index(0)})
        v = _operand(t, f.c2, f.v_operand, v_starts, _local(m2, ext_v), _letter(m2, col), _letter(m2, VALUE))
        scale = inner.op("broadcast", inner.op("reshape", c, shape=(BLOCK,)), shape=(BLOCK, g.value_size), dims=(0,))
        acc_new = inner.op("dot", p2, v, inner.op("mul", acc, scale))
        return [m_new, l_new, acc_new]

    m, l, acc = b.for_range(b.index(0), g.col_end(b, q_starts[row]), BLOCK, [m0, l0, acc0], body)
    t.b = b
    # O = acc / G(l, m): G is T5's divisor, replayed on the final statistics.
    coords = [q_starts.get(x, b.index(0)) for x in f.c1.out]
    row_tile = rows_tile + (1,)
    subs = {**{a: b.op("reshape", m, shape=row_tile) for a in f.m_arg_t5}, **{a: b.op("reshape", l, shape=row_tile) for a in f.l_arg_t5}, f.e_arg_t5: b.full(row_tile, 1.0, DType.f32)}
    divisor = _replay(t, p, f.t5, subs, row_tile, coords, target=division.inputs[1])
    out = b.op("div", acc, b.op("broadcast", b.op("reshape", divisor, shape=(BLOCK,)), shape=(BLOCK, g.value_size), dims=(0,)))
    c2 = f.c2
    o_starts = _local(m2, {**q_starts, VALUE: b.index(0)})
    out_arg = c2.launch.task.args[c2.out_arg]
    _store(t, c2.output, out_arg.view, out_arg.dtype, c2.out, out, _letter(m2, row), _letter(m2, VALUE), o_starts, _local(m2, ext_v | {row: BLOCK}))
    if write_stats:
        for name, stat in ((p.written(f.rmax), m), (p.written(f.rsum), l)):
            buffer = p.buffers[name]
            b.emit("store", (stat, *(q_starts[x] for x in f.c1.out[:-1])), {"arg": t.arg(name, full_view(buffer.type.shape), buffer.type.dtype, True)})
    return t.finish("attention", grid, pids, ("attention",))


def fuse_online_softmax(program: Program) -> Program:
    p = _Program(program)
    replaced: dict[int, list[Launch]] = {}
    removed: set[int] = set()
    for i in range(len(program.launches)):
        if i in removed:
            continue
        f = _match_forward(p, i)
        if f is None:
            continue
        forward_tasks = {f.c1.index, f.rmax, f.t3, f.rsum, f.t5, f.c2.index} | ({f.t1} if f.t1 is not None else set())
        p_name = f.c2.buffer(1 - f.v_operand)
        extra = p.only_readers(p_name) - {f.c2.index}
        if not extra:
            if p.only_readers(p.written(f.rmax)) - {f.t3, f.t5}:
                continue
            replaced[f.c2.index] = [_forward_task(p, f, write_stats=False)]
            removed |= forward_tasks - {f.c2.index}
            continue
        back = _match_backward(p, f)
        if back is None:
            continue
        from olaf.tile.online_softmax_backward import mappings

        if mappings(f, back) is None:
            continue
        backward_tasks = {back.c3.index, back.rd, back.t9, back.dq.index, back.dk.index, back.dv.index}
        last = max(backward_tasks)
        outputs = {back.dq.output, back.dk.output, back.dv.output}
        if any(r < last for name in outputs for r in p.only_readers(name)):
            continue
        from olaf.tile.online_softmax_backward import backward_tasks as build_backward

        replaced[f.c2.index] = [_forward_task(p, f, write_stats=True)]
        replaced[last] = build_backward(p, f, back, _Builder(p, f))
        removed |= (forward_tasks - {f.c2.index}) | (backward_tasks - {last})
    if not replaced:
        return program
    launches = []
    for i, launch in enumerate(program.launches):
        if i in replaced:
            launches.extend(replaced[i])
        elif i not in removed:
            launches.append(launch)
    used = {n for launch in launches for n in launch.buffers} | set(program.inputs) | set(program.outputs)
    buffers = tuple(b for b in program.buffers if b.name in used or b.kind is not BufferKind.temp)
    return replace(program, buffers=buffers, launches=tuple(launches))
