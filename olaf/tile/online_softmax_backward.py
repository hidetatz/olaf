"""The fused attention backward (see tile/online_softmax.py for the pattern and the math).

Contracts name their axes with their own letters. Everything here works with semantic names
instead: C1's batch labels and its row, col and sum labels, plus VALUE for the value axis of V.
Each contract's letters are mapped to them through buffers whose axes are known: S, P, dP and
dS have C1's output axes, Q and K C1's operand axes, V C2's V axes and dO C3's dO axes.
"""

from __future__ import annotations

from olaf.graph.types import DType
from olaf.tile.builder import TileBuilder
from olaf.tile.ir import Launch, full_view

VALUE = "@value"


def semantic(contract, operand: int, names: list[str]) -> dict[str, str] | None:
    """Letters of `contract` -> semantic names, from the semantic names of one operand's axes."""
    term = contract.terms[operand]
    if len(term) != len(names):
        return None
    out = dict(zip(term, names))
    other = contract.terms[1 - operand]
    for label in other:
        if label not in out:
            out[label] = None  # filled from the other operand's names by the caller
    return out


def complete(contract, mapping: dict, operand: int, names: list[str]) -> dict[str, str] | None:
    """Merge the names of the other operand's axes; letters must agree."""
    term = contract.terms[operand]
    if len(term) != len(names):
        return None
    for label, name in zip(term, names):
        if mapping.get(label) not in (None, name):
            return None
        mapping[label] = name
    return mapping if all(v is not None for v in mapping.values()) else None


def roles(contract, mapping: dict) -> tuple[list[str], str, list[str]]:
    """(output names, summed name, ...) of a contract in semantic names."""
    ta, tb = contract.terms
    summed = [mapping[x] for x in ta if x in tb and x not in contract.out]
    return [mapping[x] for x in contract.out], summed[0] if len(summed) == 1 else "", summed


def mappings(f, back) -> dict | None:
    """Semantic mappings of C3, dQ, dK and dV, or None if a contract does not have its role."""
    c1, lab = f.c1, f.labels
    s_names = list(c1.out)
    q_names, k_names = list(c1.terms[0]), list(c1.terms[1])
    v_names = [dict(zip(f.p_term, c1.out)).get(x, VALUE if x == lab["value"] else None) for x in f.c2.terms[f.v_operand]]
    if None in v_names:
        return None
    batch = list(lab["batch"])
    row, col, s = lab["row"], lab["col"], lab["sum"]
    out = {}
    c3 = back.c3
    m3 = {x: n for x, n in zip(c3.out, s_names)}
    m3 = complete(c3, m3 | {x: None for x in c3.terms[0] + c3.terms[1] if x not in m3}, 1 - back.do_operand_c3, v_names)
    if m3 is None or roles(c3, m3)[1] != VALUE:
        return None
    out["c3"] = m3
    do_names = [m3[x] for x in c3.terms[back.do_operand_c3]]
    for key, contract, ds_side, other_names, expect_out, expect_sum in (
        ("dq", back.dq, back.ds_operand_dq, k_names, {*batch, row, s}, col),
        ("dk", back.dk, back.ds_operand_dk, q_names, {*batch, col, s}, row),
        ("dv", back.dv, back.p_operand_dv, do_names, {*batch, col, VALUE}, row),
    ):
        m = semantic(contract, ds_side, s_names)
        m = complete(contract, m, 1 - ds_side, other_names) if m is not None else None
        if m is None:
            return None
        names, summed, _ = roles(contract, m)
        if set(names) != expect_out or summed != expect_sum:
            return None
        out[key] = m
    return out


def _local(mapping: dict, values: dict) -> dict:
    """A dict keyed by semantic names -> keyed by the contract's letters."""
    return {x: values[n] for x, n in mapping.items() if n in values}


def backward_tasks(p, f, back, g) -> list[Launch]:
    from olaf.tile.online_softmax import BLOCK, FORWARD_ROWS, FORWARD_WARPS, _Builder, _operand, _replay, _spread, _store, _Task

    maps = mappings(f, back)
    lab = f.labels
    batch = list(lab["batch"])
    row, col, s = lab["row"], lab["col"], lab["sum"]
    rank = g.rank
    c3, m3 = back.c3, maps["c3"]

    def extents(g):
        return {x: 1 for x in batch} | {row: g.rows, col: g.cols, s: g.head, VALUE: g.value_size}

    def rows_tile(g):
        return tuple([1] * len(batch) + [g.rows])
    m_name, l_name = p.written(f.rmax), p.written(f.rsum)
    d_name = p.written(back.rd)
    d_buffer = p.buffers[d_name]

    def stat(t, g, name, starts):
        buffer = p.buffers[name]
        return t.load(name, full_view(buffer.type.shape), buffer.type.dtype, [starts[x] for x in batch + [row]], rows_tile(g))

    # D = rowsum(dO * O): one tile of rows per instance.
    t = _Task()
    b = t.b
    ext = extents(g)
    pids, starts, grid = g.pids(t, row)
    starts[VALUE] = b.index(0)
    d_o = _operand(t, c3, back.do_operand_c3, _local(m3, starts), _local(m3, ext), _letter(m3, row), _letter(m3, VALUE))
    c2 = f.c2
    o_names = dict(zip(c2.out, [*f.c1.out[:-1], VALUE]))
    out_arg = c2.launch.task.args[c2.out_arg]
    o = t.load(c2.output, out_arg.view, out_arg.dtype, [starts[o_names[x]] for x in c2.out], [ext[o_names[x]] for x in c2.out])
    o = b.op("reshape", o, shape=(g.rows, g.value_size))
    d = b.op("reduce", b.op("mul", d_o, o), fn="sum", axes=(1,))
    b.emit("store", (b.op("reshape", d, shape=rows_tile(g)), *(starts[x] for x in batch + [row])), {"arg": t.arg(d_name, full_view(d_buffer.type.shape), d_buffer.type.dtype, True)})
    d_task = t.finish("attention_d", grid, pids, ("attention_d",))

    def inner_step(t: _Task, g, starts: dict, k_tile=None, v_t=None):
        """Shared by both tasks: X, P (exact), dO, dP and dS for one (row tile, col tile).

        k_tile and v_t, when given, were loaded outside the loop (the column tile is fixed)."""
        b = t.b
        x, q2d, coords = g.scores(t, {k: starts[k] for k in batch + [row]}, {col: starts[col]}, k_tile=k_tile)
        m_rows, l_rows = stat(t, g, m_name, starts), stat(t, g, l_name, starts)
        prob = g.p_exact(t, x, coords, m_rows, l_rows)
        local3 = _local(m3, {**starts, VALUE: b.index(0), s: b.index(0)})
        e3 = _local(m3, extents(g))
        d_o = _operand(t, c3, back.do_operand_c3, local3, e3, _letter(m3, row), _letter(m3, VALUE))
        if v_t is None:
            v_t = _operand(t, c3, 1 - back.do_operand_c3, local3, e3, _letter(m3, VALUE), _letter(m3, col))
        dp = b.op("reshape", b.op("dot", d_o, v_t, b.full((g.rows, g.cols), 0, DType.f32)), shape=x.type.shape)
        d_rows = stat(t, g, d_name, starts)
        tile = x.type.shape
        subs = {back.p_arg_t9: prob, back.dp_arg_t9: dp, **{a: _spread(b, d_rows, tile) for a in back.d_arg_t9}}
        ds = _replay(t, p, back.t9, subs, tile, coords)

        def widen(v):
            # Zero outside the tensor: past the last row the loaded statistics are 0, so P is
            # 0 / 0 there, and a NaN would spread through the sums over rows into every column.
            # The zeros go in before the widening cast, so the dots still see widened b16 values
            # (which is what lets them use b16 matrix instructions).
            zero = b.full(tile, 0.0, v.type.dtype)
            for axis, size in ((rank - 2, g.shape[row]), (rank - 1, g.shape[col])):
                coord = b.op("iota", coords[axis], shape=tile, axis=axis)
                inside = b.op("cmp", coord, b.op("broadcast", b.dim(size), shape=tile, dims=()), fn="lt")
                v = b.op("where", inside, v, zero)
            v = v if v.type.dtype is DType.f32 else b.op("cast", v, dtype=DType.f32)
            return b.op("reshape", v, shape=(g.rows, g.cols))

        return widen(prob), widen(ds), d_o, q2d

    def store(t, g, contract, mapping, value, rows_name, cols_name, starts):
        arg = contract.launch.task.args[contract.out_arg]
        local = _local(mapping, starts)
        _store(t, contract.output, arg.view, arg.dtype, contract.out, value, _letter(mapping, rows_name), _letter(mapping, cols_name), local, _local(mapping, extents(g)))

    # dK and dV: one tile of columns per instance, looping over rows. 64 x 64 tiles: this task
    # holds two accumulators and four score-sized tiles per iteration.
    g = _Builder(p, f, rows=BLOCK, cols=BLOCK)
    t = _Task()
    b = t.b
    pids, starts, grid = g.pids(t, col)
    starts.update({s: b.index(0), VALUE: b.index(0)})
    zeros_k = b.full((g.cols, g.head), 0.0, DType.f32)
    zeros_v = b.full((g.cols, g.value_size), 0.0, DType.f32)
    # The column tile is fixed for the whole loop: k (both orientations) and v load once, outside
    # it, so the lowering keeps them resident in shared memory.
    fixed = {**{x: starts[x] for x in batch}, col: starts[col], s: b.index(0)}
    k_tile = _operand(t, f.c1, 1, fixed, extents(g), s, col)
    k_rows = _operand(t, f.c1, 1, fixed, extents(g), col, s)
    v_t = _operand(t, c3, 1 - back.do_operand_c3, _local(m3, {**fixed, VALUE: b.index(0)}), _local(m3, extents(g)), _letter(m3, VALUE), _letter(m3, col))

    dq_arg = back.dq.launch.task.args[back.dq.out_arg]
    dq_atomic = dq_arg.dtype is DType.f32

    def kv_body(inner: TileBuilder, r0, carries):
        t.b = inner
        dk, dv = carries
        prob, ds, d_o, q2d = inner_step(t, g, {**starts, row: r0}, k_tile, v_t)
        dv = inner.op("dot", inner.op("permute", prob, perm=(1, 0)), d_o, dv)
        dk = inner.op("dot", inner.op("permute", ds, perm=(1, 0)), q2d, dk)
        if dq_atomic:
            # This tile's share of dQ = dS K, added atomically: every column tile contributes to
            # the same rows (FlashAttention-2), which saves recomputing S, P, dP and dS in a
            # separate task per row tile.
            part = inner.op("dot", ds, k_rows, inner.full((g.rows, g.head), 0.0, DType.f32))
            local = _local(maps["dq"], {**starts, row: r0, s: inner.index(0)})
            accumulate(t, back.dq, maps["dq"], part, row, s, local, extents(g))
        return [dk, dv]

    def accumulate(t, contract, mapping, value, rows_name, cols_name, local, ext):
        """atomic_add of a (rows, cols) tile into the contract's output (as _store, with addition)."""
        arg = contract.launch.task.args[contract.out_arg]
        term = contract.out
        rows, cols = _letter(mapping, rows_name), _letter(mapping, cols_name)
        unit = [x for x in term if x not in (rows, cols)]
        lext = _local(mapping, ext)
        shaped = t.b.op("reshape", value, shape=tuple([1] * len(unit) + [lext[rows], lext[cols]]))
        current = unit + [rows, cols]
        perm = tuple(current.index(x) for x in term)
        if perm != tuple(range(len(term))):
            shaped = t.b.op("permute", shaped, perm=perm)
        t.b.emit("atomic_add", (shaped, *(local[x] for x in term)), {"arg": t.arg(contract.output, arg.view, arg.dtype, True)})

    # Streamed q and dO: measured 2.34 -> 2.11 ms per layer for GPT-2 small on the RTX 3060.
    # (The forward is not streamed: with its larger shared memory it ran 0.61 -> 0.96 ms.)
    dk, dv = b.for_range(g.row_begin(b, starts[col]), b.dim(g.shape[row]), g.rows, [zeros_k, zeros_v], kv_body, stream=True)
    t.b = b
    store(t, g, back.dk, maps["dk"], dk, col, s, starts)
    store(t, g, back.dv, maps["dv"], dv, col, VALUE, starts)
    # 8 warps: this task holds two accumulators and four score-sized tiles per iteration, and
    # with 4 warps (32 registers per tile per thread) it spills. Nothing here needs warps
    # stacked along rows (its dots take operands from memory or staged accumulators).
    kv_task = t.finish("attention_dkv", grid, pids, ("attention_dkv",), warps=8)

    if dq_atomic:
        # dQ starts at zero; the dK/dV task adds into it.
        t = _Task()
        b = t.b
        gz = _Builder(p, f, rows=BLOCK, cols=BLOCK)
        pids, starts, grid = gz.pids(t, row)
        starts[s] = b.index(0)
        zero = b.full((gz.rows, gz.head), 0.0, DType.f32)
        store(t, gz, back.dq, maps["dq"], zero, row, s, starts)
        zero_task = t.finish("attention_dq_zero", grid, pids, ("attention_dq_zero",))
        return [d_task, zero_task, kv_task]

    # dQ: one tile of rows per instance, looping over columns (one accumulator: 128 rows).
    g = _Builder(p, f, rows=FORWARD_ROWS, cols=BLOCK)
    t = _Task()
    b = t.b
    pids, starts, grid = g.pids(t, row)
    starts.update({s: b.index(0), VALUE: b.index(0)})

    def q_body(inner: TileBuilder, c0, carries):
        t.b = inner
        (dq,) = carries
        _, ds, _, _ = inner_step(t, g, {**starts, col: c0})
        k2d = _operand(t, f.c1, 1, {**{x: starts[x] for x in batch}, col: c0, s: inner.index(0)}, extents(g), col, s)
        return [inner.op("dot", ds, k2d, dq)]

    (dq,) = b.for_range(b.index(0), g.col_end(b, starts[row]), g.cols, [b.full((g.rows, g.head), 0.0, DType.f32)], q_body)
    t.b = b
    store(t, g, back.dq, maps["dq"], dq, row, s, starts)
    q_task = t.finish("attention_dq", grid, pids, ("attention_dq",), warps=FORWARD_WARPS)
    return [d_task, kv_task, q_task]


def _letter(mapping: dict, name: str) -> str:
    return next(x for x, n in mapping.items() if n == name)
