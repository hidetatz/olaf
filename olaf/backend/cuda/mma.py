"""Lowering of a pipelined dot loop to mma.sync with cp.async and ldmatrix (sm_80 and later).

Matched pattern: a `for` with stages > 1 whose body loads one tile of each operand (through
permute/reshape to 2-D) and accumulates dot(a, b, acc) into its only carry, with b16 operands
and an f32 accumulator. Anything else returns None and takes the general path.

Per block: the accumulator tile (em x en) is split over a wm x wn grid of warps; each warp owns
(em/wm) x (en/wn) of it as m16n8 fragments. Each K step of the loop is one pipeline stage:
  1. every thread issues 16-byte cp.async copies of its share of the next stage's A and B tiles
     (zero-filled past the tensor edges, so the loop needs no tail code), then
  2. computes on the oldest ready stage: ldmatrix fragments, then mma.sync m16n8k16.
With S stages, S-1 copies are in flight while one stage is computed.

Shared memory keeps each operand in its global orientation (the contiguous axis innermost, as
16-byte chunks), and ldmatrix's transposing form produces the fragment layout when the global
layout is the other way round, so all four combinations of operand orientations (A or B
transposed, as in GPT-2's forward and both backward products) use the same copy code.
Chunk c of row r is stored at position c ^ ((r // rep) % min(chunks_per_row, 8)): the 8 rows that
one ldmatrix 8x8 read touches then fall into 8 different 16-byte bank groups, so the reads are
conflict-free for 64-byte rows (A tiles of 32 K values) and 256-byte rows (128-wide tiles) alike.

16-byte copies need every non-contiguous stride and the view offset to be multiples of 8
elements; the last chunk of a row copies only the elements that exist (cp.async zero-fills the
rest), so the contiguous extent may be ragged, and rows in a view's padding are zero-filled. Static violations reject the pattern; symbolic ones are recorded as
Lowered.aligned and checked at launch, with the general kernel as the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

from olaf.graph.dim import Dim
from olaf.graph.types import DType
from olaf.tile.ir import TOp, TValue, View

_STRUCT4_I32 = "!llvm.struct<(i32, i32, i32, i32)>"
_STRUCT4_F32 = "!llvm.struct<(f32, f32, f32, f32)>"
_PTX_TYPE = {DType.bf16: "bf16"}


@dataclass
class _Operand:
    arg: int
    view: View
    starts: list[TValue]  # one per view axis
    rows_axis: int  # view axis of the 2-D operand's rows
    cols_axis: int
    contiguous_cols: bool  # True: stored [rows][cols]; False: stored [cols][rows]
    rows: int
    cols: int

    @property
    def stored_rows(self) -> int:
        return self.rows if self.contiguous_cols else self.cols

    @property
    def row_elems(self) -> int:
        return self.cols if self.contiguous_cols else self.rows

    @property
    def bytes(self) -> int:
        return self.rows * self.cols * 2


def _trace_operand(e, region, value: TValue, producers: dict[TValue, TOp]) -> tuple[TOp, list[int]] | None:
    """Follow reshape/permute (and one exact b16 -> f32 cast) back to a load.

    Returns the load and the tile axes of the 2-D (rows, cols). A widening cast is exact, so an
    f32 dot of widened b16 values is the b16 dot with an f32 accumulator.
    """
    chain = []
    while value in producers and producers[value].kind in ("reshape", "permute", "cast"):
        op = producers[value]
        if op.kind == "cast" and not (op.inputs[0].type.dtype in _PTX_TYPE and op.attr("dtype") is DType.f32):
            return None
        if op.kind != "cast":
            chain.append(op)
        value = op.inputs[0]
    load = producers.get(value)
    if load is None or load.kind != "load":
        return None
    shape = list(load.attr("shape"))
    axes = list(range(len(shape)))
    for op in reversed(chain):
        if op.kind == "permute":
            perm = op.attr("perm")
            axes, shape = [axes[p] for p in perm], [shape[p] for p in perm]
        else:
            big = [(a, d) for a, d in zip(axes, shape) if d != 1]
            target = [d for d in op.attr("shape") if d != 1]
            if [d for _, d in big] != target:
                return None
            axes = [a for a, _ in big]
            shape = target
    if len(axes) != 2:
        return None
    return load, axes


def _operand(e, region, load: TOp, axes: list[int]) -> _Operand | None:
    view = e.task.args[load.attr("arg")].view
    if load.attr("dtype") not in _PTX_TYPE:
        return None
    rows_axis, cols_axis = axes
    shape = load.attr("shape")
    if view.strides[cols_axis] == 1:
        contiguous_cols = True
    elif view.strides[rows_axis] == 1:
        contiguous_cols = False
    else:
        return None
    contiguous = cols_axis if contiguous_cols else rows_axis
    if view.pad is not None and view.pad[contiguous] != (0, 0):
        return None  # padding inside a 16-byte chunk would need per-element fill
    # Chunk starts must be 16-byte aligned; a ragged contiguous extent is handled by partial copies.
    required: list[Dim] = [view.offset, *(s for i, s in enumerate(view.strides) if i != contiguous)]
    for d in required:
        if isinstance(d, int):
            if d % 8:
                return None
        elif d not in e.aligned:
            e.aligned.append(d)
    e.aligned_args.add(load.attr("arg"))  # its launch offset (tile.normalize) is checked at launch
    iv = region.params[0]
    for s in load.inputs:
        if s is not iv and s not in e.values:
            return None  # a start computed inside the loop: cannot be evaluated for a future stage
    return _Operand(load.attr("arg"), view, list(load.inputs), rows_axis, cols_axis, contiguous_cols, shape[rows_axis], shape[cols_axis])


def lower_matrix_loop(e, op: TOp, args):
    region = op.regions[0]
    lo, hi, step, *carries = args
    if len(carries) != 1 or len(region.ops) == 0:
        return None
    producers = {v: o for o in region.ops for v in o.outputs}
    dots = [o for o in region.ops if o.kind == "dot"]
    if len(dots) != 1 or region.results[0] is not dots[0].outputs[0] or dots[0].inputs[2] is not region.params[1]:
        return None
    dot = dots[0]
    a_val, b_val, _ = dot.inputs
    traced = [_trace_operand(e, region, v, producers) for v in (a_val, b_val)]
    if None in traced:
        return None
    used = {dot}
    for v in (a_val, b_val):
        while True:
            producer = producers[v]
            used.add(producer)
            if producer.kind == "load":
                break
            v = producer.inputs[0]
    # Inlined producers can leave scalar constants behind (their index arithmetic); they are pure.
    extra = set(region.ops) - used
    if any(o.kind not in ("full", "dim") or o.outputs[0].type.shape != () for o in extra):
        return None
    if dot.outputs[0].type.dtype is not DType.f32:
        return None
    a = _operand(e, region, *traced[0])
    b = _operand(e, region, *traced[1])
    if a is None or b is None or traced[0][0].attr("dtype") is not traced[1][0].attr("dtype"):
        return None
    em, ek = a.rows, a.cols
    en = b.cols
    if ek % 16 or b.rows != ek:
        return None
    grid = _warp_grid(e.task.warps, em, en)
    if grid is None:
        return None
    for operand in (a, b):
        chunks = operand.stored_rows * operand.row_elems // 8
        if chunks % e.threads or operand.row_elems % 8:
            return None
    return _MatrixLoop(e, a, b, em, en, ek, grid, op.attr("stages"), traced[0][0].attr("dtype")).emit(lo, hi, step, carries[0], region)


def _warp_grid(warps: int, em: int, en: int) -> tuple[int, int] | None:
    for wm in (2, 1, 4, 8, 16):
        if warps % wm:
            continue
        wn = warps // wm
        if em % (16 * wm) == 0 and en % (16 * wn) == 0:
            return wm, wn
    return None


class _MatrixLoop:
    def __init__(self, e, a: _Operand, b: _Operand, em: int, en: int, ek: int, grid: tuple[int, int], stages: int, dtype: DType) -> None:
        self.e, self.a, self.b = e, a, b
        self.em, self.en, self.ek = em, en, ek
        self.wm, self.wn = grid
        self.stages = stages
        self.dtype = dtype
        self.stage_bytes = a.bytes + b.bytes
        e.shared_bytes = max(e.shared_bytes, stages * self.stage_bytes)
        self.warp_m = em // self.wm
        self.warp_n = en // self.wn
        self.mi = self.warp_m // 16
        self.nj = self.warp_n // 8

    # -- entry-block (thread-id only) addressing ----------------------------------------------------

    def lane_values(self):
        e = self.e
        lane = e.entry(f"llvm.urem %tid, {e.i32(32)} : i32")
        warp = e.entry(f"llvm.udiv %tid, {e.i32(32)} : i32")
        wm_idx = e.entry(f"llvm.urem {warp}, {e.i32(self.wm)} : i32")
        wn_idx = e.entry(f"llvm.udiv {warp}, {e.i32(self.wm)} : i32")
        return lane, wm_idx, wn_idx

    def swizzled(self, operand: _Operand, row: str, chunk: str) -> str:
        """Byte offset of (stored row, 16-byte chunk) inside one operand stage, in the entry block."""
        e = self.e
        per_row = operand.row_elems // 8
        rep = max(1, 8 // per_row)
        span = min(per_row, 8)
        r = e.entry(f"llvm.udiv {row}, {e.i32(rep)} : i32") if rep > 1 else row
        r = e.entry(f"llvm.urem {r}, {e.i32(span)} : i32")
        c = e.entry(f"llvm.xor {chunk}, {r} : i32")
        row_bytes = e.entry(f"llvm.mul {row}, {e.i32(operand.row_elems * 2)} : i32")
        return e.entry(f"llvm.add {row_bytes}, {e.entry(f'llvm.mul {c}, {e.i32(16)} : i32')} : i32")

    def copy_plan(self, operand: _Operand):
        """Per copy slot: (stored row, chunk index, smem byte offset), entry block."""
        e = self.e
        per_row = operand.row_elems // 8
        plan = []
        for i in range(operand.stored_rows * per_row // e.threads):
            c = e.entry(f"llvm.add %tid, {e.i32(i * e.threads)} : i32") if i else "%tid"
            row = e.entry(f"llvm.udiv {c}, {e.i32(per_row)} : i32")
            chunk = e.entry(f"llvm.urem {c}, {e.i32(per_row)} : i32")
            plan.append((row, chunk, self.swizzled(operand, row, chunk)))
        return plan

    def fragment_offsets(self, lane: str, wm_idx: str, wn_idx: str):
        """Byte offsets of each lane's ldmatrix row address, for A[i][kk] and B[p][kk]."""
        e = self.e
        r = e.entry(f"llvm.urem {lane}, {e.i32(8)} : i32")
        j = e.entry(f"llvm.udiv {lane}, {e.i32(8)} : i32")
        j_lo = e.entry(f"llvm.urem {j}, {e.i32(2)} : i32")
        j_hi = e.entry(f"llvm.udiv {j}, {e.i32(2)} : i32")
        l16 = e.entry(f"llvm.urem {lane}, {e.i32(16)} : i32")
        h16 = e.entry(f"llvm.udiv {lane}, {e.i32(16)} : i32")
        m_base = e.entry(f"llvm.mul {wm_idx}, {e.i32(self.warp_m)} : i32")
        n_base = e.entry(f"llvm.mul {wn_idx}, {e.i32(self.warp_n)} : i32")

        def add(*terms) -> str:
            total = None
            for t in terms:
                if isinstance(t, int):
                    if t == 0:
                        continue
                    t = e.i32(t)
                total = t if total is None else e.entry(f"llvm.add {total}, {t} : i32")
            return total if total is not None else e.i32(0)

        def times8(x: str) -> str:
            return e.entry(f"llvm.mul {x}, {e.i32(8)} : i32")

        def chunk_of(x: str) -> str:
            return e.entry(f"llvm.udiv {x}, {e.i32(8)} : i32")

        a_offsets, b_offsets = [], []
        for i in range(self.mi):
            per_k = []
            for kk in range(self.ek // 16):
                if self.a.contiguous_cols:  # [M][K]: plain ldmatrix
                    row = add(m_base, i * 16, l16)
                    col = add(kk * 16, times8(h16))
                    per_k.append(self.swizzled(self.a, row, chunk_of(col)))
                else:  # [K][M]: transposing ldmatrix
                    k = add(kk * 16, times8(j_hi), r)
                    m = add(m_base, i * 16, times8(j_lo))
                    per_k.append(self.swizzled(self.a, k, chunk_of(m)))
            a_offsets.append(per_k)
        for p in range(self.nj // 2):
            per_k = []
            for kk in range(self.ek // 16):
                if self.b.contiguous_cols:  # [K][N]: transposing ldmatrix
                    k = add(kk * 16, times8(j_lo), r)
                    n = add(n_base, p * 16, times8(j_hi))
                    per_k.append(self.swizzled(self.b, k, chunk_of(n)))
                else:  # [N][K]: plain ldmatrix
                    n = add(n_base, p * 16, times8(j_hi), r)
                    k = add(kk * 16, times8(j_lo))
                    per_k.append(self.swizzled(self.b, n, chunk_of(k)))
            b_offsets.append(per_k)
        return a_offsets, b_offsets

    def accumulator_layout(self, lane: str, wm_idx: str, wn_idx: str) -> list[list[str]]:
        key = ("mma-acc", self.em, self.en, self.wm, self.wn, self.e.threads)
        cache = self.e.coords_cache
        if key in cache:
            return cache[key]
        e = self.e
        g = e.entry(f"llvm.udiv {lane}, {e.i32(4)} : i32")
        t = e.entry(f"llvm.urem {lane}, {e.i32(4)} : i32")
        t2 = e.entry(f"llvm.mul {t}, {e.i32(2)} : i32")
        m_base = e.entry(f"llvm.mul {wm_idx}, {e.i32(self.warp_m)} : i32")
        n_base = e.entry(f"llvm.mul {wn_idx}, {e.i32(self.warp_n)} : i32")
        coords = []
        for i in range(self.mi):
            for j in range(self.nj):
                for q in range(4):
                    row = e.entry(f"llvm.add {m_base}, {e.entry(f'llvm.add {g}, {e.i32(i * 16 + 8 * (q // 2))} : i32')} : i32")
                    col = e.entry(f"llvm.add {n_base}, {e.entry(f'llvm.add {t2}, {e.i32(j * 8 + q % 2)} : i32')} : i32")
                    coords.append([row, col])
        cache[key] = coords
        return coords

    # -- loop body pieces -----------------------------------------------------------------------

    def issue(self, operand: _Operand, plan, stage_base: str, k: str, iv: TValue) -> None:
        e = self.e
        base = e.emit("llvm.mlir.addressof @smem : !llvm.ptr<3>")
        offset_in_stage = 0 if operand is self.a else self.a.bytes
        starts = [k if s is iv else e.values[s].regs[0] for s in operand.starts]
        contiguous = operand.cols_axis if operand.contiguous_cols else operand.rows_axis
        strided = operand.rows_axis if operand.contiguous_cols else operand.cols_axis
        for row, chunk, smem_offset in plan:
            coords = list(starts)
            coords[strided] = e.emit(f"llvm.add {starts[strided]}, {row} : i32")
            coords[contiguous] = e.emit(f"llvm.add {starts[contiguous]}, {e.emit(f'llvm.mul {chunk}, {e.i32(8)} : i32')} : i32")
            offset, inside, padded = e.view_address(operand.view, coords)
            inside = inside or e.const(True, "i1")
            if padded is not None:
                live = e.emit(f"llvm.xor {padded}, {e.const(True, 'i1')} : i1")
                inside = e.emit(f"llvm.and {inside}, {live} : i1")
            safe = e.emit(f"llvm.select {inside}, {offset}, {e.i32(0)} : i1, i32")
            # Copy only the elements before the end of the contiguous axis; cp.async fills the
            # rest of the 16 bytes with zeros, which add nothing to the dot.
            remaining = e.emit(f"llvm.sub {e.dim(operand.view.shape[contiguous])}, {coords[contiguous]} : i32")
            count = e.emit(f"llvm.intr.smin({remaining}, {e.i32(8)}) : (i32, i32) -> i32")
            full = e.emit(f"llvm.mul {count}, {e.i32(2)} : i32")
            size = e.emit(f"llvm.select {inside}, {full}, {e.i32(0)} : i1, i32")
            src = e.global_ptr(operand.arg, safe, DType.bf16)
            byte = e.emit(f"llvm.add {stage_base}, {e.emit(f'llvm.add {smem_offset}, {e.i32(offset_in_stage)} : i32')} : i32")
            dst = e.emit(f"llvm.getelementptr {base}[{byte}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8")
            e.raw(f"nvvm.cp.async.shared.global {dst}, {src}, 16, cache = cg, {size} : !llvm.ptr<3>, !llvm.ptr<1>, i32")

    def ldmatrix(self, byte: str, trans: bool) -> list[str]:
        e = self.e
        base = e.emit("llvm.mlir.addressof @smem : !llvm.ptr<3>")
        ptr = e.emit(f"llvm.getelementptr {base}[{byte}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8")
        layout = "col" if trans else "row"
        result = e.emit(
            f"nvvm.ldmatrix {ptr} {{eltType = #nvvm.ld_st_matrix_elt_type<b16>, layout = #nvvm.mma_layout<{layout}>, num = 4 : i32, "
            f"shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>}} : (!llvm.ptr<3>) -> {_STRUCT4_I32}"
        )
        return [e.emit(f"llvm.extractvalue {result}[{i}] : {_STRUCT4_I32}") for i in range(4)]

    def compute(self, stage_base: str, acc: list[str], a_offsets, b_offsets) -> list[str]:
        e = self.e
        acc = list(acc)
        b_base = e.emit(f"llvm.add {stage_base}, {e.i32(self.a.bytes)} : i32")
        ptx = _PTX_TYPE[self.dtype]
        for kk in range(self.ek // 16):
            a_frags = [self.ldmatrix(e.emit(f"llvm.add {stage_base}, {a_offsets[i][kk]} : i32"), not self.a.contiguous_cols) for i in range(self.mi)]
            b_frags = [self.ldmatrix(e.emit(f"llvm.add {b_base}, {b_offsets[p][kk]} : i32"), self.b.contiguous_cols) for p in range(self.nj // 2)]
            for i in range(self.mi):
                for j in range(self.nj):
                    b_regs = b_frags[j // 2][(j % 2) * 2 : (j % 2) * 2 + 2]
                    base = (i * self.nj + j) * 4
                    c = acc[base : base + 4]
                    d = e.emit(
                        f"nvvm.mma.sync A[{', '.join(a_frags[i])}] B[{', '.join(b_regs)}] C[{', '.join(c)}] "
                        f"{{layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<{ptx}>, "
                        f"multiplicandBPtxType = #nvvm.mma_type<{ptx}>, shape = #nvvm.shape<m = 16, n = 8, k = 16>}} : (i32, i32, f32) -> {_STRUCT4_F32}"
                    )
                    acc[base : base + 4] = [e.emit(f"llvm.extractvalue {d}[{q}] : {_STRUCT4_F32}") for q in range(4)]
        return acc

    def emit(self, lo, hi, step, carry, region):
        from olaf.backend.cuda.lower import _Value

        e = self.e
        lane, wm_idx, wn_idx = self.lane_values()
        layout = self.accumulator_layout(lane, wm_idx, wn_idx)
        like = _Value([], False, carry.type, layout, None)
        if carry.coords is layout:
            acc = list(carry.regs)
        elif len(set(carry.regs)) == 1:
            acc = [carry.regs[0]] * len(layout)  # a splat (the zero init): no data to move
        else:
            acc = e.relayout(carry, like).regs
        a_plan, b_plan = self.copy_plan(self.a), self.copy_plan(self.b)
        a_offsets, b_offsets = self.fragment_offsets(lane, wm_idx, wn_idx)
        iv = region.params[0]
        s = self.stages

        e.barrier()  # shared memory may still be read by an earlier exchange
        for stage in range(s - 1):
            k = e.emit(f"llvm.add {lo.regs[0]}, {e.emit(f'llvm.mul {step.regs[0]}, {e.i32(stage)} : i32')} : i32")
            base = e.i32(stage * self.stage_bytes)
            self.issue(self.a, a_plan, base, k, iv)
            self.issue(self.b, b_plan, base, k, iv)
            e.raw("nvvm.cp.async.commit.group")

        header, body, done = e.label(), e.label(), e.label()
        types = ", ".join(["i32", "i32", *(["f32"] * len(acc))])
        e.raw(f"llvm.br {header}({', '.join([lo.regs[0], e.i32(0), *acc])} : {types})")
        k_now, stage_now = e.tmp(), e.tmp()
        acc_params = [e.tmp() for _ in acc]
        signature = ", ".join([f"{k_now}: i32", f"{stage_now}: i32", *(f"{p}: f32" for p in acc_params)])
        e.lines.append(f"  {header}({signature}):")
        cond = e.emit(f"llvm.icmp \"slt\" {k_now}, {hi.regs[0]} : i32")
        e.raw(f"llvm.cond_br {cond}, {body}, {done}")
        e.lines.append(f"  {body}:")
        # The oldest group is complete once at most s-2 remain in flight; the barrier then also
        # guarantees every warp finished reading the stage that the next copies overwrite.
        e.raw(f"nvvm.cp.async.wait.group {s - 2}")
        e.barrier()
        ahead = e.emit(f"llvm.add {k_now}, {e.emit(f'llvm.mul {step.regs[0]}, {e.i32(s - 1)} : i32')} : i32")
        fill_stage = e.emit(f"llvm.urem {e.emit(f'llvm.add {stage_now}, {e.i32(s - 1)} : i32')}, {e.i32(s)} : i32")
        fill_base = e.emit(f"llvm.mul {fill_stage}, {e.i32(self.stage_bytes)} : i32")
        self.issue(self.a, a_plan, fill_base, ahead, iv)
        self.issue(self.b, b_plan, fill_base, ahead, iv)
        e.raw("nvvm.cp.async.commit.group")
        read_base = e.emit(f"llvm.mul {stage_now}, {e.i32(self.stage_bytes)} : i32")
        new_acc = self.compute(read_base, acc_params, a_offsets, b_offsets)
        k_next = e.emit(f"llvm.add {k_now}, {step.regs[0]} : i32")
        stage_next = e.emit(f"llvm.urem {e.emit(f'llvm.add {stage_now}, {e.i32(1)} : i32')}, {e.i32(s)} : i32")
        e.raw(f"llvm.br {header}({', '.join([k_next, stage_next, *new_acc])} : {types})")
        e.lines.append(f"  {done}:")
        e.raw("nvvm.cp.async.wait.group 0")
        e.barrier()
        return [_Value(acc_params, False, carry.type, layout, None)]
