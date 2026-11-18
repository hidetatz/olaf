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

import math
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
    # The view axes the strided side's index runs over, outermost first: one axis, or several
    # when a tile's rows span labels that are not one affine axis (implicit GEMM rows n, h, w;
    # tile/from_graph.py). A copy's row number is split into them by division.
    strided_axes: tuple[tuple[int, int], ...] = ()  # (view axis, tile extent)

    @property
    def stored_rows(self) -> int:
        return self.rows if self.contiguous_cols else self.cols

    @property
    def row_elems(self) -> int:
        return self.cols if self.contiguous_cols else self.rows

    @property
    def bytes(self) -> int:
        return self.rows * self.cols * 2


def _trace_operand(e, region, value: TValue, producers: dict[TValue, TOp]) -> tuple[TOp, list[tuple[int, ...]]] | None:
    """Follow reshape/permute (and one exact b16 -> f32 cast) back to a load.

    Returns the load and, for the 2-D (rows, cols), the tile axes each is made of (outermost
    first; usually one). A widening cast is exact, so an f32 dot of widened b16 values is the
    b16 dot with an f32 accumulator.
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
    axes: list[tuple[int, ...]] = [(a,) for a in range(len(shape))]
    for op in reversed(chain):
        if op.kind == "permute":
            perm = op.attr("perm")
            axes, shape = [axes[p] for p in perm], [shape[p] for p in perm]
        else:
            big = [(a, d) for a, d in zip(axes, shape) if d != 1]
            target = [d for d in op.attr("shape") if d != 1]
            # Each target axis is a run of consecutive source axes (a merge, row-major).
            merged, i = [], 0
            for d in target:
                group, size = (), 1
                while i < len(big) and size < d:
                    group, size = group + big[i][0], size * big[i][1]
                    i += 1
                if size != d:
                    return None
                merged.append(group)
            if i != len(big):
                return None
            axes, shape = merged, target
    if len(axes) != 2:
        return None
    return load, axes


def _operand(e, region, load: TOp, axes: list[int]) -> _Operand | None:
    view = e.task.args[load.attr("arg")].view
    if load.attr("dtype") not in _PTX_TYPE:
        return None
    rows_axes, cols_axes = axes
    rows_axis, cols_axis = rows_axes[-1], cols_axes[-1]
    shape = load.attr("shape")
    if len(cols_axes) == 1 and view.strides[cols_axis] == 1:
        contiguous_cols = True
    elif len(rows_axes) == 1 and view.strides[rows_axis] == 1:
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
        if not _evaluable(e, s, iv):
            return None  # a start computed inside the loop from more than index arithmetic on it
    rows, cols = (math.prod(shape[a] for a in g) for g in axes)
    return _Operand(load.attr("arg"), view, list(load.inputs), rows_axis, cols_axis, contiguous_cols, rows, cols, tuple((a, shape[a]) for a in (rows_axes if contiguous_cols else cols_axes)))


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
        e.reserve(stages * self.stage_bytes)
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
            plan.append((row, chunk, self.swizzled(operand, row, chunk), self.split_row(operand, row)))
        return plan

    def split_row(self, operand: _Operand, row: str) -> list[tuple[int, str]]:
        """A stored row's index along each strided view axis (entry block): (axis, index)."""
        axes = operand.strided_axes
        if len(axes) <= 1:
            return []
        e = self.e
        parts, inner = [], 1
        for k in range(len(axes) - 1, -1, -1):
            axis, extent = axes[k]
            q = e.entry(f"llvm.udiv {row}, {e.i32(inner)} : i32") if inner > 1 else row
            parts.append((axis, e.entry(f"llvm.urem {q}, {e.i32(extent)} : i32") if k else q))
            inner *= extent
        return parts

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

    def issue(self, operand: _Operand, plan, stage_base: str, k: str, iv: TValue, offset_in_stage: int | None = None) -> None:
        """cp.async copies of operand's tile into shared memory at stage_base (+ its offset in a stage)."""
        e = self.e
        base = e.smem_base()
        if offset_in_stage is None:
            offset_in_stage = 0 if operand is self.a else self.a.bytes
        starts = [_start(e, s, iv, k) for s in operand.starts]
        contiguous = operand.cols_axis if operand.contiguous_cols else operand.rows_axis
        strided = operand.rows_axis if operand.contiguous_cols else operand.cols_axis
        for row, chunk, smem_offset, parts in plan:
            coords = list(starts)
            if parts:
                for axis, index in parts:
                    coords[axis] = e.emit(f"llvm.add {starts[axis]}, {index} : i32")
            else:
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
        base = e.smem_base()
        ptr = e.emit(f"llvm.getelementptr {base}[{byte}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8")
        layout = "col" if trans else "row"
        result = e.emit(
            f"nvvm.ldmatrix {ptr} {{eltType = #nvvm.ld_st_matrix_elt_type<b16>, layout = #nvvm.mma_layout<{layout}>, num = 4 : i32, "
            f"shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>}} : (!llvm.ptr<3>) -> {_STRUCT4_I32}"
        )
        return [e.emit(f"llvm.extractvalue {result}[{i}] : {_STRUCT4_I32}") for i in range(4)]

    def compute(self, stage_base: str, acc: list[str], a_offsets, b_offsets, a_registers=None, b_base: str | None = None) -> list[str]:
        """mma over one stage. a_registers[i][kk], when given, are A fragments already in registers."""
        e = self.e
        acc = list(acc)
        b_base = b_base or e.emit(f"llvm.add {stage_base}, {e.i32(self.a.bytes)} : i32")
        ptx = _PTX_TYPE[self.dtype]
        for kk in range(self.ek // 16):
            if a_registers is not None:
                a_frags = [a_registers[i][kk] for i in range(self.mi)]
            else:
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
        from olaf.backend.cuda.lower import MmaAcc, _Value

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
        return [_Value(acc_params, False, carry.type, layout, None, MmaAcc(self.em, self.en, self.wm, self.wn, 0, 1, 2))]


def exactly_b16(e, value: TValue) -> bool:
    """value holds b16 values exactly: a bf16 tile, or f32 widened from one (through permute/reshape)."""
    while True:
        if value.type.dtype in _PTX_TYPE:
            return True
        op = e.producers.get(value)
        if op is None or op.kind not in ("cast", "permute", "reshape"):
            return False
        if op.kind == "cast" and not (op.inputs[0].type.dtype in _PTX_TYPE and op.attr("dtype") is DType.f32):
            return False
        value = op.inputs[0]


def lower_register_dot(e, dot: TOp, a, b, acc):
    """dot(a, b, acc) inside a task body (not a pipelined loop), with mma.sync.

    Each operand reaches the fragments the cheapest way it can:
    - resident: a load that an enclosing loop copied to shared memory once (make_resident);
    - a plain b16 load from global memory (through widening casts, permutes, reshapes) is copied
      straight into shared memory with 16-byte async copies, as in the pipelined loop;
    - an A operand that is itself an accumulator of the same warp rows (warps stacked along M,
      one warp per 16 rows) becomes A fragments in registers: the m16n8 accumulator of columns
      16k..16k+15 holds exactly the m16k16 A fragment of that k block (FlashAttention's P);
    - anything else is staged from registers into shared memory.
    Returns None when shapes or dtypes do not fit, for the general path.
    """
    from olaf.backend.cuda.lower import MmaAcc, _Value

    a_t, b_t = dot.inputs[0], dot.inputs[1]
    if acc.type.dtype is not DType.f32 or not (exactly_b16(e, a_t) and exactly_b16(e, b_t)):
        return None
    m, k = a.type.shape
    n = b.type.shape[1]
    if k % 16 or n % 16 or m % 16:
        return None
    # Warps along M first: every warp then owns whole rows, which row reductions and the
    # register A path need.
    warps = e.task.warps
    grid = (warps, 1) if m % (16 * warps) == 0 else _warp_grid(warps, m, n)
    if grid is None:
        return None
    a_found, b_found = _global_operand(e, a_t), _global_operand(e, b_t)
    a_op = a_found[0] if a_found else _Operand(-1, None, [], 0, 1, _natural(a) != "transposed", m, k)
    b_op = b_found[0] if b_found else _Operand(-1, None, [], 0, 1, _natural(b) != "transposed", k, n)
    loop = _MatrixLoop(e, a_op, b_op, m, n, k, grid, 1, DType.bf16)
    lane, wm_idx, wn_idx = loop.lane_values()
    layout = loop.accumulator_layout(lane, wm_idx, wn_idx)
    structure = MmaAcc(m, n, grid[0], grid[1], 0, 1, 2)
    if acc.coords is layout:
        acc_regs = list(acc.regs)
    elif len(set(acc.regs)) == 1:
        acc_regs = [acc.regs[0]] * len(layout)
    else:
        acc_regs = e.relayout(acc, _Value([], False, acc.type, layout, None, structure)).regs
    a_registers = _register_a(e, a, grid, m, k) if a_found is None else None
    # Byte offsets relative to scratch (resident copies sit before it: negative offsets).
    offsets = {}
    cursor = 0
    ready = set()  # operands already in shared memory (resident or streamed by the loop)
    for name, found, operand in (("a", a_found, a_op), ("b", b_found, b_op)):
        if found is not None and found[1] in e.resident:
            offsets[name] = e.i32(e.resident[found[1]] - e.scratch_base)
            ready.add(name)
        elif found is not None and found[1] in e.streamed:
            offsets[name] = e.streamed[found[1]]
            ready.add(name)
        elif name == "a" and a_registers is not None:
            offsets[name] = e.i32(0)
            ready.add(name)
        else:
            offsets[name] = e.i32(cursor)
            cursor += operand.bytes
    e.reserve(cursor)
    if len(ready) < 2:
        e.barrier()
    copied = False
    for name, found, operand, value in (("a", a_found, a_op, a), ("b", b_found, b_op, b)):
        if name in ready:
            continue
        if found is not None:
            loop.issue(operand, loop.copy_plan(operand), offsets[name], None, None, offset_in_stage=0)
            copied = True
        elif _natural(value) is not None:
            _stage_accumulator(e, operand, value, offsets[name])
        else:
            _stage(e, loop, operand, value, offsets[name])
    if copied:
        # (Waits for every group, including a loop's prefetch: copying here defeats streaming,
        # which is why loops stream or keep resident every operand they can.)
        e.raw("nvvm.cp.async.commit.group")
        e.raw("nvvm.cp.async.wait.group 0")
    if len(ready) < 2:
        e.barrier()
    a_offsets, b_offsets = loop.fragment_offsets(lane, wm_idx, wn_idx)
    out = loop.compute(offsets["a"], acc_regs, a_offsets, b_offsets, a_registers, offsets["b"])
    e.barrier()
    return _Value(out, False, acc.type, layout, None, structure)


def make_resident(e, region) -> None:
    """Copy the dot operands a loop body loads from outside the loop into shared memory, once.

    The fused attention loops multiply the same q (forward) or k and v (backward) tile in every
    iteration; staging them anew each time costs a copy, a wait and a barrier per dot. Resident
    copies sit at the start of shared memory and scratch moves past them while the body is
    emitted (Emitter.scratch_base).
    """
    inside = set()

    def walk(body):
        for op in body.ops:
            inside.add(id(op))
            for r in op.regions:
                walk(r)

    walk(region)
    wanted = []
    seen = set()

    def scan(body):
        for op in body.ops:
            if op.kind == "dot":
                for value in op.inputs[:2]:
                    found = _global_operand(e, value)
                    if found is None:
                        continue
                    operand, load = found
                    if id(load) in inside or id(load) in seen or load in e.resident or any(s not in e.values for s in load.inputs):
                        continue
                    seen.add(id(load))
                    wanted.append((operand, load))
            for r in op.regions:
                scan(r)

    scan(region)
    if not wanted:
        return
    total = 0
    placed = []
    for operand, load in wanted:
        placed.append((operand, load, total))
        total += -(-operand.bytes // 16) * 16
    e.reserve(total)
    e.barrier()
    # Only the copy helpers are used: no pipeline stages, so nothing is reserved for them.
    helper = _MatrixLoop(e, placed[0][0], placed[0][0], placed[0][0].rows, placed[0][0].cols, 16, (1, 1), 0, DType.bf16)
    for operand, load, offset in placed:
        helper.issue(operand, helper.copy_plan(operand), e.i32(offset), None, None, offset_in_stage=0)
        e.resident[load] = e.scratch_base + offset
    e.raw("nvvm.cp.async.commit.group")
    e.raw("nvvm.cp.async.wait.group 0")
    e.barrier()
    e.scratch_base += total


def _global_operand(e, value: TValue):
    """(the dot operand as an _Operand copied from global memory, its load op) for a plain b16 load, else None."""
    producers = e.producers
    traced = _trace_operand(e, None, value, producers)
    if traced is None:
        return None
    load, axes = traced
    view = e.task.args[load.attr("arg")].view
    if load.attr("dtype") not in _PTX_TYPE or view.pad is not None or e.task.args[load.attr("arg")].writes:
        return None
    if any(len(g) != 1 for g in axes):
        return None
    rows_axis, cols_axis = (g[0] for g in axes)
    contiguous_cols = view.strides[cols_axis] == 1
    if not contiguous_cols and view.strides[rows_axis] != 1:
        return None
    contiguous = cols_axis if contiguous_cols else rows_axis
    for d in [view.offset, *(st for i, st in enumerate(view.strides) if i != contiguous)]:
        if not isinstance(d, int) or d % 8:
            return None  # the launch-time alignment check covers only the pipelined loop's arguments
    if e.task.args[load.attr("arg")] is not None:
        e.aligned_args.add(load.attr("arg"))
    shape = load.attr("shape")
    operand = _Operand(load.attr("arg"), view, list(load.inputs), rows_axis, cols_axis, contiguous_cols, shape[rows_axis], shape[cols_axis])
    chunks = operand.stored_rows * operand.row_elems // 8
    return (operand, load) if chunks % e.threads == 0 and operand.row_elems % 8 == 0 else None


def _register_a(e, a, grid, m: int, k: int):
    """A fragments from an accumulator in registers (see lower_register_dot), or None."""
    from olaf.backend.cuda.lower import MmaAcc

    s = a.structure
    if not isinstance(s, MmaAcc) or a.type.rank != 2 or (s.row_axis, s.col_axis) != (0, 1):
        return None
    if (s.em, s.en, s.wm, s.wn) != (m, k, grid[0], 1) or grid[1] != 1:
        return None
    nj = s.nj

    def pack(x: str, y: str) -> str:
        x, y = (r if a.type.dtype is DType.bf16 else e.emit(f"llvm.fptrunc {r} : f32 to bf16") for r in (x, y))
        vec = e.emit("llvm.mlir.undef : vector<2xbf16>")
        vec = e.emit(f"llvm.insertelement {x}, {vec}[{e.i32(0)} : i32] : vector<2xbf16>")
        vec = e.emit(f"llvm.insertelement {y}, {vec}[{e.i32(1)} : i32] : vector<2xbf16>")
        return e.emit(f"llvm.bitcast {vec} : vector<2xbf16> to i32")

    frags = []
    for i in range(s.mi):
        per_k = []
        for kk in range(k // 16):
            lo = (i * nj + 2 * kk) * 4
            hi = (i * nj + 2 * kk + 1) * 4
            r = a.regs
            per_k.append([pack(r[lo], r[lo + 1]), pack(r[lo + 2], r[lo + 3]), pack(r[hi], r[hi + 1]), pack(r[hi + 2], r[hi + 3])])
        frags.append(per_k)
    return frags


def _stage(e, loop: _MatrixLoop, operand: _Operand, value, base: int) -> None:
    """Write a register tile as a row-major b16 operand into shared memory, with the swizzle."""
    shape = value.type.shape
    cols = shape[1]
    per_row = cols // 8
    rep = max(1, 8 // per_row)
    span = min(per_row, 8)
    smem = e.smem_base()
    for slot, coords in enumerate(e.layout_of(value)):
        row, col = coords if len(coords) == 2 else (e.i32(0), coords[0])
        reg = value.regs[slot] if not value.replicated else value.regs[0]
        if value.type.dtype is DType.f32:
            reg = e.emit(f"llvm.fptrunc {reg} : f32 to bf16")  # exact: the value is a widened bf16
        chunk = e.emit(f"llvm.udiv {col}, {e.i32(8)} : i32")
        within = e.emit(f"llvm.urem {col}, {e.i32(8)} : i32")
        r = e.emit(f"llvm.udiv {row}, {e.i32(rep)} : i32") if rep > 1 else row
        swz = e.emit(f"llvm.xor {chunk}, {e.emit(f'llvm.urem {r}, {e.i32(span)} : i32')} : i32")
        byte = e.emit(f"llvm.add {e.emit(f'llvm.mul {row}, {e.i32(cols * 2)} : i32')}, {e.emit(f'llvm.mul {swz}, {e.i32(16)} : i32')} : i32")
        byte = e.emit(f"llvm.add {byte}, {e.emit(f'llvm.add {e.emit(f'llvm.mul {within}, {e.i32(2)} : i32')}, {base} : i32')} : i32")
        ptr = e.emit(f"llvm.getelementptr {smem}[{byte}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8")
        guard = e.valid_of(value, slot)
        e.guarded(guard, lambda reg=reg, ptr=ptr: e.raw(f"llvm.store {reg}, {ptr} : bf16, !llvm.ptr<3>"))


def _natural(value) -> str | None:
    """"direct" or "transposed" for a 2-D accumulator value (as a dot operand), else None."""
    from olaf.backend.cuda.lower import MmaAcc

    st = value.structure
    if not isinstance(st, MmaAcc) or value.type.rank != 2:
        return None
    return "direct" if (st.row_axis, st.col_axis) == (0, 1) else "transposed"


def _stage_accumulator(e, operand: _Operand, value, base: int) -> None:
    """Store an accumulator tile in its natural [row][col] orientation, two columns per store.

    Slots q and q+1 of a fragment are adjacent columns of one row, so they pack into one 4-byte
    bf16x2 store; for a transposed operand this layout is [K][M] (or [N][K]), which the
    transposing ldmatrix reads as the operand.
    """
    st = value.structure
    row_elems = operand.row_elems
    per_row = row_elems // 8
    rep = max(1, 8 // per_row)
    span = min(per_row, 8)
    smem = e.smem_base()
    for slot in range(0, len(value.regs), 2):
        coords = value.coords[slot]
        row, col = coords[st.row_axis], coords[st.col_axis]
        x, y = value.regs[slot], value.regs[slot + 1]
        if value.type.dtype is DType.f32:
            x, y = (e.emit(f"llvm.fptrunc {r} : f32 to bf16") for r in (x, y))
        vec = e.emit("llvm.mlir.undef : vector<2xbf16>")
        vec = e.emit(f"llvm.insertelement {x}, {vec}[{e.i32(0)} : i32] : vector<2xbf16>")
        vec = e.emit(f"llvm.insertelement {y}, {vec}[{e.i32(1)} : i32] : vector<2xbf16>")
        word = e.emit(f"llvm.bitcast {vec} : vector<2xbf16> to i32")
        chunk = e.emit(f"llvm.udiv {col}, {e.i32(8)} : i32")
        within = e.emit(f"llvm.urem {col}, {e.i32(8)} : i32")
        r = e.emit(f"llvm.udiv {row}, {e.i32(rep)} : i32") if rep > 1 else row
        swz = e.emit(f"llvm.xor {chunk}, {e.emit(f'llvm.urem {r}, {e.i32(span)} : i32')} : i32")
        byte = e.emit(f"llvm.add {e.emit(f'llvm.mul {row}, {e.i32(row_elems * 2)} : i32')}, {e.emit(f'llvm.mul {swz}, {e.i32(16)} : i32')} : i32")
        byte = e.emit(f"llvm.add {byte}, {e.emit(f'llvm.add {e.emit(f'llvm.mul {within}, {e.i32(2)} : i32')}, {base} : i32')} : i32")
        ptr = e.emit(f"llvm.getelementptr {smem}[{byte}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8")
        e.raw(f"llvm.store {word}, {ptr} : i32, !llvm.ptr<3>")


def make_streamed(e, region, lo) -> list:
    """Double-buffer the dot operands a loop body loads at the loop variable, one iteration ahead.

    The fused attention loops load a new k and v tile (forward) or q and dO tile (backward)
    every iteration. Copying it right before its dot leaves the copy latency exposed; instead the
    copy of iteration i+1 is issued at the top of iteration i (prefetch_streamed) into the other
    of two buffers, and only iteration i's copy is waited for. Before the loop, iteration 0's
    copies are issued. Returns the plan for prefetch_streamed.
    """
    iv = region.params[0]
    inside = {id(op) for op in region.ops}
    plan, seen = [], set()
    for op in region.ops:
        if op.kind != "dot":
            continue
        for value in op.inputs[:2]:
            found = _global_operand(e, value)
            if found is None:
                continue
            operand, load = found
            if id(load) not in inside or id(load) in seen:
                continue
            if any(_start(e, s, iv, "") is None for s in load.inputs):
                continue  # a start computed inside the body: not known an iteration ahead
            seen.add(id(load))
            plan.append([operand, load, 0])
    if not plan:
        return []
    total = 0
    for item in plan:
        item[2] = total
        total += 2 * (-(-item[0].bytes // 16) * 16)
    e.reserve(total)
    e.barrier()
    helper = _MatrixLoop(e, plan[0][0], plan[0][0], plan[0][0].rows, plan[0][0].cols, 16, (1, 1), 0, DType.bf16)
    for operand, load, offset in plan:
        helper.issue(operand, helper.copy_plan(operand), e.i32(offset), lo.regs[0], iv, offset_in_stage=0)
    e.raw("nvvm.cp.async.commit.group")
    for item in plan:
        item[2] += e.scratch_base  # absolute
    e.scratch_base += total
    return [(operand, load, offset, helper) for operand, load, offset in plan]


def prefetch_streamed(e, plan, iv_reg: str, lo, step, iv: TValue) -> None:
    """At the top of an iteration: start the next iteration's copies, wait for this one's."""
    count = e.emit(f"llvm.sdiv {e.emit(f'llvm.sub {iv_reg}, {lo.regs[0]} : i32')}, {step.regs[0]} : i32")
    slot = e.emit(f"llvm.urem {count}, {e.i32(2)} : i32")
    other = e.emit(f"llvm.sub {e.i32(1)}, {slot} : i32")
    nxt = e.emit(f"llvm.add {iv_reg}, {step.regs[0]} : i32")
    for operand, load, offset, helper in plan:
        size = -(-operand.bytes // 16) * 16
        base = e.emit(f"llvm.add {e.i32(offset - e.scratch_base)}, {e.emit(f'llvm.mul {other}, {e.i32(size)} : i32')} : i32")
        helper.issue(operand, helper.copy_plan(operand), base, nxt, iv, offset_in_stage=0)
        e.streamed[load] = e.emit(f"llvm.add {e.i32(offset - e.scratch_base)}, {e.emit(f'llvm.mul {slot}, {e.i32(size)} : i32')} : i32")
    e.raw("nvvm.cp.async.commit.group")
    e.raw("nvvm.cp.async.wait.group 1")
    e.barrier()


_INDEX_OPS = {"add": "llvm.add", "sub": "llvm.sub", "mul": "llvm.mul", "floordiv": "llvm.udiv", "mod": "llvm.urem"}


def _evaluable(e, value: TValue, iv: TValue | None) -> bool:
    """A start _start can compute for any stage: the loop variable, a value from outside the
    loop, a scalar constant, or index arithmetic on those (a loop over several reduced labels
    splits its variable into each label's position by division; tile/from_graph.py)."""
    if (iv is not None and value is iv) or value in e.values:
        return True
    op = e.producers.get(value)
    if op is None or op.outputs[0].type.shape != ():
        return False
    if op.kind == "full":
        return True
    return op.kind in _INDEX_OPS and all(_evaluable(e, v, iv) for v in op.inputs)


def _start(e, value: TValue, iv: TValue | None, k: str) -> str | None:
    """The register holding a copy's start at loop position k: k for the loop variable, an
    emitted value, a scalar constant (which a loop body may create before it is emitted), or
    index arithmetic on those, emitted again for this k. Indices are never negative, so the
    divisions are unsigned."""
    if iv is not None and value is iv:
        return k
    if value in e.values:
        return e.values[value].regs[0]
    op = e.producers.get(value)
    if op is not None and op.kind == "full" and op.outputs[0].type.shape == ():
        return e.i32(op.attr("value"))
    if op is not None and op.kind in _INDEX_OPS and op.outputs[0].type.shape == ():
        a, b = (_start(e, v, iv, k) for v in op.inputs)
        if a is None or b is None:
            return None
        return e.emit(f"{_INDEX_OPS[op.kind]} {a}, {b} : i32")
    return None
