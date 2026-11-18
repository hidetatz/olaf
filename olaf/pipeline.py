"""The compile pipeline from a traced step to a Tile IR program, in its fixed pass order."""

from __future__ import annotations

from olaf.backend.capability import CPU, Capability
from olaf.graph.ir import Graph
from olaf.passes.canonicalize import canonicalize
from olaf.passes.cse import eliminate_common_subexpressions
from olaf.passes.decompose import decompose
from olaf.passes.unroll import unroll_scans
from olaf.tile.from_graph import to_tile
from olaf.tile.fuse import fuse, fuse_epilogues, fuse_row_epilogues, fuse_sibling_reductions, fuse_siblings
from olaf.tile.ir import Program
from olaf.tile.normalize import align_rows, eliminate_concats, eliminate_copies, eliminate_pad_copies, hoist_offsets
from olaf.tile.online_softmax import fuse_online_softmax
from olaf.tile.rows import fuse_rows


def lower_graph(graph: Graph) -> Graph:
    """Graph IR passes: composites to core ops, scans unrolled, views folded, common ops merged."""
    return eliminate_common_subexpressions(canonicalize(unroll_scans(decompose(graph))))


def to_program(graph: Graph, capability: Capability = CPU, *, fused: bool = True, schedules=None) -> Program:
    """graph must be the output of lower_graph. schedules: search.tune results (None: defaults)."""
    program = to_tile(graph, capability, schedules)
    if fused:
        # Epilogues first as well: a cheap consumer of a dot (a bias add) read by several tasks
        # would otherwise be duplicated into each of them, each materializing its own copy.
        program = fuse_epilogues(program, shared_cheap_only=True)
        program = fuse_sibling_reductions(fuse_siblings(eliminate_concats(eliminate_copies(fuse_epilogues(fuse_row_epilogues(fuse_rows(fuse_online_softmax(fuse(eliminate_pad_copies(program))))))))))
    return hoist_offsets(align_rows(program))
