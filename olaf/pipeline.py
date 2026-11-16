"""The compile pipeline from a traced step to a Tile IR program, in its fixed pass order."""

from __future__ import annotations

from olaf.backend.capability import CPU, Capability
from olaf.graph.ir import Graph
from olaf.passes.canonicalize import canonicalize
from olaf.passes.cse import eliminate_common_subexpressions
from olaf.passes.decompose import decompose
from olaf.passes.unroll import unroll_scans
from olaf.tile.from_graph import to_tile
from olaf.tile.fuse import fuse
from olaf.tile.ir import Program
from olaf.tile.normalize import align_rows, hoist_offsets
from olaf.tile.online_softmax import fuse_online_softmax


def lower_graph(graph: Graph) -> Graph:
    """Graph IR passes: composites to core ops, scans unrolled, views folded, common ops merged."""
    return eliminate_common_subexpressions(canonicalize(unroll_scans(decompose(graph))))


def to_program(graph: Graph, capability: Capability = CPU, *, fused: bool = True) -> Program:
    """graph must be the output of lower_graph."""
    program = to_tile(graph, capability)
    if fused:
        program = fuse_online_softmax(fuse(program))
    return hoist_offsets(align_rows(program))
