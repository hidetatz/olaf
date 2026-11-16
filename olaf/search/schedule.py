"""Schedules: the per-op choices that the search tunes and conversion applies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DotSchedule:
    """Tile shape, pipeline depth and warps of a matrix-instruction dot (see tile.from_graph)."""

    bm: int
    bn: int
    bk: int
    stages: int
    warps: int

    def shared_bytes(self, element_bytes: int = 2) -> int:
        return self.stages * (self.bm * self.bk + self.bk * self.bn) * element_bytes


def dot_key(spec: str, shapes, dtype: str, capability_digest: str) -> str:
    """What a dot schedule depends on: the contraction, operand shapes and dtype, and the device."""
    return f"{spec}|{'x'.join(map(str, shapes[0]))}|{'x'.join(map(str, shapes[1]))}|{dtype}|{capability_digest[:16]}"
