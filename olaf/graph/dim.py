"""Symbolic dimension expressions.

sympy is avoided to protect compile time. Only integer +, -, *, // and % are needed, and
shape equality runs constantly, so expressions stay in normal form and compare structurally.

The normal form is a sum of monomial -> integer coefficient. Constant results always become
int, so `==` is enough to compare `Dim`s. Simplification is only polynomial, so equal values
can have different forms (e.g. `(S//2)*2+S%2` and `S` for an unconstrained S).
Those are rejected as shape mismatches, which is safer than wrongly accepting a shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce


@dataclass(frozen=True)
class Sym:
    """Atom of a symbolic dimension. Constraints are part of the type: they simplify `S%128` and are checked at bind time."""

    name: str
    multiple_of: int = 1
    min: int = 0
    max: int | None = None

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class _Div:
    num: DimExpr
    den: int

    def __str__(self) -> str:
        return f"({self.num}//{self.den})"


@dataclass(frozen=True)
class _Mod:
    num: DimExpr
    den: int

    def __str__(self) -> str:
        return f"({self.num}%{self.den})"


Atom = Sym | _Div | _Mod
Monomial = tuple[Atom, ...]


@dataclass(frozen=True)
class DimExpr:
    terms: tuple[tuple[Monomial, int], ...]

    def __add__(self, other: Dim) -> Dim:
        return _from_terms({**_as_terms(self)}, _as_terms(other))

    __radd__ = __add__

    def __neg__(self) -> Dim:
        return self * -1

    def __sub__(self, other: Dim) -> Dim:
        return self + (-other)

    def __rsub__(self, other: Dim) -> Dim:
        return (-self) + other

    def __mul__(self, other: Dim) -> Dim:
        out: dict[Monomial, int] = {}
        for ma, ca in _as_terms(self).items():
            for mb, cb in _as_terms(other).items():
                m = tuple(sorted(ma + mb, key=str))
                out[m] = out.get(m, 0) + ca * cb
        return _cancel_exact_division(out)

    __rmul__ = __mul__

    def __floordiv__(self, other: int) -> Dim:
        den = _positive_const(other)
        if all(c % den == 0 for _, c in self.terms):
            return _normalize({m: c // den for m, c in self.terms})
        return _atom(_Div(self, den))

    def __mod__(self, other: int) -> Dim:
        den = _positive_const(other)
        rest = {m: c for m, c in self.terms if not _term_divisible(m, c, den)}
        if not rest:
            return 0
        kept = _normalize(rest)
        return kept % den if isinstance(kept, int) else _atom(_Mod(kept, den))

    def __str__(self) -> str:
        parts = []
        for mono, coef in self.terms:
            factors = [str(a) for a in mono]
            if not factors:
                parts.append(str(coef))
            elif coef == 1:
                parts.append("*".join(factors))
            else:
                parts.append("*".join([str(coef), *factors]))
        return "+".join(parts)


Dim = int | DimExpr


def sym(name: str, *, multiple_of: int = 1, min: int = 0, max: int | None = None) -> DimExpr:
    return _atom_expr(Sym(name, multiple_of, min, max))


def evaluate(d: Dim, env: Mapping[str, int]) -> int:
    if isinstance(d, int):
        return d
    return sum(coef * reduce(lambda p, a: p * _eval_atom(a, env), mono, 1) for mono, coef in d.terms)


def symbols(d: Dim) -> set[Sym]:
    if isinstance(d, int):
        return set()
    found: set[Sym] = set()
    for mono, _ in d.terms:
        for atom in mono:
            found |= {atom} if isinstance(atom, Sym) else symbols(atom.num)
    return found


def as_symbol(d: Dim) -> Sym | None:
    """Return the symbol if `d` is exactly one symbol."""
    if isinstance(d, DimExpr) and len(d.terms) == 1:
        mono, coef = d.terms[0]
        if coef == 1 and len(mono) == 1 and isinstance(mono[0], Sym):
            return mono[0]
    return None


def bind(env: dict[str, int], s: Sym, value: int) -> None:
    """Assign a concrete value to a symbol and check the constraints in its type."""
    if value < s.min or (s.max is not None and value > s.max) or value % s.multiple_of != 0:
        raise ValueError(f"dimension {s.name}={value} violates its constraints: {s!r}")
    if env.setdefault(s.name, value) != value:
        raise ValueError(f"conflicting values for dimension {s.name}: {env[s.name]} and {value}")


def product(dims) -> Dim:
    return reduce(lambda a, b: a * b, dims, 1)


def _as_terms(d: Dim) -> dict[Monomial, int]:
    if isinstance(d, int):
        return {(): d}
    return dict(d.terms)


def _from_terms(a: dict[Monomial, int], b: dict[Monomial, int]) -> Dim:
    for m, c in b.items():
        a[m] = a.get(m, 0) + c
    return _normalize(a)


def _normalize(terms: dict[Monomial, int]) -> Dim:
    kept = {m: c for m, c in terms.items() if c != 0}
    if not kept:
        return 0
    if set(kept) == {()}:
        return kept[()]
    return DimExpr(tuple(sorted(kept.items(), key=lambda t: "*".join(map(str, t[0])))))


def _cancel_exact_division(terms: dict[Monomial, int]) -> Dim:
    """Rewrite `(S//k)*k` to `S` only when the constraints prove exact division (appears in head-split reshapes)."""
    total: Dim = 0
    for mono, coef in terms.items():
        for atom in mono:
            if isinstance(atom, _Div) and coef % atom.den == 0 and atom.num % atom.den == 0:
                rest = list(mono)
                rest.remove(atom)
                total = total + atom.num * _normalize({tuple(rest): coef // atom.den})
                break
        else:
            total = total + _normalize({mono: coef})
    return total


def _atom_expr(atom: Atom) -> DimExpr:
    return DimExpr((((atom,), 1),))


def _atom(atom: Atom) -> Dim:
    return _atom_expr(atom)


def _positive_const(d: int) -> int:
    if not isinstance(d, int) or d <= 0:
        raise ValueError(f"divisor must be a positive constant: {d}")
    return d


def _term_divisible(mono: Monomial, coef: int, den: int) -> bool:
    multiple = abs(coef)
    if not mono:
        return multiple % den == 0
    for atom in mono:
        if isinstance(atom, Sym):
            multiple *= atom.multiple_of
    return multiple % den == 0


def _eval_atom(atom: Atom, env: Mapping[str, int]) -> int:
    if isinstance(atom, Sym):
        if atom.name not in env:
            raise ValueError(f"dimension {atom.name} has no value")
        return env[atom.name]
    if isinstance(atom, _Div):
        return evaluate(atom.num, env) // atom.den
    return evaluate(atom.num, env) % atom.den
