"""Tenant reporting strategies (INTERFACES.md §10).

A strategy turns what a tenant knows when it reports for interval k into a report z_i (Mbit/s).

Units: the truthful report theta_i and the report z_i are in Mbit/s; the report ceiling C_i is in Mbit/s.

Admission rule (Contract.C): reports are clipped to [0, C_i]. This clipping is part of the contract, not a numerical
safeguard, so it is never silent here: every strategy counts the reports it had to clip in ``n_clipped``. A negative or
non-finite truthful report is an input error and raises ``ValueError``.

Information (theory_design.md Theorem 2, assumption A3; INTERFACES.md §11 step 3): a tenant may condition only on its
own history (``OwnHistory``). The one exception is ``BestResponder``: it is an *analysis device* that computes the
ex-post per-interval best response against the opponents' reports supplied by the episode (truthful opponents in
E2/E3), and it therefore receives a ``StageGame`` that a real tenant would not have.

Protocol used by ``sim/episode.py`` (duck-typed, see ``TenantPolicy``)::

    z_i = policy.report(i, theta_i, C_i, history=h_i, game=g_i)   # g_i only built if policy.needs_game
    policy.reset()                                                 # at the start of every window

Open-loop strategies (``Truthful``, ``Multiplier``) also offer ``report_series(theta, C)`` for a whole window, and
``profile_reports`` applies a profile of open-loop strategies to a (K, n) array at once.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

PRICE_RULES = ("full", "fixed", "usage", "none")


# --------------------------------------------------------------------------- information objects
@dataclass
class OwnHistory:
    """A tenant's own past, in interval order (what A3 allows a tenant to condition on).

    Attributes (lists, one entry per past interval):
        theta: own truthful reports (Mbit/s); z: own submitted reports (Mbit/s);
        quota: own integer quotas (PRB); price: own charges (credits per interval).
    """

    theta: list = field(default_factory=list)
    z: list = field(default_factory=list)
    quota: list = field(default_factory=list)
    price: list = field(default_factory=list)

    def append(self, theta: float, z: float, quota: float, price: float) -> None:
        """Record one finished interval (units as in the class docstring)."""
        self.theta.append(float(theta))
        self.z.append(float(z))
        self.quota.append(float(quota))
        self.price.append(float(price))

    def __len__(self) -> int:
        return len(self.z)


@dataclass(frozen=True)
class StageGame:
    """The one-interval game handed to an analytic best responder.

    Attributes:
        rule: quota rule ``Q(z) -> x`` (PRB) built by the episode from (contract, mechanism, rate, priority map);
            it is evaluated on report profiles of shape (..., n).
        z_minus_i: opponents' reports (Mbit/s) in the layout ``pricing.response.best_response`` expects
            (the episode passes truthful opponents for the ex-post best response).
        kwargs: extra keyword arguments forwarded verbatim to ``best_response`` (e.g. ``ct``, ``shape``, ``mech``,
            ``r``, usage-tariff ``nu``); this keeps the strategy independent of the pricing module's signature tail.
    """

    rule: Callable[[np.ndarray], np.ndarray]
    z_minus_i: np.ndarray
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class TenantPolicy(Protocol):
    """Duck-typed interface every tenant strategy implements."""

    name: str
    needs_game: bool
    n_clipped: int

    def report(self, i: int, theta: float, C: float, history: OwnHistory | None = None,
               game: StageGame | None = None) -> float: ...

    def reset(self) -> None: ...


# --------------------------------------------------------------------------- helpers
def _check_theta_C(theta, C) -> tuple[np.ndarray, np.ndarray]:
    """Validate truthful reports (Mbit/s) and ceilings (Mbit/s); return float arrays."""
    theta = np.asarray(theta, dtype=float)
    C = np.asarray(C, dtype=float)
    if not np.all(np.isfinite(theta)) or np.any(theta < 0):
        raise ValueError("truthful reports theta must be finite and >= 0 (Mbit/s)")
    if not np.all(np.isfinite(C)) or np.any(C <= 0):
        raise ValueError("report ceilings C must be finite and > 0 (Mbit/s)")
    return theta, C


def admit(z, C) -> tuple[np.ndarray, np.ndarray]:
    """Apply the contract's admission rule: clip reports to [0, C] (Mbit/s).

    Returns ``(z_admitted, clipped_mask)``; raises on negative or non-finite reports (those are bugs, not reports).
    """
    z = np.asarray(z, dtype=float)
    C = np.asarray(C, dtype=float)
    if not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("reports must be finite and >= 0 (Mbit/s)")
    clipped = z > C
    return np.where(clipped, C, z), clipped


# --------------------------------------------------------------------------- strategies
class Multiplier:
    """Constant report multiplier: z_i = min(m * theta_i, C_i) (m = 1 is truthful).

    Args:
        m: multiplier, finite and >= 0 (m = 0 means 'always report inactive').
    """

    needs_game = False

    def __init__(self, m: float) -> None:
        m = float(m)
        if not np.isfinite(m) or m < 0:
            raise ValueError(f"multiplier must be finite and >= 0, got {m}")
        self.m = m
        self.name = f"x{m:g}"
        self.n_clipped = 0

    def reset(self) -> None:
        """No per-window state."""

    def report(self, i: int, theta: float, C: float, history: OwnHistory | None = None,
               game: StageGame | None = None) -> float:
        """Return min(m * theta, C) in Mbit/s; clipped reports are counted in ``n_clipped``."""
        th, c = _check_theta_C(theta, C)
        z, clipped = admit(self.m * th, c)
        self.n_clipped += int(np.sum(clipped))
        return float(z)

    def report_series(self, theta, C) -> np.ndarray:
        """Vectorised reports for a whole window: theta (K,) Mbit/s -> z (K,) Mbit/s."""
        th, c = _check_theta_C(theta, C)
        z, clipped = admit(self.m * th, c)
        self.n_clipped += int(np.sum(clipped))
        return z

    def __repr__(self) -> str:
        return f"Multiplier({self.m:g})"


class Truthful(Multiplier):
    """Report the truthful report theta_i (admitted to the ceiling C_i if the type exceeds it): Multiplier(1)."""

    def __init__(self) -> None:
        super().__init__(1.0)
        self.name = "truthful"

    def __repr__(self) -> str:
        return "Truthful()"


def _default_best_response(*args, **kwargs):
    """Resolve ``slicemech.pricing.response.best_response`` lazily (it is owned by the pricing builder)."""
    from slicemech.pricing.response import best_response  # noqa: PLC0415  (import on first use)

    return best_response(*args, **kwargs)


class BestResponder:
    """Ex-post per-interval best response under a given price rule (analysis device; see module docstring).

    Each call evaluates ``best_response(i, theta_i, game.z_minus_i, game.rule, price=self.price, **game.kwargs)``
    (INTERFACES.md §6) and reports its maximiser ``z_star``. The last result is kept in ``last`` and every result's
    ``(gain, exact)`` is appended to ``log`` so the episode can record certified gains.

    Args:
        price: 'full' | 'fixed' | 'usage' | 'none' — the price the tenant optimises against.
        br_fn: optional replacement for ``best_response`` with the same signature (used by tests).
    """

    needs_game = True

    def __init__(self, price: str, br_fn: Callable[..., Any] | None = None) -> None:
        if price not in PRICE_RULES:
            raise ValueError(f"price must be one of {PRICE_RULES}, got {price!r}")
        self.price = price
        self.name = f"best_response[{price}]"
        self._br = br_fn if br_fn is not None else _default_best_response
        self.n_clipped = 0
        self.last = None
        self.log: list[tuple[float, bool]] = []

    def reset(self) -> None:
        """Forget the per-window log (the clip counter is cumulative)."""
        self.last = None
        self.log = []

    def report(self, i: int, theta: float, C: float, history: OwnHistory | None = None,
               game: StageGame | None = None) -> float:
        """Return the best-response report z* in [0, C] (Mbit/s).

        A type above the ceiling is first admitted to C_i (counted in ``n_clipped``), because the value only
        depends on admitted traffic and ``best_response`` requires theta_i in [0, C_i].

        Raises:
            ValueError: if no ``StageGame`` is supplied or the solver returns a report outside [0, C].
        """
        if game is None:
            raise ValueError("BestResponder needs the StageGame (rule and opponents' reports) of the interval")
        th, c = _check_theta_C(theta, C)
        th, clipped = admit(th, c)              # the type is the admitted rate: traffic above C_i is policed away
        self.n_clipped += int(np.sum(clipped))
        br = self._br(i, float(th), game.z_minus_i, game.rule, price=self.price, **dict(game.kwargs))
        z = float(br.z_star)
        if not np.isfinite(z) or z < 0 or z > float(c) * (1 + 1e-12):
            raise ValueError(f"best response returned z*={z} outside [0, C={float(c)}]")
        self.last = br
        self.log.append((float(br.gain), bool(br.exact)))
        return min(z, float(c))

    def __repr__(self) -> str:
        return f"BestResponder({self.price!r})"


# --------------------------------------------------------------------------- profiles
def multiplier_profile(m: Sequence[float]) -> list[Multiplier]:
    """One ``Multiplier`` per tenant from a multiplier vector m (n,)."""
    return [Multiplier(v) for v in np.asarray(m, dtype=float).ravel()]


def profile_reports(policies: Sequence[Any], theta, C) -> np.ndarray:
    """Reports of a profile of open-loop strategies for a whole window.

    Args:
        policies: n strategies, each with ``report_series`` (``Truthful`` / ``Multiplier``).
        theta: truthful reports, shape (K, n), Mbit/s.
        C: report ceilings, shape (n,), Mbit/s.

    Returns:
        z: reports, shape (K, n), Mbit/s.
    """
    theta = np.asarray(theta, dtype=float)
    C = np.asarray(C, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != len(policies) or C.shape != (len(policies),):
        raise ValueError(f"need theta (K, n) and C (n,) with n = {len(policies)} policies; "
                         f"got {theta.shape} and {C.shape}")
    out = np.empty_like(theta)
    for j, pol in enumerate(policies):
        if not hasattr(pol, "report_series"):
            raise TypeError(f"{pol!r} is not open-loop (no report_series); call report() per interval instead")
        out[:, j] = pol.report_series(theta[:, j], C[j])
    return out
