"""Priority maps ``z -> w`` with the context fixed at construction (INTERFACES.md §4; theory_design.md Thm 1, Cor. 1).

Every map exposes ``weights(z)`` for reports of shape ``(..., n)`` (Mbit/s), returning positive weights of the same
shape, and the flag ``monotone_by_construction``: True when, for every contract, the induced own quota is
nondecreasing in the own report.  By Theorem 1 that is exactly the ratio condition ``w_j / w_i`` nonincreasing in
``z_i`` for active ``j``; for softmax scores ``y = a + Lam psi(z)`` it is ``Lam[i, i] >= Lam[j, i]`` (Corollary 1).

The report transform is ``psi(z) = log1p(z / mu)`` with ``mu`` the training-mean demand (Mbit/s).
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from slicemech.config import Contract, Mechanism

__all__ = ["PriorityMap", "MAX_SCORE_SPREAD", "psi", "softmax", "ProtectedMap", "GeneralScoreMap", "FixedMap",
           "CallableMap", "ratio_violation", "necessity_witness"]


@runtime_checkable
class PriorityMap(Protocol):
    """Report-dependent priority vector for one interval (context frozen)."""

    monotone_by_construction: bool

    def weights(self, z: np.ndarray) -> np.ndarray:  # z (..., n) -> w (..., n) > 0
        ...


def _vec(name: str, a, n: int | None = None) -> np.ndarray:
    a = np.array(a, dtype=float)
    if a.ndim != 1 or (n is not None and a.shape != (n,)):
        raise ValueError(f"{name} must be a 1-D array" + (f" of length {n}" if n is not None else ""))
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    a.setflags(write=False)
    return a


def _reports(z, n: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim == 0 or z.shape[-1] != n:
        raise ValueError(f"reports must have the tenant axis last with n={n}, got shape {z.shape}")
    if not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("reports must be finite and >= 0 (Mbit/s)")
    return z


def psi(z, mu) -> np.ndarray:
    """Report transform ``psi(z) = log1p(z / mu)`` (dimensionless; ``z`` and ``mu`` in Mbit/s)."""
    return np.log1p(np.asarray(z, dtype=float) / np.asarray(mu, dtype=float))


#: Largest supported score spread ``max(y) - min(y)`` per row (dimensionless). The weight ratio is then at most
#: ``e^690 = 1.1e299 < allocation.MAX_WEIGHT_RATIO``: every weight is a normal float with full precision and the
#: allocator's water fill stays exact (round-2 review: spreads of about 709 to 745 used to reach the allocator and
#: raise a spurious "water-fill total mismatch"; spreads above 708 also give subnormal weights with lost precision).
MAX_SCORE_SPREAD = 690.0


def softmax(y) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Raises ValueError for non-finite scores and ArithmeticError when a row's score spread exceeds
    :data:`MAX_SCORE_SPREAD` (the weights would leave the allocator's supported ratio range).
    """
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("scores must be finite")
    top = y.max(axis=-1, keepdims=True)
    spread = top[..., 0] - y.min(axis=-1)
    if np.any(spread > MAX_SCORE_SPREAD):
        raise ArithmeticError(f"softmax score spread {float(np.max(spread)):.1f} exceeds MAX_SCORE_SPREAD = "
                              f"{MAX_SCORE_SPREAD} (weight ratio beyond the allocator's supported range)")
    e = np.exp(y - top)
    w = e / e.sum(axis=-1, keepdims=True)
    if np.any(w <= 0):
        raise ArithmeticError("softmax weight underflow; weights must stay > 0")
    return w


class ProtectedMap:
    """Protected (own-report-only) scores ``y_i = a_i + kappa_i psi(z_i)``, ``w = softmax(y)``.

    ``kappa >= 0`` is asserted, so every opponent/own weight ratio is nonincreasing in the own report and the map
    is monotone for every contract and every parameter value (Corollary 1(iii)).
    """

    monotone_by_construction = True

    def __init__(self, a, kappa, mu):
        self.a = _vec("a", a)
        n = self.a.size
        self.kappa = _vec("kappa", kappa, n)
        self.mu = _vec("mu", mu, n)
        if np.any(self.kappa < 0):
            raise ValueError("ProtectedMap needs kappa >= 0")
        if np.any(self.mu <= 0):
            raise ValueError("mu must be > 0 (Mbit/s)")

    @property
    def n(self) -> int:
        return self.a.size

    def scores(self, z) -> np.ndarray:
        """Scores ``y`` (dimensionless), shape of ``z``."""
        return self.a + self.kappa * psi(_reports(z, self.n), self.mu)

    def weights(self, z) -> np.ndarray:
        return softmax(self.scores(z))

    def __repr__(self) -> str:
        return f"ProtectedMap(a={self.a.tolist()}, kappa={self.kappa.tolist()})"


class GeneralScoreMap:
    """General linear scores ``y = a + Lam @ psi(z)`` (``y_i = a_i + sum_k Lam[i, k] psi_k(z_k)``), softmax weights.

    Monotone for every contract iff ``Lam[i, i] >= Lam[j, i]`` for all ``j != i`` (Corollary 1(i)).
    """

    def __init__(self, a, Lam, mu):
        self.a = _vec("a", a)
        n = self.a.size
        Lam = np.array(Lam, dtype=float)
        if Lam.shape != (n, n) or not np.all(np.isfinite(Lam)):
            raise ValueError(f"Lam must be a finite ({n}, {n}) matrix")
        Lam.setflags(write=False)
        self.Lam = Lam
        self.mu = _vec("mu", mu, n)
        if np.any(self.mu <= 0):
            raise ValueError("mu must be > 0 (Mbit/s)")

    @property
    def n(self) -> int:
        return self.a.size

    def violations(self) -> list[tuple[int, int, float]]:
        """All ``(i, j, Lam[j, i] - Lam[i, i])`` with ``Lam[j, i] > Lam[i, i]``: tenant i's report lifts j's score more."""
        n, L = self.n, self.Lam
        return [(i, j, float(L[j, i] - L[i, i])) for i in range(n) for j in range(n) if j != i and L[j, i] > L[i, i]]

    def satisfies_ratio_condition(self) -> bool:
        """Corollary 1: ``Lam[i, i] >= Lam[j, i]`` for every ``i`` and ``j != i``."""
        return not self.violations()

    @property
    def monotone_by_construction(self) -> bool:
        return self.satisfies_ratio_condition()

    def scores(self, z) -> np.ndarray:
        """Scores ``y = a + Lam psi(z)`` (dimensionless), shape of ``z``.

        The sum over reports is accumulated elementwise in a fixed order (k = 0..n-1), not by a matrix product: a
        BLAS product rounds differently for different batch sizes, so the same profile could get scores, weights
        and integer quotas that differ in the last bits depending on how many profiles are evaluated together
        (round-2 review). Every row's result is now independent of the batch.
        """
        p = psi(_reports(z, self.n), self.mu)
        acc = p[..., 0:1] * self.Lam[:, 0]
        for k in range(1, self.n):
            acc = acc + p[..., k:k + 1] * self.Lam[:, k]
        return self.a + acc

    def weights(self, z) -> np.ndarray:
        return softmax(self.scores(z))

    def __repr__(self) -> str:
        return f"GeneralScoreMap(n={self.n}, ratio_condition={self.satisfies_ratio_condition()})"


class FixedMap:
    """Report-independent weights (static or equal priorities; also the frozen vector of the fixed-priority price)."""

    monotone_by_construction = True

    def __init__(self, w):
        self.w = _vec("w", w)
        if np.any(self.w <= 0):
            raise ValueError("FixedMap weights must be > 0")

    @property
    def n(self) -> int:
        return self.w.size

    def weights(self, z) -> np.ndarray:
        z = _reports(z, self.n)
        return np.broadcast_to(self.w, z.shape).copy()

    def __repr__(self) -> str:
        return f"FixedMap({self.w.tolist()})"


class CallableMap:
    """Wrap an arbitrary function ``z -> w`` (numpy, or torch with ``torch_input=True``), e.g. an unconstrained net.

    Args:
        fn: function of reports of shape ``(..., n)`` returning positive weights of the same shape (numpy array or
            torch tensor).
        monotone: the caller's declaration that the map satisfies the ratio condition (default False).
        torch_input: convert reports to a float64 torch tensor before calling ``fn``.
        n: optional number of tenants for input validation.
    """

    def __init__(self, fn: Callable, monotone: bool = False, torch_input: bool = False, n: int | None = None):
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.fn, self.monotone_by_construction = fn, bool(monotone)
        self.torch_input, self._n = bool(torch_input), n

    def weights(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self._n is not None:
            z = _reports(z, self._n)
        arg = z
        if self.torch_input:
            import torch  # noqa: PLC0415  (optional dependency, imported on first use)
            arg = torch.as_tensor(z, dtype=torch.float64)
        out = self.fn(arg)
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        w = np.asarray(out, dtype=float)
        if w.shape != z.shape:
            raise ValueError(f"CallableMap fn returned shape {w.shape}, expected {z.shape}")
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("CallableMap fn must return finite positive weights")
        return w

    def __repr__(self) -> str:
        return f"CallableMap(monotone={self.monotone_by_construction})"


# ------------------------------------------------------------------------------------------ necessity witness
def ratio_violation(pmap, i: int, z, u_max: float, n_grid: int = 257, rtol: float = 1e-12):
    """Search an own-report grid on ``(0, u_max]`` for a rise of ``w_j / w_i`` (Theorem 1(a) violated).

    Args:
        pmap: priority map; i: own tenant; z: report profile (Mbit/s, opponents used, own ignored);
        u_max: largest own report searched (Mbit/s); n_grid: grid size; rtol: relative rise tolerated as rounding.

    Returns:
        ``(j, u, v)`` with ``0 < u < v`` and the largest relative ratio rise among active ``j`` (``z_j > 0``), or
        ``None`` if no rise above ``rtol`` exists on the grid.
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    u = np.linspace(0.0, float(u_max), int(n_grid))[1:]
    Z = np.repeat(z[None, :], u.size, axis=0)
    Z[:, i] = u
    W = np.asarray(pmap.weights(Z), dtype=float)
    best = None
    for j in range(n):
        if j == i or z[j] <= 0:
            continue
        lr = np.log(W[:, j]) - np.log(W[:, i])             # log ratio along the own-report path
        run_min = np.minimum.accumulate(lr)                 # lowest ratio at or before each grid point
        rise = lr - run_min
        k = int(np.argmax(rise))
        if rise[k] > rtol and (best is None or rise[k] > best[0]):
            k0 = int(np.argmin(lr[: k + 1]))
            best = (float(rise[k]), j, float(u[k0]), float(u[k]))
    return None if best is None else best[1:]


def necessity_witness(pmap, i: int, j: int, z, u: float, v: float, B: int | None = None,
                      cls: str = "video") -> dict:
    """Witness contract of Theorem 1 for a ratio violation at ``(i, j, z_-i, u, v)``.

    With ``sigma(t) = w_i / (w_i + w_j)`` at ``(t, z_-i)`` and ``sigma(v) < sigma(u)``, the contract
    ``r = 1, g = 0, gamma = 0, M_i = M_j = B, M_k = 0 (bystanders), eta = max(1, B/u, B/z_j)`` puts the cell in the need
    stage with zero floors, so ``X_i(t) = B sigma(t)`` and the own quota falls from ``u`` to ``v``.

    Args:
        pmap: the violating priority map; i, j: own and opponent tenant; z: profile (Mbit/s; ``z_j > 0``);
        u, v: own reports with ``0 < u < v``; B: PRBs (default: large enough for an integer drop of >= 3 PRB);
        cls: class label for the dummy contract.

    Returns:
        dict(ct, mech, r, z_u, z_v, sigma_u, sigma_v, B, lemma_c_overlap) where ``lemma_c_overlap`` says whether
        ``B`` lies in ``(u / sigma(u), v / sigma(v))`` (then Lemma C also rules out every payment).
    """
    z = np.array(z, dtype=float)
    n = z.size
    if not (0 < u < v) or z[j] <= 0 or i == j:
        raise ValueError("need 0 < u < v, j != i and an active opponent z_j > 0")
    z_u, z_v = z.copy(), z.copy()
    z_u[i], z_v[i] = u, v
    W = np.asarray(pmap.weights(np.stack([z_u, z_v])), dtype=float)
    s_u = W[0, i] / (W[0, i] + W[0, j])
    s_v = W[1, i] / (W[1, i] + W[1, j])
    if not s_v < s_u:
        raise ValueError("no ratio violation: sigma(v) >= sigma(u)")
    if B is None:
        B = int(max(106, np.ceil(4.0 / (s_u - s_v))))
    eta = float(max(1.0, B / u, B / z[j]))
    M = np.zeros(n)
    M[i] = M[j] = float(B)
    C = np.maximum(np.maximum(z_u, z_v), 1.0)
    mu = np.asarray(getattr(pmap, "mu", np.ones(n)), dtype=float)
    ct = Contract(cls=(cls,) * n, g=np.zeros(n), M=M, C=C, pi=np.ones(n), delay_ms=np.full(n, 100.0), mu=mu)
    mech = Mechanism(B=int(B), gamma=0.0, eta=eta, Delta=1.0, integer=True)
    return dict(ct=ct, mech=mech, r=np.ones(n), z_u=z_u, z_v=z_v, sigma_u=float(s_u), sigma_v=float(s_v), B=int(B),
                lemma_c_overlap=bool(u / s_u < B < v / s_v))
