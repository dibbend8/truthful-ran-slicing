"""Shared configuration dataclasses (binding interface; see code/INTERFACES.md §1).

Units: Mbit/s for rates and demands, PRB for quotas, s for intervals, ms for slots and delay budgets,
credits per interval for values and prices. Arrays over tenants have shape (n,).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import yaml
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]          # .../rewrite_2026-10-18
PARAMS = yaml.safe_load(open(ROOT / "spec" / "params.yaml"))


@dataclass(frozen=True)
class Contract:
    cls: tuple
    g: np.ndarray
    M: np.ndarray
    C: np.ndarray
    pi: np.ndarray
    delay_ms: np.ndarray
    mu: np.ndarray

    @property
    def n(self) -> int:
        return len(self.cls)

    def __post_init__(self):
        n = len(self.cls)
        for name in ("g", "M", "C", "pi", "delay_ms", "mu"):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (n,):
                raise ValueError(f"Contract.{name} must have shape ({n},), got {a.shape}")
            object.__setattr__(self, name, a)
        if np.any(self.g < 0) or np.any(self.M < self.g) or np.any(self.C <= 0) or np.any(self.pi <= 0):
            raise ValueError("need 0 <= g <= M, C > 0, pi > 0")


@dataclass(frozen=True)
class Mechanism:
    B: int = int(PARAMS["network"]["B"])
    gamma: float = float(PARAMS["mechanism"]["gamma_fairness"])
    eta: float = float(PARAMS["mechanism"]["eta_headroom"])
    Delta: float = float(PARAMS["network"]["interval_s"])
    integer: bool = PARAMS["mechanism"]["quotas"] == "integer_floor"


@dataclass(frozen=True)
class Shape:
    beta: tuple
    rho: tuple

    def __post_init__(self):
        for b, r in zip(self.beta, self.rho):
            b = np.asarray(b, float); r = np.asarray(r, float)
            if b.shape != r.shape or np.any(b < 0) or np.any(r < 0):
                raise ValueError("shape atoms must be nonnegative and aligned")
            if abs(r.sum() - 1) > 1e-9 or abs((r * b).sum() - 1) > 1e-9:
                raise ValueError("need sum rho = sum rho*beta = 1")


def contract_from_params(classes: list[str], mu: np.ndarray) -> Contract:
    """Build a contract for the given tenant classes and training-mean demands mu (Mbit/s) from spec/params.yaml."""
    P = PARAMS["classes"]
    mu = np.asarray(mu, float)
    return Contract(
        cls=tuple(classes),
        g=np.array([P[c]["gbr_x_mu"] for c in classes]) * mu,
        M=np.array([P[c]["mbr_x_mu"] for c in classes]) * mu,
        C=np.array([P[c]["ceiling_x_mu"] for c in classes]) * mu,
        pi=np.array([P[c]["tariff_per_mbit"] for c in classes], float),
        delay_ms=np.array([P[c]["delay_budget_ms"] for c in classes], float),
        mu=mu,
    )
