"""Training-channel attacker (INTERFACES.md §10; experiment_design.md §6 and E5).

On each test date the windows are split in order into retraining windows (the first ``n_train``, default 4) and
evaluation windows (the next ``n_eval``, default 4). The target tenant (a video or URLLC tenant) reports
``z = min(alpha * theta, C)`` in every interval of the retraining windows and truthfully afterwards; every other
tenant is truthful throughout.

What the attack can reach depends on the retraining pipeline (``control/retrain.py``):

* stored-report retraining reads the logged reports, i.e. ``stored_reports(windows, plan)`` -> distorted target column;
* measurement-rebuilt retraining reads only released measurements, which this module never touches, so its parameters
  must be bit-identical for every alpha (H5a).

Disjoint exogenous pools (round-1 review, finding 1; revised in round 2): inside a split every window draws its
URLLC/mMTC users and CQI runs from the same few captures, so without care the evaluation windows would replay the
very traffic-seconds the retraining windows were fine-tuned on. A plan therefore names the pool phase of each
window group (``train_part`` = 'A', ``eval_part`` = 'B' by default; ``data.compose.compose_window(...,
pool_part=...)``). The two phases use disjoint halves of every TRACTOR capture (interleaved 2-s blocks) and of the
Raca runs, and which half serves which phase alternates with the date (``data.compose.physical_part``). Compose the
retraining windows with ``plan.train_part`` and the evaluation windows with ``plan.eval_part``.

E5 protocol (round-2 review, finding on the DEV pilot): DEV dates carry only 6 standard windows (every 4 h), too
few for 4 + 4. The E5 windows of a date - on DEV for the pilot that fixes the pre-registered H5b sign, and on TEST
- are therefore the 8 windows at the TEST offsets ``60 + 10800 j`` s after 11:00 UTC (:data:`E5_OFFSETS`,
:func:`e5_layout`): the first ``E5_N_TRAIN = 4`` are retraining windows (phase A), the next ``E5_N_EVAL = 4``
evaluation windows (phase B). DEV pilot and TEST thus share the retraining budget, the time-of-day gap and the
pool phases.

H5b reference (round-2 review): the later gain of the distorting tenant is measured against the stored-report run
with ``alpha = REFERENCE_ALPHA = 1`` on the same date (same retraining windows and checkpoint, truthful logs), not
against the base checkpoint, so that anything common to every alpha (retraining itself, differences between the
retraining and evaluation windows) cancels. Report each window's class loads (mean measured demand per tenant over
the scored intervals, Mbit/s, and ``eval.headroom.window_load``) next to every E5 result.

Units: theta, z and C in Mbit/s. Clipping at C is the contract's admission rule and is counted, never silent.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .strategies import Multiplier, Truthful, admit

ALPHAS = (0.5, 0.7, 1.5, 2.0)
#: Within-date reference arm of H5b: stored-report retraining on truthful logs.
REFERENCE_ALPHA = 1.0
TARGET_CLASSES = ("video", "urllc")
POOL_PARTS = ("A", "B")          # retraining windows use phase A, evaluation windows phase B
#: Pre-registered E5 protocol: retraining / evaluation windows per date and their offsets (s after 11:00 UTC), equal
#: to the TEST window offsets (``data.compose.window_offsets('test')``) and used on DEV for the pilot as well.
E5_N_TRAIN = 4
E5_N_EVAL = 4
E5_OFFSETS = tuple(60 + 10800 * j for j in range(E5_N_TRAIN + E5_N_EVAL))
E5_SPLITS = ("dev", "test")


def e5_alphas() -> tuple[float, ...]:
    """Distortion levels of an E5 date: the reference alpha = 1 followed by :data:`ALPHAS`."""
    return (REFERENCE_ALPHA,) + tuple(ALPHAS)


def e5_layout(split: str) -> tuple[tuple[int, str], ...]:
    """The E5 windows of a ``split`` ('dev' pilot or 'test') date in time order: ``(offset_s, pool_part)`` pairs,
    the first :data:`E5_N_TRAIN` with phase 'A' (retraining) and the next :data:`E5_N_EVAL` with phase 'B'."""
    if split not in E5_SPLITS:
        raise ValueError(f"E5 runs on {E5_SPLITS} dates (DEV = the pilot), not on {split!r}")
    return tuple((o, "A" if j < E5_N_TRAIN else "B") for j, o in enumerate(E5_OFFSETS))


def target_index(cls: Sequence[str], target: str | int) -> int:
    """Tenant index of the attacker: an int is checked and returned; a class name selects its first tenant."""
    n = len(cls)
    if isinstance(target, int | np.integer):
        if not 0 <= int(target) < n:
            raise ValueError(f"target index {target} out of range for n = {n}")
        return int(target)
    hits = [j for j, c in enumerate(cls) if c == target]
    if not hits:
        raise ValueError(f"no tenant of class {target!r} in {tuple(cls)}")
    return hits[0]


def distort(theta, C, target: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Reports of one window with the target's column multiplied by alpha.

    Args:
        theta: truthful reports, shape (K, n), Mbit/s.
        C: report ceilings, shape (n,), Mbit/s.
        target: attacker's tenant index.
        alpha: distortion factor, finite and > 0.

    Returns:
        (z, n_clipped): z (K, n) Mbit/s with every column admitted to [0, C], and the number of reports the ceiling
        clipped per tenant, shape (n,) (non-target tenants are clipped only if their type exceeds C).
    """
    theta = np.asarray(theta, dtype=float)
    C = np.asarray(C, dtype=float)
    if theta.ndim != 2 or C.shape != (theta.shape[1],):
        raise ValueError(f"need theta (K, n) and C (n,), got {theta.shape} and {C.shape}")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"alpha must be finite and > 0, got {alpha}")
    if not 0 <= target < theta.shape[1]:
        raise ValueError("target index out of range")
    raw = theta.copy()
    raw[:, target] *= alpha
    z, clipped = admit(raw, C[None, :])
    return z, clipped.sum(0).astype(np.int64)


@dataclass(frozen=True)
class PoisonPlan:
    """Which windows of a date the target distorts.

    Attributes:
        target: attacker's tenant index; alpha: distortion factor;
        train_ids: retraining windows (attacked); eval_ids: evaluation windows (truthful);
        train_part, eval_part: exogenous pool parts the two phases are composed from (None: shared pools, which
            replays the retraining traffic in the evaluation; allowed only for diagnostics).
    """

    target: int
    alpha: float
    train_ids: tuple
    eval_ids: tuple
    train_part: str | None = "A"
    eval_part: str | None = "B"

    def __post_init__(self):
        if set(self.train_ids) & set(self.eval_ids):
            raise ValueError("retraining and evaluation windows must be disjoint")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and > 0")
        for part in (self.train_part, self.eval_part):
            if part is not None and part not in POOL_PARTS:
                raise ValueError(f"pool part must be one of {POOL_PARTS} or None, got {part!r}")
        if self.train_part is not None and self.train_part == self.eval_part:
            raise ValueError("retraining and evaluation windows must use disjoint pool parts")

    def is_attack_window(self, window_id: Hashable) -> bool:
        return window_id in self.train_ids

    def policies(self, window_id: Hashable, n: int) -> list:
        """Tenant strategies for ``window_id``: Multiplier(alpha) for the target in attacked windows, else truthful.

        Fresh objects are returned on every call; read their ``n_clipped`` counters after the window.
        """
        if not 0 <= self.target < n:
            raise ValueError("target index out of range")
        pols: list = [Truthful() for _ in range(n)]
        if self.is_attack_window(window_id):
            pols[self.target] = Multiplier(self.alpha)
        return pols


def plan_date(window_ids: Sequence[Hashable], target: int, alpha: float, n_train: int = E5_N_TRAIN,
              n_eval: int = E5_N_EVAL, train_part: str | None = "A", eval_part: str | None = "B") -> PoisonPlan:
    """Split one date's windows (in time order) into ``n_train`` attacked and the next ``n_eval`` truthful windows.

    ``window_ids`` are the ids of the date's windows in time order; compose the first ``n_train`` with
    ``pool_part=train_part`` and the rest with ``pool_part=eval_part`` (the ids differ by part). For E5 use the
    windows of :func:`e5_layout` (8 per date on DEV and TEST); the 6 standard DEV windows are too few.
    """
    ids = list(window_ids)
    if len(ids) < n_train + n_eval:
        raise ValueError(f"date has {len(ids)} windows, need {n_train + n_eval}")
    if len(set(ids)) != len(ids):
        raise ValueError("window ids must be unique")
    return PoisonPlan(target=int(target), alpha=float(alpha), train_ids=tuple(ids[:n_train]),
                      eval_ids=tuple(ids[n_train:n_train + n_eval]), train_part=train_part, eval_part=eval_part)


def stored_reports(windows: Iterable, plan: PoisonPlan) -> tuple[dict, dict]:
    """The report log a stored-report retraining would read for the attacked windows.

    Args:
        windows: objects with ``id``, ``theta`` (K, n) Mbit/s and ``contract.C`` (n,) (``data.Window``).
        plan: the poison plan; only windows in ``plan.train_ids`` are returned.

    Returns:
        (reports, n_clipped): dicts keyed by window id with z (K, n) Mbit/s and per-tenant clip counts (n,).
    """
    reports, clips = {}, {}
    for w in windows:
        if not plan.is_attack_window(w.id):
            continue
        z, nc = distort(w.theta, w.contract.C, plan.target, plan.alpha)
        reports[w.id], clips[w.id] = z, nc
    missing = set(plan.train_ids) - set(reports)
    if missing:
        raise ValueError(f"windows {sorted(map(str, missing))} of the plan were not supplied")
    return reports, clips
