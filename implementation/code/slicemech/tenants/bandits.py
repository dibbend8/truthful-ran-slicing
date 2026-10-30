"""Learning tenants: multiplier bandits (INTERFACES.md §10; experiment_design.md §6).

Each *episode* is one window. At the start of an episode every tenant's bandit picks an arm = a report multiplier m
from ``MULTIPLIERS`` and reports ``z = min(m * theta, C)`` for the whole window (``strategies.Multiplier``). All tenants
learn simultaneously. The reward of tenant i is its declared utility over the window relative to the all-truthful run
on the same window (common random numbers):

    gain_i = (U_i(m) - U_i(1)) / |U_i(1)|        (dimensionless; U in credits summed over the scored intervals)

* ``EpsGreedy`` uses ``gain_i`` directly (sample-mean arm values, exploration eps_e = max(0.05, 0.985**e)).
* ``EXP3`` needs rewards in [0, 1]; it receives ``squash(gain_i) = (1 + tanh(gain_i / reward_scale)) / 2``, a strictly
  increasing map (so arm rankings are preserved; nothing is clipped). Choose ``reward_scale`` near the size of the
  gains at stake (default 0.05 = 5 % of truthful utility): gains beyond ~2 scales saturate near 1 and EXP3 can no
  longer tell those arms apart. Even unsaturated, 400 episodes over 9 arms (regret ~ sqrt(T K ln K) ~ 89 episodes)
  cannot separate arms whose gains differ by under ~1 %, so report EXP3's end-state *distribution*, not its argmax.

Window evaluations are memoised by (window id, multiplier-arm profile) in ``MemoizedEvaluator``; share one memo across
the runs (seeds) of a design so that repeated profiles are simulated once.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MULTIPLIERS = np.array([0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0])
TRUTH_ARM = int(np.flatnonzero(MULTIPLIERS == 1.0)[0])


def eps_schedule(episode: int, floor: float = 0.05, decay: float = 0.985) -> float:
    """Exploration probability of episode e (0-based): max(floor, decay**e)."""
    if episode < 0:
        raise ValueError("episode index must be >= 0")
    return max(floor, decay ** episode)


def squash(gain, scale: float = 0.05) -> np.ndarray:
    """Strictly increasing map of a relative gain to (0, 1): (1 + tanh(gain / scale)) / 2; squash(0) = 0.5."""
    if scale <= 0:
        raise ValueError("reward scale must be > 0")
    return 0.5 * (1.0 + np.tanh(np.asarray(gain, dtype=float) / scale))


def _argmax_random_ties(values: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the maximum of ``values``; exact ties are broken uniformly at random."""
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if best.size == 1 else int(rng.choice(best))


# --------------------------------------------------------------------------- bandit algorithms
class EpsGreedy:
    """Epsilon-greedy over K arms with sample-mean values (initial value 0 = 'no gain over truth').

    Args:
        n_arms: number of arms K.
        rng: numpy Generator (all randomness of this learner).
        floor, decay: exploration schedule eps_e = max(floor, decay**e).
    """

    def __init__(self, n_arms: int, rng: np.random.Generator, floor: float = 0.05, decay: float = 0.985) -> None:
        if n_arms < 1:
            raise ValueError("need at least one arm")
        self.K = int(n_arms)
        self.rng = rng
        self.floor, self.decay = float(floor), float(decay)
        self.value = np.zeros(self.K)
        self.count = np.zeros(self.K, dtype=np.int64)

    def epsilon(self, episode: int) -> float:
        return eps_schedule(episode, self.floor, self.decay)

    def probabilities(self, episode: int) -> np.ndarray:
        """Selection probabilities at ``episode`` (greedy ties share the exploitation mass)."""
        eps = self.epsilon(episode)
        best = self.value == self.value.max()
        return eps / self.K + (1 - eps) * best / best.sum()

    def select(self, episode: int) -> int:
        """Explore uniformly with probability eps_e, else play the greedy arm."""
        if self.rng.random() < self.epsilon(episode):
            return int(self.rng.integers(self.K))
        return _argmax_random_ties(self.value, self.rng)

    def update(self, arm: int, reward: float) -> None:
        """Incremental sample mean of the (unbounded) reward of ``arm``."""
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        self.count[arm] += 1
        self.value[arm] += (reward - self.value[arm]) / self.count[arm]

    def greedy(self) -> int:
        """Current greedy arm (lowest index among exact ties, so the read-out is deterministic)."""
        return int(np.argmax(self.value))


class EXP3:
    """EXP3 (Auer et al., 2002) over K arms with rewards in [0, 1].

    p = (1 - gamma) * softmax(log_w) + gamma / K; after playing arm a with reward x: log_w[a] += gamma * (x / p[a]) / K.

    Args:
        n_arms: number of arms K.
        rng: numpy Generator.
        gamma: exploration rate; default min(1, sqrt(K ln K / ((e - 1) * horizon))) (Auer et al., Cor. 3.2).
        horizon: number of episodes used for the default gamma.
    """

    def __init__(self, n_arms: int, rng: np.random.Generator, gamma: float | None = None, horizon: int = 400) -> None:
        if n_arms < 2:
            raise ValueError("EXP3 needs at least two arms")
        self.K = int(n_arms)
        self.rng = rng
        if gamma is None:
            gamma = min(1.0, math.sqrt(self.K * math.log(self.K) / ((math.e - 1.0) * horizon)))
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = float(gamma)
        self.log_w = np.zeros(self.K)
        self.count = np.zeros(self.K, dtype=np.int64)
        self._p_last: np.ndarray | None = None

    def probabilities(self, episode: int | None = None) -> np.ndarray:
        """Current mixed strategy (sums to 1)."""
        w = np.exp(self.log_w - self.log_w.max())
        return (1 - self.gamma) * w / w.sum() + self.gamma / self.K

    def select(self, episode: int | None = None) -> int:
        p = self.probabilities()
        self._p_last = p
        return int(self.rng.choice(self.K, p=p))

    def update(self, arm: int, reward: float) -> None:
        """Importance-weighted exponential update; reward must lie in [0, 1]."""
        if not (0.0 <= reward <= 1.0):
            raise ValueError(f"EXP3 rewards must lie in [0, 1], got {reward}")
        p = self._p_last if self._p_last is not None else self.probabilities()
        self.count[arm] += 1
        self.log_w[arm] += self.gamma * (reward / p[arm]) / self.K
        self._p_last = None

    def greedy(self) -> int:
        """Arm with the largest current probability."""
        return int(np.argmax(self.log_w))


ALGORITHMS = {"exp3": EXP3, "eps": EpsGreedy}


# --------------------------------------------------------------------------- memoised evaluation
class MemoizedEvaluator:
    """Cache of window evaluations keyed by (window id, arm-index profile).

    Args:
        evaluate: ``evaluate(window_id, m) -> utilities`` with m (n,) multipliers. It returns either the per-tenant
            declared utility summed over the window's scored intervals, shape (n,) credits, or a mapping with key
            ``'utility'`` (n,) plus any extra per-episode diagnostics (scalars or arrays, e.g. realized welfare).
        arms: the multiplier values the arm indices refer to.
    """

    def __init__(self, evaluate: Callable[[Hashable, np.ndarray], Any], arms: Sequence[float] = MULTIPLIERS) -> None:
        self.evaluate = evaluate
        self.arms = np.asarray(arms, dtype=float)
        self.cache: dict[tuple, dict] = {}
        self.calls = 0

    def __call__(self, window_id: Hashable, arm_idx: Sequence[int]) -> dict:
        key = (window_id, tuple(int(a) for a in arm_idx))
        if key not in self.cache:
            out = self.evaluate(window_id, self.arms[list(key[1])])
            self.calls += 1
            res = dict(out) if isinstance(out, Mapping) else {"utility": out}
            u = np.asarray(res["utility"], dtype=float)
            if u.shape != (len(key[1]),) or not np.all(np.isfinite(u)):
                raise ValueError(f"evaluate must return finite utilities of shape ({len(key[1])},), got {u.shape}")
            res["utility"] = u
            self.cache[key] = res
        return self.cache[key]


@dataclass
class BanditRun:
    """History of one learning-tenant run (E episodes, n tenants, K arms).

    Attributes:
        algo: 'exp3' | 'eps'; arms (K,): multipliers; window (E,): window id per episode;
        arm (E, n) and m (E, n): chosen arm index / multiplier; gain (E, n): relative gain over truth;
        reward (E, n): reward fed to the learner; greedy_m (E, n): greedy multiplier after the update;
        utility (E, n): declared window utility (credits); final_prob (n, K): final selection probabilities;
        extras: per-episode diagnostics returned by ``evaluate`` (stacked over episodes).
    """

    algo: str
    arms: np.ndarray
    window: list
    arm: np.ndarray
    m: np.ndarray
    gain: np.ndarray
    reward: np.ndarray
    greedy_m: np.ndarray
    utility: np.ndarray
    final_prob: np.ndarray
    extras: dict = field(default_factory=dict)

    def final_greedy(self) -> np.ndarray:
        """Greedy multiplier per tenant at the end of the run (n,)."""
        return self.greedy_m[-1]

    def final_play_share(self, last: int = 50) -> np.ndarray:
        """Share of the last ``last`` episodes in which each tenant played each arm, shape (n, K)."""
        a = self.arm[-last:]
        return np.stack([(a == k).mean(0) for k in range(len(self.arms))], axis=1)


def run_learning_tenants(evaluate: Callable | MemoizedEvaluator, window_ids: Sequence[Hashable], n: int,
                         algo: str = "exp3", episodes: int = 400, seed: int = 0,
                         arms: Sequence[float] = MULTIPLIERS, reward_scale: float = 0.05,
                         exp3_gamma: float | None = None) -> BanditRun:
    """Run n simultaneous multiplier bandits for ``episodes`` windows drawn uniformly from ``window_ids``.

    Args:
        evaluate: ``evaluate(window_id, m (n,)) -> utilities (n,)`` (see ``MemoizedEvaluator``) or a shared memo.
        window_ids: pool of episode windows (one is drawn uniformly at random per episode).
        n: number of tenants.
        algo: 'exp3' or 'eps' (eps-greedy with eps = max(0.05, 0.985**episode)).
        episodes: number of episodes (default 400, experiment_design.md §6).
        seed: run seed; tenants' learners and the window draw use independent child streams.
        arms: multiplier values; must contain 1.0 (the truthful reference).
        reward_scale: scale of ``squash`` for EXP3 rewards (relative gain 0.05 -> reward 0.88).
        exp3_gamma: EXP3 exploration rate (default from the horizon).

    Returns:
        BanditRun.

    Raises:
        ValueError: if a tenant's truthful window utility is 0 (relative gain undefined) or inputs are invalid.
    """
    if algo not in ALGORITHMS:
        raise ValueError(f"algo must be one of {tuple(ALGORITHMS)}")
    arms = np.asarray(arms, dtype=float)
    hit = np.flatnonzero(arms == 1.0)
    if hit.size != 1:
        raise ValueError("arms must contain the truthful multiplier 1.0 exactly once")
    truth_arm = int(hit[0])
    if len(window_ids) == 0 or n < 1 or episodes < 1:
        raise ValueError("need at least one window, one tenant and one episode")
    memo = evaluate if isinstance(evaluate, MemoizedEvaluator) else MemoizedEvaluator(evaluate, arms)
    if not np.array_equal(memo.arms, arms):
        raise ValueError("shared memo was built for different arms")

    ss = np.random.SeedSequence(seed)
    child = ss.spawn(n + 1)
    draw_rng = np.random.default_rng(child[0])
    K = len(arms)
    if algo == "exp3":
        learners = [EXP3(K, np.random.default_rng(child[1 + i]), gamma=exp3_gamma, horizon=episodes)
                    for i in range(n)]
    else:
        learners = [EpsGreedy(K, np.random.default_rng(child[1 + i])) for i in range(n)]

    E = episodes
    rec = dict(arm=np.zeros((E, n), dtype=np.int64), gain=np.zeros((E, n)), reward=np.zeros((E, n)),
               greedy=np.zeros((E, n), dtype=np.int64), utility=np.zeros((E, n)))
    windows, extras = [], {}
    truth_profile = (truth_arm,) * n
    for e in range(E):
        wid = window_ids[int(draw_rng.integers(len(window_ids)))]
        a = np.array([lr.select(e) for lr in learners], dtype=np.int64)
        u_truth = memo(wid, truth_profile)["utility"]
        if np.any(u_truth == 0):
            raise ValueError(f"window {wid!r}: truthful utility is 0 for tenant(s) "
                             f"{np.flatnonzero(u_truth == 0).tolist()}; relative gain undefined")
        res = memo(wid, a)
        gain = (res["utility"] - u_truth) / np.abs(u_truth)
        reward = squash(gain, reward_scale) if algo == "exp3" else gain
        for i, lr in enumerate(learners):
            lr.update(int(a[i]), float(reward[i]))
        rec["arm"][e], rec["gain"][e], rec["reward"][e] = a, gain, reward
        rec["greedy"][e] = [lr.greedy() for lr in learners]
        rec["utility"][e] = res["utility"]
        windows.append(wid)
        for key, val in res.items():
            if key != "utility":
                extras.setdefault(key, []).append(np.asarray(val))
    final_prob = np.stack([lr.probabilities(E) for lr in learners])
    return BanditRun(algo=algo, arms=arms, window=windows, arm=rec["arm"], m=arms[rec["arm"]], gain=rec["gain"],
                     reward=rec["reward"], greedy_m=arms[rec["greedy"]], utility=rec["utility"],
                     final_prob=final_prob, extras={k: np.stack(v) for k, v in extras.items()})


def run_design(evaluate: Callable, window_ids: Sequence[Hashable], n: int, algo: str = "exp3", runs: int = 4,
               episodes: int = 400, seed0: int = 0, **kw) -> tuple[list[BanditRun], MemoizedEvaluator]:
    """``runs`` independent runs (seeds seed0, seed0 + 1, ...) sharing one memo.

    Returns (list of BanditRun, memo)."""
    if not isinstance(evaluate, MemoizedEvaluator):
        evaluate = MemoizedEvaluator(evaluate, kw.get("arms", MULTIPLIERS))
    memo = evaluate
    out = [run_learning_tenants(memo, window_ids, n, algo=algo, episodes=episodes, seed=seed0 + r, **kw)
           for r in range(runs)]
    return out, memo
