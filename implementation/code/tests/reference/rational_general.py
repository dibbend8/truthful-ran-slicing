"""Exact general-n quota pricing without exponential active-set enumeration.

Public binary floats have their exact binary rational meaning, as in the
certified three-tenant reference. The random seed/noise and all public state
must be frozen before reports. Preferences concern integer reserved quotas,
not delivered packets. See README.md for the construction and its limits.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction as F
import math

CASH_QUANTUM = F(1, 1_000_000)


def rational(value):
    if isinstance(value, F):
        return value
    if isinstance(value, int):
        return F(value)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("finite rational public coefficient required")
    return F(value)


def priority_menu(n=3, low=F(3, 20), *, anchor=None):
    low = rational(low)
    if not isinstance(n, int) or n < 3 or not 0 < low < 1:
        raise ValueError("n>=3 and 0<low<1 required")
    high, middle = n - (n - 1) * low, (n - low) / (n - 1)
    high_order = (2, 1, 0) if n == 3 else range(n)
    base = ((F(1),) * n,
            *(tuple(high if i == j else low for i in range(n)) for j in high_order),
            *(tuple(low if i == j else middle for i in range(n)) for j in range(n)))
    if anchor is None:
        return base
    anchor = _positive_weights(anchor, n)
    return tuple(tuple(b * w for b, w in zip(anchor, row)) for row in base)


@dataclass(frozen=True)
class PublicContract:
    budget: F
    gbr: tuple
    mbr: tuple
    ceiling: tuple
    tariff: tuple
    fairness: F
    eta: F

    def __post_init__(self):
        for name in ("budget", "fairness", "eta"):
            object.__setattr__(self, name, rational(getattr(self, name)))
        for name in ("gbr", "mbr", "ceiling", "tariff"):
            object.__setattr__(self, name, tuple(map(rational, getattr(self, name))))
        n = len(self.gbr)
        if (n < 3 or any(len(getattr(self, name)) != n for name in ("mbr", "ceiling", "tariff"))
                or self.budget.denominator != 1 or self.budget < 1
                or not 0 <= self.fairness <= 1 or self.eta < 1
                or any(not 0 <= g <= m or cap <= 0 or pi <= 0
                       for g, m, cap, pi in zip(self.gbr, self.mbr, self.ceiling, self.tariff))):
            raise ValueError("invalid public general-n integer contract")

    @property
    def n(self):
        return len(self.gbr)

    @classmethod
    def from_contract(cls, contract):
        return cls(contract.prbs, contract.gbr, contract.mbr, contract.ceiling,
                   contract.tariff, contract.fairness, contract.eta)


@dataclass(frozen=True)
class PublicDecision:
    intercept: tuple
    influence: tuple
    forecast: tuple
    scale: tuple
    alpha: F
    low: F = F(3, 20)
    anchor: tuple | None = None

    def __post_init__(self):
        for name in ("intercept", "influence", "forecast", "scale"):
            object.__setattr__(self, name, tuple(map(rational, getattr(self, name))))
        for name in ("alpha", "low"):
            object.__setattr__(self, name, rational(getattr(self, name)))
        n = len(self.influence)
        anchor = (F(1),) * n if self.anchor is None else _positive_weights(self.anchor, n)
        object.__setattr__(self, "anchor", anchor)
        if (n < 3 or len(self.intercept) != 2 * n + 1
                or len(self.forecast) != n or len(self.scale) != n
                or not 0 <= self.alpha <= 1 or not 0 < self.low < 1
                or any(x < 0 for x in self.influence + self.forecast)
                or any(x <= 0 for x in self.scale)):
            raise ValueError("invalid public general-n decision")

    @property
    def n(self):
        return len(self.influence)

    @classmethod
    def from_prepared(cls, prepared, *, low=F(3, 20), anchor=None):
        logits, noise = tuple(prepared.public_logits), tuple(prepared.noise)
        if len(logits) != len(noise):
            raise ValueError("logit/noise lengths must agree before addition")
        return cls(tuple(rational(a) + rational(b) for a, b in zip(logits, noise)),
                   prepared.influence, prepared.forecast, prepared.scale, prepared.alpha, low, anchor)

    @property
    def menu(self):
        return priority_menu(self.n, self.low, anchor=self.anchor)

    def scores(self, z):
        z = tuple(map(rational, z))
        if len(z) != self.n:
            raise ValueError("decision/report dimension mismatch")
        terms = tuple(b * lam * ((1 - self.alpha) * f + self.alpha * u) / scale
                      for b, lam, f, u, scale in zip(self.anchor, self.influence, self.forecast, z, self.scale))
        total = sum(terms)
        n, low = self.n, self.low
        high, middle = n - (n - 1) * low, (n - low) / (n - 1)
        order = (2, 1, 0) if n == 3 else range(n)
        linear = (total,
                  *(low * total + (high - low) * terms[j] for j in order),
                  *(middle * total + (low - middle) * terms[j] for j in range(n)))
        return tuple(q + v for q, v in zip(self.intercept, linear))

    def action(self, z):
        scores = self.scores(z)
        return max(range(len(scores)), key=lambda a: (scores[a], -a))

    def intervals(self, z, i):
        """Upper envelope of at most four distinct own-report slopes.

        Within an equal-slope group, the highest intercept/lowest index wins
        for every own report, so discarded lines can never affect the price.
        Endpoints use action() for allocation; isolated ties have zero measure.
        """
        top = z[i]
        if top == 0:
            return []
        zero = list(z)
        zero[i] = F(0)
        intercept = self.scores(zero)
        n, low = self.n, self.low
        high, middle = n - (n - 1) * low, (n - low) / (n - 1)
        order = (2, 1, 0) if n == 3 else range(n)
        weights = (F(1), *(high if j == i else low for j in order),
                   *(low if j == i else middle for j in range(n)))
        groups = {}
        for a, own_weight in enumerate(weights):
            slope = self.alpha * self.influence[i] * self.anchor[i] * own_weight / self.scale[i]
            if slope not in groups or (intercept[a], -a) > (intercept[groups[slope]], -groups[slope]):
                groups[slope] = a
        lines = [(slope, intercept[a], a) for slope, a in groups.items()]
        edges = {F(0), top}
        for k, (s, t, _) in enumerate(lines):
            for ss, tt, _ in lines[:k]:
                cross = (tt - t) / (s - ss)
                if 0 < cross < top:
                    edges.add(cross)
        edges = sorted(edges)
        out = []
        for a, b in zip(edges, edges[1:]):
            mid = (a + b) / 2
            selected = max(lines, key=lambda p: (p[0] * mid + p[1], -p[2]))[2]
            if out and out[-1][2] == selected:
                out[-1] = (out[-1][0], b, selected)
            else:
                out.append((a, b, selected))
        return out


@dataclass(frozen=True, init=False)
class ExactShape:
    """Uniform atoms, normalized to exact mean one; identity includes atoms."""
    atoms: tuple
    prefix: tuple
    n: int

    def __init__(self, atoms):
        atoms = tuple(sorted(map(rational, atoms)))
        if not atoms or any(x < 0 for x in atoms) or sum(atoms) <= 0:
            raise ValueError("nonnegative finite atoms with positive mean required")
        mean = sum(atoms) / len(atoms)
        atoms = tuple(x / mean for x in atoms)
        prefix = [F(0)]
        for x in atoms:
            prefix.append(prefix[-1] + x)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "prefix", tuple(prefix))
        object.__setattr__(self, "n", len(atoms))

    def value(self, quota, theta, rate, tariff):
        if theta == 0:
            return F(0)
        capacity = rate * quota
        j = bisect_right(self.atoms, capacity / theta)
        return tariff * (theta * self.prefix[j] + capacity * (self.n - j)) / self.n

    def integral(self, quota, a, b, rate, tariff):
        if not a < b or quota == 0:
            return F(0)
        capacity = rate * quota
        j = bisect_right(self.atoms, capacity / b)
        k = self.n if a == 0 else bisect_left(self.atoms, capacity / a)
        k = max(j, k)
        return tariff * ((b - a) * self.prefix[j] + (k - j) * capacity
                         - a * (self.prefix[k] - self.prefix[j])) / self.n


def _clip(x, lo, hi):
    return min(max(x, lo), hi)


def _fill(lo, hi, weights, budget):
    """O(n log n) exact event sweep in the water level."""
    lower = sum(lo)
    if lower > budget or any(a > b for a, b in zip(lo, hi)):
        raise ArithmeticError("infeasible bounds")
    if sum(hi) <= budget:
        return tuple(hi)
    if lower == budget:
        return tuple(lo)
    events = {}
    for a, b, w in zip(lo, hi, weights):
        events[a / w] = events.get(a / w, F(0)) + w
        events[b / w] = events.get(b / w, F(0)) - w
    previous, value, slope = F(0), lower, F(0)
    for level, delta in sorted(events.items()):
        next_value = value + slope * (level - previous)
        if slope > 0 and next_value >= budget:
            root = previous + (budget - value) / slope
            result = tuple(_clip(root * w, a, b) for a, b, w in zip(lo, hi, weights))
            if sum(result) != budget:
                raise ArithmeticError("waterfill identity failure")
            return result
        previous, value, slope = level, next_value, slope + delta
    raise ArithmeticError("uncovered water level")


def _inputs(z, rates, contract):
    c = contract if isinstance(contract, PublicContract) else PublicContract.from_contract(contract)
    z, rates = tuple(map(rational, z)), tuple(map(rational, rates))
    if (len(z) != c.n or len(rates) != c.n
            or any(not 0 <= u <= cap for u, cap in zip(z, c.ceiling))
            or any(rate <= 0 for rate in rates)):
        raise ValueError("supported reports and positive rate estimates required")
    return z, rates, c


def _allocation(z, weights, rates, c):
    reservation = tuple(min(u, g) / r for u, g, r in zip(z, c.gbr, rates))
    total = sum(reservation)
    if total > c.budget:
        return tuple(c.budget * v / total for v in reservation)
    headroom = tuple(min(c.eta * u, m) / r for u, m, r in zip(z, c.mbr, rates))
    if sum(headroom) < c.budget:
        cap = tuple(m / r if u > 0 else F(0) for u, m, r in zip(z, c.mbr, rates))
        return _fill(headroom, cap, weights, c.budget)
    lo = tuple(min(v + c.fairness * (c.budget - total) / c.n, h)
               for v, h in zip(reservation, headroom))
    return _fill(lo, headroom, weights, c.budget)


def exact_allocation(z, weights, rates, contract):
    z, rates, c = _inputs(z, rates, contract)
    weights = _positive_weights(weights, c.n)
    return _allocation(z, weights, rates, c)


def _positive_weights(weights, n):
    weights = tuple(map(rational, weights))
    if len(weights) != n or any(w <= 0 for w in weights):
        raise ValueError("positive weights matching contract required")
    return weights


def allocate(z, decision, rates, contract):
    """Allocation-only entry point, independent of the pricing work limit."""
    z, rates, c = _inputs(z, rates, contract)
    d = decision if isinstance(decision, PublicDecision) else PublicDecision.from_prepared(decision)
    if d.n != c.n:
        raise ValueError("contract/decision dimension mismatch")
    selected = d.action(z)
    weights = _action_weights(d, selected)
    fractional = _allocation(z, weights, rates, c)
    integer = tuple(map(int, fractional))
    return dict(action=selected, fractional=fractional, integer=integer,
                resource_unused=int(c.budget) - sum(integer))


def _action_weights(d, selected):
    """Materialize only one menu row, rather than the full O(n^2) menu."""
    n, low = d.n, d.low
    if selected == 0:
        row = (F(1),) * n
    elif selected <= n:
        own = (2, 1, 0)[selected - 1] if n == 3 else selected - 1
        high = n - (n - 1) * low
        row = tuple(high if j == own else low for j in range(n))
    else:
        own, middle = selected - n - 1, (n - low) / (n - 1)
        row = tuple(low if j == own else middle for j in range(n))
    return tuple(b * w for b, w in zip(d.anchor, row))


def _at(pair, u):
    return pair[0] * u + pair[1]


@dataclass(frozen=True)
class _Stage:
    left: F
    right: F
    kind: str
    lo: tuple
    hi: tuple
    other: F = F(0)


def report_stages(z, rates, c, i):
    """O(n) intervals of affine lower/upper bounds, independent of action.

    The positive-side limit at zero is intentional: activation may jump there,
    whereas actual zero-report allocation is evaluated separately by allocate.
    """
    top = z[i]
    if top == 0:
        return []
    n = c.n
    opp_b = sum(min(z[j], c.gbr[j]) / rates[j] for j in range(n) if j != i)
    opp_h = sum(min(c.eta * z[j], c.mbr[j]) / rates[j] for j in range(n) if j != i)
    edges = {F(0), top}
    for v in (c.gbr[i], c.mbr[i] / c.eta, (c.budget - opp_b) * rates[i],
              (c.budget - opp_h) * rates[i] / c.eta):
        if 0 < v < top:
            edges.add(v)
    edges = sorted(edges)
    out = []
    for a, b in zip(edges, edges[1:]):
        mid = (a + b) / 2
        gamma = [(F(0), min(z[j], c.gbr[j]) / rates[j]) for j in range(n)]
        gamma[i] = (1 / rates[i], F(0)) if mid < c.gbr[i] else (F(0), c.gbr[i] / rates[i])
        if sum(_at(p, mid) for p in gamma) > c.budget:
            out.append(_Stage(a, b, "outage", tuple(gamma), (), opp_b))
            continue
        headroom = [(F(0), min(c.eta * z[j], c.mbr[j]) / rates[j]) for j in range(n)]
        headroom[i] = ((c.eta / rates[i], F(0)) if c.eta * mid < c.mbr[i]
                       else (F(0), c.mbr[i] / rates[i]))
        if sum(_at(p, mid) for p in headroom) < c.budget:
            cap = tuple((F(0), c.mbr[j] / rates[j] if j == i or z[j] > 0 else F(0)) for j in range(n))
            out.append(_Stage(a, b, "fill", tuple(headroom), cap))
            continue
        slack = (-sum(p[0] for p in gamma), c.budget - sum(p[1] for p in gamma))
        floors = [(g[0] + c.fairness * slack[0] / n, g[1] + c.fairness * slack[1] / n)
                  for g in gamma]
        sub = {a, b}
        for f, h in zip(floors, headroom):
            if f[0] != h[0]:
                cross = (h[1] - f[1]) / (f[0] - h[0])
                if a < cross < b:
                    sub.add(cross)
        sub = sorted(sub)
        for aa, bb in zip(sub, sub[1:]):
            mm = (aa + bb) / 2
            lo = tuple(f if _at(f, mm) <= _at(h, mm) else h for f, h in zip(floors, headroom))
            out.append(_Stage(aa, bb, "fill", lo, tuple(headroom)))
    if sum(s.right - s.left for s in out) != top:
        raise ArithmeticError("incomplete report-stage partition")
    return out


def _stage_quota(stage, weights, u, i, budget):
    if stage.kind == "outage":
        own = _at(stage.lo[i], u)
        return budget * own / (own + stage.other)
    return _fill(tuple(_at(p, u) for p in stage.lo),
                 tuple(_at(p, u) for p in stage.hi), weights, budget)[i]


def _first_ge(pair, target, a, b):
    slope, intercept = pair
    if slope < 0:
        raise ArithmeticError("own bound must be nondecreasing")
    if _at(pair, a) >= target:
        return a
    if slope == 0 or _at(pair, b) < target:
        return None
    return (target - intercept) / slope


def _opponent_threshold(stage, weights, q, a, b, i, budget):
    """First u with q + sum_{j!=i} clip(q*w_j/w_i,L_j(u),U_j)<=B.

    Opponent upper bounds are constant and their lower bounds nonincreasing.
    Each contributes at most one derivative-change event, hence O(n log n).
    """
    value, slope, events = F(q), F(0), {}
    for j, (lo, hi, w) in enumerate(zip(stage.lo, stage.hi, weights)):
        if j == i:
            continue
        if lo[0] > 0 or hi[0] != 0:
            raise ArithmeticError("unsupported parametric opponent bounds")
        target = F(q) * w / weights[i]
        if target >= hi[1]:
            value += hi[1]
        elif _at(lo, a) > target:
            value += _at(lo, a)
            slope += lo[0]
            if lo[0] < 0:
                cross = (target - lo[1]) / lo[0]
                if a < cross < b:
                    events[cross] = events.get(cross, F(0)) - lo[0]
        else:
            value += target
    previous = a
    if value <= budget:
        return a
    for edge in sorted({*events, b}):
        next_value = value + slope * (edge - previous)
        if next_value <= budget and slope < 0:
            return previous + (budget - value) / slope
        previous, value = edge, next_value
        slope += events.get(edge, F(0))
    return None


def quota_threshold(stage, weights, q, a, b, i, budget):
    """Exact first crossing of fractional quota q on a continuous stage."""
    if stage.kind == "outage":
        slope, intercept = stage.lo[i]
        if q == budget or slope == 0:
            raise ArithmeticError("unreachable outage threshold")
        root = (q * stage.other - (budget - q) * intercept) / ((budget - q) * slope)
    else:
        own_hi = _first_ge(stage.hi[i], q, a, b)
        own_lo = _first_ge(stage.lo[i], q, a, b)
        other = _opponent_threshold(stage, weights, q, a, b, i, budget)
        candidates = []
        if own_lo is not None:
            candidates.append(own_lo)
        if own_hi is not None and other is not None:
            candidates.append(max(own_hi, other))
        if not candidates:
            raise ArithmeticError("uncovered integer threshold")
        root = min(candidates)
    if not a <= root <= b:
        raise ArithmeticError("threshold outside its report stage")
    return root


class PricingLimitExceeded(RuntimeError):
    """No price/certificate is returned when the configured work cap is hit."""


def quota_panels(z, d, rates, c, i, *, work=None, max_thresholds=100_000):
    """Complete own-report integer quota panels on [0,z_i], ignoring points.

    Only integers actually crossed on a stage are inverted. Upward selector
    jumps need no inversion; they are already exact interval boundaries.
    """
    intervals = [(a, b, _action_weights(d, selected)) for a, b, selected in d.intervals(z, i)]
    return _quota_panels_intervals(z, intervals, rates, c, i,
                                  work=work, max_thresholds=max_thresholds)


def _quota_panels_intervals(z, intervals, rates, c, i, *, work=None, max_thresholds=100_000):
    """Common exact backend for menu selection and one fixed weight vector."""
    if work is None:
        work = dict(thresholds=0, waterfills=0, selector_intervals=0, stage_intervals=0)
    stages = report_stages(z, rates, c, i)
    work["selector_intervals"] += len(intervals)
    work["stage_intervals"] += len(stages)
    out = []
    for a, b, weights in intervals:
        for stage in stages:
            aa, bb = max(a, stage.left), min(b, stage.right)
            if aa >= bb:
                continue
            left = _stage_quota(stage, weights, aa, i, c.budget)
            right = _stage_quota(stage, weights, bb, i, c.budget)
            work["waterfills"] += 2 * (stage.kind == "fill")
            if right < left or left < 0 or right > c.budget:
                raise ArithmeticError("nonmonotone/infeasible stage endpoint")
            crossings = int(right) - int(left)
            if max_thresholds is not None and work["thresholds"] + crossings > max_thresholds:
                raise PricingLimitExceeded("exact threshold work cap exceeded; no quote certified")
            work["thresholds"] += crossings
            previous, quota = aa, int(left)
            for q in range(int(left) + 1, int(right) + 1):
                root = quota_threshold(stage, weights, q, aa, bb, i, c.budget)
                if root < previous:
                    raise ArithmeticError("unordered exact quota thresholds")
                if root > previous:
                    out.append((previous, root, quota))
                previous, quota = root, q
            if previous < bb:
                out.append((previous, bb, quota))
    # Adjacent equal quotas can span stage changes, but never gaps or decreases.
    merged = []
    for a, b, quota in out:
        if merged and (merged[-1][1] != a or quota < merged[-1][2]):
            raise ArithmeticError("complete-map panel coverage/monotonicity failure")
        if merged and merged[-1][2] == quota:
            merged[-1] = (merged[-1][0], b, quota)
        else:
            merged.append((a, b, quota))
    if sum(b - a for a, b, _ in merged) != z[i]:
        raise ArithmeticError("incomplete integer quota partition")
    return merged


def quote(z, decision, rates, contract, shapes, *, cash_quantum=CASH_QUANTUM,
          max_thresholds=100_000):
    """Whole fixed-G menu envelope, exact prices, then public cash rounding.

    max_thresholds is a deterministic work guard, not a timeout or precision
    tolerance. A rejected call returns no payment or incentive certificate.
    """
    z, rates, c = _inputs(z, rates, contract)
    d = decision if isinstance(decision, PublicDecision) else PublicDecision.from_prepared(decision)
    if d.n != c.n:
        raise ValueError("contract/decision dimension mismatch")
    shapes, quantum = _pricing_inputs(shapes, c, cash_quantum, max_thresholds)
    result = allocate(z, d, rates, c)
    return _settle(z, rates, c, shapes, quantum, result,
                   lambda i, work: quota_panels(z, d, rates, c, i,
                                               work=work, max_thresholds=max_thresholds))


def quote_static(z, weights, rates, contract, shapes, *, cash_quantum=CASH_QUANTUM,
                 max_thresholds=100_000):
    """Exact envelope for arbitrary fixed public positive weights, n>=3.

    Weights need not be a menu row or sum to n. They must remain the same
    under all own-report counterfactuals. The returned action is None and
    weights contains the admitted exact vector; all settlement fields retain
    quote() meanings. No menu or learned decision is required.
    """
    z, rates, c = _inputs(z, rates, contract)
    weights = _positive_weights(weights, c.n)
    shapes, quantum = _pricing_inputs(shapes, c, cash_quantum, max_thresholds)
    fractional = _allocation(z, weights, rates, c)
    integer = tuple(map(int, fractional))
    result = dict(action=None, weights=weights, fractional=fractional, integer=integer,
                  resource_unused=int(c.budget) - sum(integer))
    return _settle(z, rates, c, shapes, quantum, result,
                   lambda i, work: _quota_panels_intervals(
                       z, [(F(0), z[i], weights)] if z[i] else [], rates, c, i,
                       work=work, max_thresholds=max_thresholds))


def _pricing_inputs(shapes, c, cash_quantum, max_thresholds):
    shapes = tuple(shapes)
    quantum = rational(cash_quantum)
    if (len(shapes) != c.n or not all(isinstance(s, ExactShape) for s in shapes)
            or quantum <= 0 or (max_thresholds is not None
                               and (not isinstance(max_thresholds, int) or max_thresholds < 0))):
        raise ValueError("matching exact shapes and positive public quantum required")
    return shapes, quantum


def _settle(z, rates, c, shapes, quantum, result, panel_factory):
    work = dict(thresholds=0, waterfills=0, selector_intervals=0, stage_intervals=0)
    prices, counts = [], []
    for i in range(c.n):
        panels = panel_factory(i, work)
        integral = sum((shapes[i].integral(q, a, b, rates[i], c.tariff[i])
                        for a, b, q in panels), F(0))
        price = shapes[i].value(result["integer"][i], z[i], rates[i], c.tariff[i]) - integral
        if price < 0:
            raise ArithmeticError("negative exact complete-map price")
        prices.append(price)
        counts.append(len(panels))
    units = tuple(int(p / quantum + F(1, 2)) for p in prices)
    result.update(exact_price=tuple(prices), cash_units=units,
                  cash_price=tuple(q * quantum for q in units), panels=tuple(counts),
                  max_epoch_deviation_gain=quantum, truthful_IR_loss=quantum / 2, work=work)
    return result
