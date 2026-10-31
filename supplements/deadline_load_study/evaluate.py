"""Batched evaluator for the explicitly exploratory deadline/load study.

Uses upstream feature, quota, certified payment and FIFO replay implementations.
Only open-loop constant report multipliers are supported. No policy is retrained.
Missing payments are NaN: delivery welfare remains defined, utilities do not.
"""
from __future__ import annotations

import dataclasses
import time
import numpy as np
import torch

from slicemech.control import features, nets
from slicemech.model import allocation
from slicemech.model.scores import FixedMap, ProtectedMap, softmax
from slicemech.model.shapes import stack_shape, value
from slicemech.pricing import envelope
from slicemech.sim import episode, fluid
from slicemech.tenants.strategies import Multiplier


FLUID_FIELDS = ('arrivals_Mbit', 'on_time_Mbit', 'dropped_Mbit', 'late_Mbit',
                'backlog_start_Mbit', 'backlog_Mbit', 'unused_Mbit')


def _forecaster(controller):
    forecast = getattr(controller, 'forecaster', None)
    if forecast is None:
        if (getattr(controller, 'uses_context_features', True)
                or getattr(controller, 'alloc_input', 'reports') == 'forecast'):
            raise ValueError('A fitted controller.forecaster is required')
        forecast = features.OperatorAR.default()
    return forecast


def _multipliers(multipliers, n):
    m = np.ones(n) if multipliers is None else np.asarray(multipliers, dtype=float)
    if m.ndim == 0:
        m = np.full(n, float(m))
    if m.shape != (n,) or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError(f'multipliers must be finite nonnegative scalar or shape ({n},)')
    return m


def evaluate_window(window, controller, price_rule, shape, mech, d=2,
                    multipliers=None, price_tenants=None, *, price_start_interval=0,
                    calculate_fixed=False, deviation_start_interval=0):
    """Return ``arrays, summary, checks, timing_s, metadata`` for one whole window.

    Arrays have (interval, tenant) shape except ``scored`` (interval). Payments
    are calculated from price_start_interval onward, only for selected tenants.
    The default zero includes warm-up. Skipping warm-up bills is valid for these
    fixed multiplier strategies because billing does not alter queues/reports.
    calculate_fixed also computes the frozen-priority bill on the SAME allocation
    path, using the original certified routine; its timing is included in prices. ``price_tenants=None`` means all; ``[]`` skips payment calculation.
    Unpriced prices/utility entries remain NaN rather than assuming zero bills.
    Reports are truthful before deviation_start_interval and use the supplied
    multipliers from that interval onward. Default zero preserves whole-window
    deviations. All summaries exclude window.warmup; replay starts at zero.
    """
    started = time.perf_counter()
    ct = window.contract
    n = ct.n
    raw = np.asarray(window.theta, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != n or not np.all(np.isfinite(raw)) or np.any(raw < 0):
        raise ValueError('window.theta must be finite nonnegative (K,n)')
    K = len(raw)
    warmup = int(window.warmup)
    if not 0 <= warmup < K:
        raise ValueError('At least one scored interval is required')
    if not isinstance(d, (int, np.integer)) or isinstance(d, bool) or d < 1:
        raise ValueError('d must be a positive integer')
    if not mech.integer:
        raise ValueError('This study evaluates the original whole-PRB mechanism')
    if (not isinstance(price_start_interval, (int, np.integer)) or isinstance(price_start_interval, bool)
            or not 0 <= price_start_interval <= K):
        raise ValueError('price_start_interval must be an integer in [0,K]')
    price_start_interval = int(price_start_interval)
    if (not isinstance(deviation_start_interval, (int, np.integer)) or isinstance(deviation_start_interval, bool)
            or not 0 <= deviation_start_interval <= K):
        raise ValueError('deviation_start_interval must be an integer in [0,K]')
    deviation_start_interval = int(deviation_start_interval)
    if calculate_fixed and not isinstance(controller, nets.ProtectedNet):
        raise ValueError('The secondary fixed-price comparison is supported for ProtectedNet only')
    pr = episode.as_price_rule(price_rule)
    selected = list(range(n)) if price_tenants is None else list(price_tenants)
    if len(set(selected)) != len(selected) or any(not isinstance(i, (int, np.integer)) or not 0 <= i < n for i in selected):
        raise ValueError('price_tenants must contain distinct valid integer tenant indices')
    if pr.nu is not None and pr.nu.shape != (n,):
        raise ValueError('Usage tariffs must have one entry per tenant')
    mult = _multipliers(multipliers, n)
    reports = np.minimum(raw * mult[None, :], ct.C)
    admitted = np.minimum(raw, ct.C)
    reports[:deviation_start_interval] = admitted[:deviation_start_interval]
    forecast = _forecaster(controller)
    t0 = time.perf_counter()
    wf = features.window_features(window, d, ct, forecaster=forecast, mech=mech)
    feature_seconds = time.perf_counter() - t0
    reports_enter = getattr(controller, 'alloc_input', 'reports') == 'reports'
    if getattr(controller, 'alloc_input', 'reports') not in ('reports', 'forecast'):
        raise ValueError('Unknown controller allocation input')
    z_alloc = reports if reports_enter else np.asarray(wf.f, dtype=float)
    rate_source = getattr(controller, 'rate_source', 'hat')
    if rate_source not in ('hat', 'true'):
        raise ValueError('Unknown controller rate source')
    r = np.asarray(window.rate_hat if rate_source == 'hat' else window.rate_true, dtype=float)
    r_true = np.asarray(window.rate_true, dtype=float)
    direct = hasattr(controller, 'quota_policy')
    if direct and (not reports_enter or pr.charge == 'fixed'):
        raise ValueError('Direct controllers use reports and have no fixed-priority price')
    t0 = time.perf_counter()
    maps = [None] * K
    weights = np.full((K, n), np.nan)
    if isinstance(controller, nets.PriorityNet):
        parameter = next(controller.parameters())
        ctensor = torch.as_tensor(wf.c_tenant, dtype=parameter.dtype, device=parameter.device)
        gtensor = torch.as_tensor(wf.c_global, dtype=parameter.dtype, device=parameter.device)
        with torch.no_grad():
            if isinstance(controller, nets.ProtectedNet):
                a, kappa = controller.heads(ctensor, gtensor)
                a, kappa = a.cpu().numpy(), kappa.cpu().numpy()
                # Same NumPy score calculation as the deployed ProtectedMap.
                weights = softmax(a + kappa * np.log1p(z_alloc / ct.mu))
                if selected and reports_enter and pr.charge == 'full':
                    maps = [ProtectedMap(a[k], kappa[k], ct.mu) for k in range(K)]
            else:
                zn = torch.as_tensor(z_alloc / ct.mu, dtype=parameter.dtype, device=parameter.device)
                scores = controller(ctensor, gtensor, zn).cpu().numpy()
                weights = softmax(scores)
        # Freezing the already-evaluated vector is EXACT for fixed-priority pricing.
        # A full non-monotone envelope must never be replaced by this fixed map.
        if selected and reports_enter and pr.charge == 'full' and not isinstance(controller, nets.ProtectedNet):
            if isinstance(controller, nets.UnconstrainedNet):
                raise ValueError('Unconstrained full-envelope pricing is outside this study; use fixed')
            maps = [FixedMap(weights[k]) for k in range(K)]
        quotas = allocation.integer(z_alloc, weights, r, ct, mech)
    elif direct:
        quotas = np.empty((K, n), dtype=np.int64)
        for k in range(K):
            maps[k] = controller.quota_policy(wf.context(k), ct=ct, mech=mech, shape=shape, r=r[k])
            quotas[k] = allocation.QuotaRule(ct, mech, r[k], maps[k])(z_alloc[k])
    else:
        for k in range(K):
            maps[k] = controller.priority_map(wf.context(k))
            weights[k] = maps[k].weights(z_alloc[k])
        quotas = allocation.integer(z_alloc, weights, r, ct, mech)
    quota_seconds = time.perf_counter() - t0
    if np.any(quotas < 0) or np.any(quotas.sum(axis=1) > mech.B):
        raise ArithmeticError('Quota feasibility failed')
    beta, rho = stack_shape(shape)
    declared = value(quotas, admitted, r, ct.pi, beta, rho, mech.Delta)
    reported = value(quotas, reports, r, ct.pi, beta, rho, mech.Delta)
    charged = np.full((K, n), np.nan)
    lo = np.full_like(charged, np.nan)
    hi = np.full_like(charged, np.nan)
    fixed_price = np.full_like(charged, np.nan)
    fixed_lo = np.full_like(charged, np.nan)
    fixed_hi = np.full_like(charged, np.nan)
    t0 = time.perf_counter()
    if selected:
        for k in range(price_start_interval, K):
            if not reports_enter or pr.charge == 'none':
                charged[k, selected] = lo[k, selected] = hi[k, selected] = 0.0
                continue
            if pr.charge == 'usage':
                charged[k, selected] = pr.nu[selected] * quotas[k, selected]
                lo[k, selected] = hi[k, selected] = charged[k, selected]
                continue
            policy = FixedMap(weights[k]) if pr.charge == 'fixed' else maps[k]
            rule = allocation.QuotaRule(ct, mech, r[k], policy)
            if selected == list(range(n)):
                fn = envelope.full_prices if pr.charge == 'full' else envelope.fixed_priority_prices
                results = fn(reports[k], rule, ct, shape, mech, r[k])
            else:
                fn = envelope.full_price if pr.charge == 'full' else envelope.fixed_priority_price
                results = [fn(i, reports[k], rule, ct, shape, mech, r[k]) for i in selected]
            for i, result in zip(selected, results):
                if result.quota != quotas[k, i]:
                    raise ArithmeticError(f'Pricing quota differs from allocation at k={k}, tenant={i}')
                charged[k, i], lo[k, i], hi[k, i] = result.p, result.p_lo, result.p_hi
    if calculate_fixed and selected:
        if pr.charge == 'fixed':
            fixed_price[:], fixed_lo[:], fixed_hi[:] = charged, lo, hi
        else:
            for k in range(price_start_interval, K):
                rule = allocation.QuotaRule(ct, mech, r[k], FixedMap(weights[k]))
                if selected == list(range(n)):
                    results = envelope.fixed_priority_prices(reports[k], rule, ct, shape, mech, r[k])
                else:
                    results = [envelope.fixed_priority_price(i, reports[k], rule, ct, shape, mech, r[k])
                               for i in selected]
                for i, result in zip(selected, results):
                    if result.quota != quotas[k, i]:
                        raise ArithmeticError(f'Secondary fixed-price quota differs at k={k}, tenant={i}')
                    fixed_price[k, i], fixed_lo[k, i], fixed_hi[k, i] = result.p, result.p_lo, result.p_hi
    price_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    slots = int(round(mech.Delta * fluid.SLOTS_PER_S))
    if abs(slots - mech.Delta * fluid.SLOTS_PER_S) > 1e-9 or len(window.arrivals_ms) != K * slots:
        raise ValueError('Arrivals do not match interval/slot dimensions')
    cumulative = fluid.cumulative_arrivals(window.arrivals_ms)
    replay, cleared_start, state_end = fluid.replay(cumulative, quotas, r_true, ct.delay_ms,
                                                   slots_per_interval=slots)
    fluid_seconds = time.perf_counter() - t0
    arrays = {name: np.asarray(getattr(replay, name)) for name in FLUID_FIELDS}
    realized = arrays['on_time_Mbit'] * ct.pi[None, :]
    arrays.update(theta=raw, theta_adm=admitted, report=reports, z_alloc=z_alloc, r_alloc=r,
                  quota=quotas, w=weights, V_declared=declared, V_report=reported,
                  realized_value=realized, price=charged, price_lo=lo, price_hi=hi,
                  utility_declared=declared-charged, utility_realized=realized-charged,
                  fixed_price=fixed_price, fixed_price_lo=fixed_lo, fixed_price_hi=fixed_hi,
                  fixed_utility_declared=declared-fixed_price, fixed_utility_realized=realized-fixed_price,
                  scored=np.arange(K) >= warmup, cleared_start_Mbit=cleared_start)
    residual = (arrays['arrivals_Mbit'] - arrays['on_time_Mbit'] - arrays['dropped_Mbit']
                - arrays['late_Mbit'] - arrays['backlog_Mbit'] + arrays['backlog_start_Mbit'])
    if np.any(np.abs(residual) > 1e-9 * (1.0 + cumulative[-1])):
        raise ArithmeticError('Fluid conservation failed')
    handover = float(np.max(np.abs(arrays['backlog_start_Mbit'][1:] - arrays['backlog_Mbit'][:-1]))) if K > 1 else 0.0
    if handover != 0:
        raise ArithmeticError('Queue handover failed')
    finite = np.isfinite(charged)
    bracket_excess = float(np.max(np.r_[0.0, (lo-charged)[finite], (charged-hi)[finite]]))
    if bracket_excess > 1e-10:
        raise ArithmeticError('Payment lies outside its certified interval')
    paid = np.zeros(n, dtype=bool)
    paid[selected] = price_start_interval <= warmup
    s = slice(warmup, None)
    on = arrays['on_time_Mbit'][s].sum(axis=0)
    expired = arrays['dropped_Mbit'][s].sum(axis=0)
    resolved = on + expired
    expiry_share = np.divide(expired, resolved, out=np.full(n, np.nan), where=resolved > 0)
    summary = {
        'scored_intervals': K-warmup,
        'delivery_welfare': float(realized[s].sum(axis=1).mean()),
        'on_time_Mbit': on.tolist(), 'dropped_Mbit': expired.tolist(),
        'arrivals_Mbit': arrays['arrivals_Mbit'][s].sum(axis=0).tolist(),
        'late_Mbit': arrays['late_Mbit'][s].sum(axis=0).tolist(),
        'initial_backlog_Mbit': arrays['backlog_start_Mbit'][warmup].tolist(),
        'final_backlog_Mbit': arrays['backlog_Mbit'][-1].tolist(),
        'resolved_byte_expiry_share': expiry_share.tolist(),
        'mean_realized_value': realized[s].mean(axis=0).tolist(),
        'mean_contract_value': declared[s].mean(axis=0).tolist(),
        'mean_price': charged[s].mean(axis=0).tolist(),
        'mean_delivery_utility': (realized-charged)[s].mean(axis=0).tolist(),
        'mean_contract_utility': (declared-charged)[s].mean(axis=0).tolist(),
        'priced_tenants': selected,
        'mean_fixed_price': fixed_price[s].mean(axis=0).tolist(),
        'mean_fixed_delivery_utility': (realized-fixed_price)[s].mean(axis=0).tolist(),
        'mean_fixed_contract_utility': (declared-fixed_price)[s].mean(axis=0).tolist(),
    }
    # JSON-safe summaries retain absence explicitly; numerical arrays keep NaN.
    summary = {key: [None if isinstance(v, float) and not np.isfinite(v) else v for v in val]
               if isinstance(val, list) else val for key, val in summary.items()}
    checks = {'conservation_max_Mbit': float(np.max(np.abs(residual))), 'handover_max_Mbit': handover,
              'budget_excess_PRB': float(max(0, np.max(quotas.sum(axis=1)-mech.B))),
              'price_bracket_violation': bracket_excess,
              'max_price_bracket_width': float(np.max((hi-lo)[finite])) if finite.any() else None,
              'pricing_quota_matches': True,
              'max_fixed_price_bracket_width': float(np.nanmax(fixed_hi-fixed_lo))
                  if np.isfinite(fixed_price).any() else None,
              'scored_boundary_conservation_Mbit': (arrays['arrivals_Mbit'][s].sum(axis=0)-on-expired
                  -arrays['late_Mbit'][s].sum(axis=0)-arrays['backlog_Mbit'][-1]
                  +arrays['backlog_start_Mbit'][warmup]).tolist()}
    timing = {'features': feature_seconds, 'quotas': quota_seconds, 'prices': price_seconds,
              'fluid': fluid_seconds, 'total': time.perf_counter()-started}
    metadata = {'window_id': str(getattr(window, 'id', '')), 'split': str(window.split),
                'date': str(window.date), 'K': K, 'warmup': warmup, 'd': int(d),
                'controller': episode._controller_name(controller), 'price_rule': pr.charge,
                'multipliers': mult.tolist(), 'price_tenants': selected,
                'price_start_interval': price_start_interval, 'calculate_fixed': bool(calculate_fixed),
                'deviation_start_interval': deviation_start_interval,
                'utility_defined': paid.tolist(), 'delay_ms': np.asarray(ct.delay_ms).tolist(),
                'summary_units': 'welfare/value/utility/price: credits per scored interval; traffic: scored-window Mbit'}
    return {'arrays': arrays, 'summary': summary, 'checks': checks, 'timing_s': timing, 'metadata': metadata}


def _prefix_window(window, intervals):
    if intervals is None or intervals >= len(window.theta):
        return window
    K = int(intervals)
    if K < 1:
        raise ValueError('Validation interval count must be positive')
    slots = len(window.arrivals_ms) // len(window.theta)
    changes = {}
    for name in ('theta', 'measured', 'rate_hat', 'rate_true'):
        changes[name] = getattr(window, name)[:K].copy()
    changes['arrivals_ms'] = window.arrivals_ms[:K*slots].copy()
    changes['warmup'] = min(int(window.warmup), K-1)
    if hasattr(window, 'meta'):
        changes['meta'] = dict(window.meta)
        if 'rate_age_s' in changes['meta']:
            changes['meta']['rate_age_s'] = np.asarray(changes['meta']['rate_age_s'])[:K].copy()
    return dataclasses.replace(window, **changes)


def validate_against_original(window, controllers, shape, mech, d=2, max_intervals=12,
                              deviation_tenant=0, deviation_multiplier=1.5,
                              tolerance=1e-7, max_seconds=None, price_start_interval=0,
                              calculate_fixed=False, deviation_start_interval=0):
    """Compare on TRAIN only; controllers maps labels to (controller, price_rule).

    Truth is checked for every supplied arm, with additional one-tenant deviations
    for ProtectedNet and UnconstrainedNet. Baselines use no payments. A caller may
    set max_seconds to abort between cases if pricing is too expensive; no partial
    validation is reported as success. ``max_intervals=None`` checks the full window.
    """
    if str(window.split) != 'train':
        raise ValueError('Implementation validation must use TRAIN windows only')
    window = _prefix_window(window, max_intervals)
    started = time.perf_counter()
    records = []

    class DelayedMultiplier(Multiplier):
        """Original policy interface; OwnHistory length is the current interval."""
        def __init__(self, factor):
            super().__init__(factor)
            self.truthful = Multiplier(1.0)

        def report(self, i, theta, C, history=None, game=None):
            if history is None:
                raise ValueError('Delayed validation strategy needs OwnHistory')
            if len(history) < deviation_start_interval:
                before = self.truthful.n_clipped
                report = self.truthful.report(i, theta, C, history=history, game=game)
                self.n_clipped += self.truthful.n_clipped-before
                return report
            return super().report(i, theta, C, history=history, game=game)

    for label, (controller, rule) in controllers.items():
        priced = isinstance(controller, (nets.ProtectedNet, nets.UnconstrainedNet))
        cases = [('truth', np.ones(window.contract.n))]
        if priced:
            m = np.ones(window.contract.n)
            m[deviation_tenant] = deviation_multiplier
            cases.append(('deviation', m))
        truth_arrays = None
        for case, mult in cases:
            if max_seconds is not None and time.perf_counter()-started > max_seconds:
                raise TimeoutError('Validation runtime guard exceeded before all comparisons completed')
            selected = None if priced and case == 'truth' else ([deviation_tenant] if priced else [])
            secondary = bool(calculate_fixed and isinstance(controller, nets.ProtectedNet))
            fast = evaluate_window(window, controller, rule, shape, mech, d, mult, selected,
                                   price_start_interval=price_start_interval, calculate_fixed=secondary,
                                   deviation_start_interval=deviation_start_interval)
            original_rule = rule if priced else 'none'
            original = episode.run_window(window, controller, original_rule,
                         [DelayedMultiplier(m) for m in mult], d, mech, shape, record=True,
                         price_columns='all' if secondary else 'billed')
            checks = {}
            names = ['report', 'z_alloc', 'quota', 'V_declared', 'V_report', 'realized_value', *FLUID_FIELDS]
            for name in names:
                want = original[name].to_numpy().reshape(len(window.theta), window.contract.n)
                got = fast['arrays'][name]
                err = float(np.max(np.abs(got-want)))
                checks[name] = err
                threshold = 0 if name == 'quota' else tolerance
                if err > threshold:
                    raise AssertionError(f'{label}/{case}: {name} difference {err} > {threshold}')
            if priced:
                inds = list(range(window.contract.n)) if selected is None else selected
                for name in ('price', 'utility_declared', 'utility_realized'):
                    want = original[name].to_numpy().reshape(len(window.theta), window.contract.n)[price_start_interval:, inds]
                    got = fast['arrays'][name][price_start_interval:, inds]
                    err = float(np.max(np.abs(got-want)))
                    checks[name] = err
                    if err > tolerance:
                        raise AssertionError(f'{label}/{case}: {name} difference {err}')
            if secondary:
                want = original['p_fix'].to_numpy().reshape(len(window.theta), window.contract.n)[price_start_interval:, inds]
                got = fast['arrays']['fixed_price'][price_start_interval:, inds]
                err = float(np.max(np.abs(got-want)))
                checks['secondary_fixed_price'] = err
                if err > tolerance:
                    raise AssertionError(f'{label}/{case}: secondary fixed price difference {err}')
            if case == 'truth':
                truth_arrays = fast['arrays']
            elif deviation_start_interval > 0:
                for name in ('report', 'quota', 'on_time_Mbit', 'dropped_Mbit', 'backlog_Mbit'):
                    err = float(np.max(np.abs(fast['arrays'][name][:deviation_start_interval]
                                              - truth_arrays[name][:deviation_start_interval])))
                    checks['truthful_warmup_'+name] = err
                    if err != 0:
                        raise AssertionError(f'{label}: shared truthful warm-up differs in {name}')
                if deviation_start_interval < len(window.theta):
                    err = float(np.max(np.abs(fast['arrays']['backlog_start_Mbit'][deviation_start_interval]
                                              - truth_arrays['backlog_start_Mbit'][deviation_start_interval])))
                    checks['shared_backlog_at_deviation_start'] = err
                    if err != 0:
                        raise AssertionError(f'{label}: deviation starts from a different queue')
            records.append({'controller': label, 'case': case, 'K': len(window.theta),
                            'max_abs_differences': checks, 'fast_timing_s': fast['timing_s'],
                            'original_seconds': original.attrs['seconds']})
    return {'passed': True, 'split': 'train', 'cases': records,
            'seconds': time.perf_counter()-started, 'absolute_tolerance': tolerance}
