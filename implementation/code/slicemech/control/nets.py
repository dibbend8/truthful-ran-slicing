"""Priority networks (INTERFACES.md §9; experiment_design.md §4; theory_design.md §1.6, Corollary 1).

All nets share an equivariant DeepSets encoder over tenants:
``e_i = MLP(c_i)`` (2 x 64, tanh) on the report-free per-tenant context, pooled ``p = [mean_i e_i, max_i e_i]``.
A head on ``[e_i, p, c_global]`` (one hidden layer of 64 tanh units) produces the score ``y_i``:

* :class:`ProtectedNet`   -> ``a_i = 4 tanh(.)``, ``kappa_i = 2 sigmoid(.)``;
  ``y_i = a_i + kappa_i * log1p(z_i / mu_i)``. ``kappa >= 0`` and ``y_j`` does not depend on ``z_i``, so the
  ratio condition of Theorem 1 holds for every parameter value (Corollary 1(iii)) -> ``ProtectedMap``.
  :class:`Kappa0Net` is the kappa = 0 ablation (reports enter only the contract bounds).
* :class:`UnconstrainedNet` -> the head also receives per-tenant report features
  ``[psi(z_i), z_i/mu_i, z_i/mu_i - f_i/mu_i]`` and their pooled mean/max over tenants; ``y_i`` is linear
  output plus a trainable report-prior skip term ``kappa0 * psi(z_i)`` (scalar ``kappa0``, initialised to
  ``report_prior = 1``; no monotonicity guarantee) -> ``CallableMap(monotone=False)``. The skip term gives the
  unconstrained arm the same report prior at initialisation as the protected arm (``kappa = 2 sigmoid(~0) ~ 1``),
  so H4a compares equal starting policies (round-1 review, finding 10); ``report_prior=None`` removes it.
* :class:`MeasOnlyNet`    -> ``a_i = 4 tanh(.)`` only; reports are unused and the allocator receives the
  operator forecast ``f`` instead of reports (``alloc_input == 'forecast'``) -> ``FixedMap(softmax(a))``.

Nets are float64 torch modules; parameters are initialised from a private RNG stream seeded by ``seed``
(the global torch RNG is untouched). ``forward`` is batched: ``c_t (B, n, F)``, ``c_g (B, G)``,
``zn (B, n) = z / mu`` (reports normalised by the training mean) -> ``y (B, n)``.

Every net carries ``forecaster`` (a ``features.OperatorAR`` or None): the operator forecast its contexts were
built with. ``ppo.train`` sets it, ``save_net``/``load_net`` persist it, and ``ppo``/``sim.episode`` use it when no
forecaster is passed, so a net is never evaluated on contexts from a different forecaster by accident (round-1
review, finding 5).
"""
from __future__ import annotations

import copy
import hashlib
import numpy as np
import torch
from torch import nn

from . import features as _feat

DTYPE = torch.float64
A_SCALE = 4.0        # a_i = 4 tanh(.)
KAPPA_SCALE = 2.0    # kappa_i = 2 sigmoid(.)
HIDDEN = 64
N_REPORT_FEATURES = 3   # psi(z_i), z_i/mu_i, z_i/mu_i - f_i/mu_i


def softmax_np(y: np.ndarray) -> np.ndarray:
    """Softmax over the last axis (float64): ``slicemech.model.scores.softmax``, which raises on non-finite
    scores and on weight underflow (score spread above ~745), so training, static, oracle and deployed maps
    all obey one positivity rule."""
    return _scores_module().softmax(np.asarray(y, float))


def _scores_module():
    from ..model import scores          # lazy: model/ is written by another builder
    return scores


class DeepSetsEncoder(nn.Module):
    """``e_i = tanh(W2 tanh(W1 c_i + b1) + b2)``; pool ``[mean, max]`` over the tenant axis (-2)."""

    def __init__(self, n_in: int, hidden: int = HIDDEN):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, c_t: torch.Tensor):
        e = self.mlp(c_t)                                            # (B, n, H)
        pool = torch.cat([e.mean(dim=-2), e.amax(dim=-2)], dim=-1)  # (B, 2H)
        return e, pool


class PriorityNet(nn.Module):
    """Base class: encoder + head. Subclasses define ``n_out``, ``_extra_in`` and ``_scores``."""

    kind = "base"
    n_out = 1
    alloc_input = "reports"          # what the contract allocator receives: 'reports' or 'forecast'
    uses_reports_in_scores = False
    uses_context_features = True     # sim.episode requires an operator forecaster for such controllers
    forecaster = None                # features.OperatorAR the contexts are built with (set by ppo.train)
    _extra_in = 0

    def __init__(self, n_tenant_features: int = _feat.N_TENANT_FEATURES,
                 n_global_features: int = _feat.N_GLOBAL_FEATURES, hidden: int = HIDDEN, seed: int = 0,
                 **kwargs):
        super().__init__()
        self._spec = dict(kind=self.kind, n_tenant_features=int(n_tenant_features),
                          n_global_features=int(n_global_features), hidden=int(hidden), seed=int(seed), **kwargs)
        self.F, self.G, self.H = int(n_tenant_features), int(n_global_features), int(hidden)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.encoder = DeepSetsEncoder(self.F, self.H)
            self.head = nn.Sequential(nn.Linear(3 * self.H + self.G + self._extra_in, self.H), nn.Tanh(),
                                      nn.Linear(self.H, self.n_out))
        self.to(DTYPE)

    # -- shared plumbing ---------------------------------------------------------------------------
    def spec(self) -> dict:
        """Constructor arguments (for :func:`make_net` / :func:`load_net`)."""
        return dict(self._spec)

    def _check(self, c_t, c_g):
        if c_t.dim() != 3 or c_t.shape[-1] != self.F:
            raise ValueError(f"c_tenant must have shape (B, n, {self.F}), got {tuple(c_t.shape)}")
        if c_g.dim() != 2 or c_g.shape[-1] != self.G or c_g.shape[0] != c_t.shape[0]:
            raise ValueError(f"c_global must have shape (B, {self.G}), got {tuple(c_g.shape)}")

    def _head_input(self, e, pool, c_g):
        n = e.shape[-2]
        return torch.cat([e, pool[:, None, :].expand(-1, n, -1), c_g[:, None, :].expand(-1, n, -1)], dim=-1)

    def head_out(self, c_t, c_g, extra=None):
        """Raw head outputs (B, n, n_out); ``extra`` (B, n, _extra_in) is appended to the head input."""
        self._check(c_t, c_g)
        e, pool = self.encoder(c_t)
        h = self._head_input(e, pool, c_g)
        if extra is not None:
            h = torch.cat([h, extra], dim=-1)
        return self.head(h)

    def forward(self, c_t, c_g, zn=None):
        """Mean score vector ``y (B, n)``; ``zn = z/mu`` (B, n) is required iff the scores use reports."""
        raise NotImplementedError

    # -- deployment -----------------------------------------------------------------------------------
    @staticmethod
    def _ctx_tensors(ctx):
        c_t, c_g = ctx[0], ctx[1]
        c_t = torch.as_tensor(np.asarray(c_t, float), dtype=DTYPE)[None]
        c_g = torch.as_tensor(np.asarray(c_g, float), dtype=DTYPE)[None]
        return c_t, c_g

    def alloc_reports(self, ctx, z):
        """Report vector handed to the contract allocator: the reports ``z``, or the forecast ``ctx.f``."""
        if self.alloc_input == "forecast":
            return np.asarray(ctx.f, float)
        if z is None:
            raise ValueError(f"{self.kind} net allocates on reports; z is required")
        return np.asarray(z, float)

    def priority_map(self, ctx, z=None):
        """Deterministic priority map for context ``ctx`` (a :class:`features.Context`). ``z`` is ignored:
        the map is built from the context alone and evaluated on reports later via ``.weights(z)``.
        The map is a snapshot: later changes to the net's parameters do not affect it."""
        raise NotImplementedError

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ProtectedNet(PriorityNet):
    """Own-report score network ``y_i = a_i(c) + kappa_i(c) log1p(z_i/mu_i)`` with ``kappa_i >= 0``."""

    kind = "protected"
    n_out = 2
    uses_reports_in_scores = True
    kappa_zero = False

    def heads(self, c_t, c_g):
        """``(a (B, n) in (-4, 4), kappa (B, n) in [0, 2])``."""
        o = self.head_out(c_t, c_g)
        a = A_SCALE * torch.tanh(o[..., 0])
        kappa = torch.zeros_like(a) if self.kappa_zero else KAPPA_SCALE * torch.sigmoid(o[..., 1])
        return a, kappa

    def forward(self, c_t, c_g, zn=None):
        a, kappa = self.heads(c_t, c_g)
        if self.kappa_zero:
            return a
        if zn is None:
            raise ValueError("ProtectedNet scores need normalised reports zn = z/mu")
        return a + kappa * torch.log1p(zn)

    def heads_np(self, ctx):
        """``(a, kappa)`` as float64 numpy arrays (n,) for one context."""
        with torch.no_grad():
            a, kappa = self.heads(*self._ctx_tensors(ctx))
        return a[0].numpy().astype(float), kappa[0].numpy().astype(float)

    def priority_map(self, ctx, z=None):
        a, kappa = self.heads_np(ctx)
        return _scores_module().ProtectedMap(a, kappa, np.asarray(ctx.mu, float))


class Kappa0Net(ProtectedNet):
    """kappa = 0 ablation: ``y_i = a_i(c)``; reports enter only the contract bounds (``ProtectedMap`` with
    ``kappa = 0``, which is monotone)."""

    kind = "kappa0"
    n_out = 1
    uses_reports_in_scores = False
    kappa_zero = True


class UnconstrainedNet(PriorityNet):
    """``y_i = head([e_i, pool, c_global, r_i, pooled r]) + kappa0 * psi(z_i)`` with report features ``r_i``
    (no constraint). ``kappa0`` is one trainable scalar initialised to ``report_prior`` (default 1, the protected
    net's initial kappa); ``report_prior=None`` drops the term (no extra parameter)."""

    kind = "unconstrained"
    n_out = 1
    uses_reports_in_scores = True
    _extra_in = 3 * N_REPORT_FEATURES          # own report features + pooled mean + pooled max

    def __init__(self, *args, report_prior: float | None = 1.0, **kwargs):
        if report_prior is not None and not np.isfinite(report_prior):
            raise ValueError("report_prior must be finite or None")
        super().__init__(*args, report_prior=None if report_prior is None else float(report_prior), **kwargs)
        if report_prior is None:
            self.report_skip = None
        else:   # created after the encoder/head, so their seeded initialisation is unchanged
            self.report_skip = nn.Parameter(torch.tensor(float(report_prior), dtype=DTYPE))

    def _skip(self, zn):
        return 0.0 if self.report_skip is None else self.report_skip * torch.log1p(zn)

    @staticmethod
    def report_features(c_t, zn):
        """(B, n, 9): ``[psi(z_i), z_i/mu_i, z_i/mu_i - f_i/mu_i]`` and their mean/max over tenants."""
        if zn is None:
            raise ValueError("UnconstrainedNet scores need normalised reports zn = z/mu")
        fn = c_t[..., _feat.IDX_FORECAST]
        r = torch.stack([torch.log1p(zn), zn, zn - fn], dim=-1)          # (B, n, 3)
        n = r.shape[-2]
        pooled = torch.cat([r.mean(dim=-2), r.amax(dim=-2)], dim=-1)    # (B, 6)
        return torch.cat([r, pooled[:, None, :].expand(-1, n, -1)], dim=-1)

    def forward(self, c_t, c_g, zn=None):
        return self.head_out(c_t, c_g, self.report_features(c_t, zn))[..., 0] + self._skip(zn)

    def priority_map(self, ctx, z=None):
        c_t, c_g = self._ctx_tensors(ctx)
        mu = np.asarray(ctx.mu, float)
        n = c_t.shape[1]
        with torch.no_grad():
            e, pool = self.encoder(c_t)
            base = self._head_input(e, pool, c_g)                      # (1, n, 3H+G), report-free part
        head = copy.deepcopy(self.head).requires_grad_(False)          # snapshot: later training can't leak in
        skip = None if self.report_skip is None else float(self.report_skip.detach())

        def weights_fn(zq):
            zq = np.asarray(zq, float)
            if zq.shape[-1] != n:
                raise ValueError(f"report vector must end in n={n}")
            flat = zq.reshape(-1, n)
            with torch.no_grad():
                zn = torch.as_tensor(flat / mu, dtype=DTYPE)
                extra = UnconstrainedNet.report_features(c_t.expand(len(flat), -1, -1), zn)
                y = head(torch.cat([base.expand(len(flat), -1, -1), extra], dim=-1))[..., 0]
                if skip is not None:
                    y = y + skip * torch.log1p(zn)
            return softmax_np(y.numpy()).reshape(zq.shape)

        return _scores_module().CallableMap(weights_fn, monotone=False)


class MeasOnlyNet(PriorityNet):
    """Intercept-only scores ``y_i = a_i(c)``; the allocator uses the operator forecast instead of reports."""

    kind = "meas_only"
    n_out = 1
    alloc_input = "forecast"

    def forward(self, c_t, c_g, zn=None):
        return A_SCALE * torch.tanh(self.head_out(c_t, c_g)[..., 0])

    def priority_map(self, ctx, z=None):
        with torch.no_grad():
            a = self.forward(*self._ctx_tensors(ctx))[0].numpy().astype(float)
        return _scores_module().FixedMap(softmax_np(a))


_KINDS = {"protected": ProtectedNet, "kappa0": Kappa0Net, "unconstrained": UnconstrainedNet,
          "meas_only": MeasOnlyNet}


def make_net(kind: str, seed: int = 0, **kwargs) -> PriorityNet:
    """Build a priority network: ``kind`` in {'protected', 'kappa0', 'unconstrained', 'meas_only'}."""
    if kind not in _KINDS:
        raise ValueError(f"unknown net kind {kind!r}; expected one of {sorted(_KINDS)}")
    return _KINDS[kind](seed=seed, **kwargs)


def clone_net(net: PriorityNet) -> PriorityNet:
    """Independent deep copy (parameters and spec)."""
    return copy.deepcopy(net)


def param_vector(net: nn.Module) -> np.ndarray:
    """All parameters concatenated in ``state_dict`` order (float64 numpy)."""
    return np.concatenate([v.detach().cpu().numpy().ravel().astype(float) for v in net.state_dict().values()])


def param_digest(net: nn.Module) -> str:
    """SHA-256 of the exact parameter bytes in ``state_dict`` order (bit-identity checks, E5)."""
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_net(net: PriorityNet, path) -> None:
    """Write spec, parameters and the operator forecaster the net was trained with (``net.forecaster``)."""
    fc = getattr(net, "forecaster", None)
    torch.save(dict(spec=net.spec(), state=net.state_dict(), forecaster=None if fc is None else fc.to_dict()), path)


def load_net(path) -> PriorityNet:
    """Inverse of :func:`save_net` (restores ``net.forecaster``; None for files written without one)."""
    blob = torch.load(path, weights_only=False)
    spec = dict(blob["spec"])
    kind = spec.pop("kind")
    if kind == "unconstrained" and "report_prior" not in spec:   # written before the report-prior skip term
        spec["report_prior"] = None
    net = _KINDS[kind](**spec)
    net.load_state_dict(blob["state"])
    fc = blob.get("forecaster")
    net.forecaster = None if fc is None else _feat.OperatorAR.from_dict(fc)
    return net
