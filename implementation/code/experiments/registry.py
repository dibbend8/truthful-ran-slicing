"""Registry of training configurations, learned arms, hyper-parameters and model files (experiment drivers).

``experiments/train.py`` writes what this module describes; the experiment drivers (e0-e7) read trained models,
static fits and the DEV hyper-parameter selection through it, so every driver resolves the same files.

Orchestrator decisions used here (binding, 2026-10-19; full text in ``code/STATUS.md`` §0 and
``prereg/PREREG_DRAFT.md`` §0):

* **D1** GBR fractions x 0.8 (g/mu = 0.40 video, 0.64 URLLC-like, 0.16 mMTC-like). Enters through the calibrations
  (``spec/params.yaml`` -> ``slicemech.data.calibrate``); nothing here overrides it.
* **D2** Offered-load normalization per date on every split (counts so that each date's mean offered traffic of class c
  is share_c x rho x B x rbar_TRAIN). Every window here is composed by ``Calibration.window``, which applies the
  per-date counts; a load-sweep cell has its own calibration at its rho (nested per-date counts, own contract mu).
* **D6** Learner objective = lookahead return with horizon h = 3 intervals: :func:`ppo_config` sets
  ``PPOConfig.horizon = 3`` explicitly (short-horizon reinforcement learning, not a pure contextual bandit).
* D3, D4, D5 and D7 concern the analysis drivers (subgroups by window offered load, honest-tenant harm, the E5
  protocol, the E4b kappa-scaling frontier); the models and static fits registered here are their inputs.

Configurations (:data:`CONFIGS`): ``default`` (n = 3, d = 2 s, rho = 0.9, video lead 4.004 s; 5 seeds) and the sweep
cells ``d1, d5, d15`` (measurement delay), ``rho0.5, rho0.7, rho1.1`` (offered load), ``lead0, lead1.05`` (video lead
time), ``n6, n9`` (scaling), 3 seeds each. Learned arms (:data:`ARMS`): ``protected``, ``unconstrained``,
``meas_only``, ``kappa0`` (``control.nets`` kinds of the same names).

Hyper-parameters: the DEV tuning grid lr x sigma0 x M (:data:`TUNING_GRID`) is run on the default configuration only,
identically for every learned arm; each arm's cell is selected by DEV realized welfare (mean over the tuning seeds
of the checkpoint-selected DEV welfare) and reused for every other configuration (``selection.json``).

DEV halves (round-3 review: the same 126 DEV windows used to select checkpoints and cells AND to estimate the
primaries for the power analysis, a winner's-curse bias that differs by arm). :func:`dev_role` splits the 21 DEV
dates by their position in ``puffer.split_dates('dev')``: even positions (11 dates, 66 windows) are the SELECTION
half (checkpoint selection, the hyper-parameter cell, the static vector's DEV choice); odd positions (10 dates, 60
windows) are the ESTIMATION half (the DEV estimates that feed the power analysis and the H5b sign; never used for a
choice). ``window_keys('dev', role=...)`` gives either half. The static fit has the same DEV access as a learned arm:
its deployed vector is chosen on the selection half among the TRAIN fit's :data:`STATIC_DEV_CANDIDATES` best
candidates (8 cells x 9 checkpoints = 72 DEV-scored candidates per learned-arm seed).

Layout below ``results/models`` (``root``):

* ``<config>/<arm>/seed<k>/{model.pt, config.json, history.csv}`` -- deployed model (``nets.save_net``: spec, float64
  state dict and the operator forecaster), its job record and per-iteration PPO history;
* ``default/<arm>/tune/<cell>/seed<k>/`` -- tuning runs (the selected cell's runs are copied to
  ``default/<arm>/seed<k>``);
* ``<config>/static/{static.json, candidates.npz}`` -- static fit (the d sweep cells share the default's: the static
  map reads no context, so d does not enter it);
* ``selection.json`` -- the DEV selection per arm, with every cell's scores;
* ``index.csv`` -- one row per finished job (written by ``train.py``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
import pathlib
import sys
from dataclasses import dataclass

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from slicemech import io as sio  # noqa: E402
from slicemech.config import PARAMS  # noqa: E402

# ----------------------------------------------------------------------------------------------- paths
ROOT = sio.ROOT
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
SMOKE_DIR = RESULTS_DIR / "raw" / "smoke_train"
CALIB_DIR = sio.DATA_DIR / "calibration"
#: Large NR outputs go here (not used by training; listed so every driver names one place).
NR_LARGE_DIR = pathlib.Path("/media/darpan/New Volume/ICC_rewrite_nr")

MODEL_FILE = "model.pt"
CONFIG_FILE = "config.json"
HISTORY_FILE = "history.csv"
STATIC_FILE = "static.json"
STATIC_CANDIDATES = "candidates.npz"
SELECTION_FILE = "selection.json"
INDEX_FILE = "index.csv"

SPLITS = ("train", "dev", "test")

DECISIONS = {
    "D1": "GBR fractions x 0.8: g/mu = 0.40 (video), 0.64 (URLLC-like), 0.16 (mMTC-like); TRAIN-only decision, applied "
          "in the calibrations",
    "D2": "per-date offered-load normalization on every split: counts so that each date's mean offered traffic of "
          "class c equals share_c x rho x B x rbar_TRAIN (shares 0.6/0.3/0.1, split equally among a class's tenants); "
          "load sweeps scale rho",
    "D6": "learner objective = lookahead return, horizon h = 3 intervals (PPOConfig.horizon = 3): short-horizon "
          "reinforcement learning, not a pure contextual bandit",
}

# ----------------------------------------------------------------------------------------------- arms
ARMS = ("protected", "unconstrained", "meas_only", "kappa0")
#: ``control.nets.make_net`` kind of each arm.
ARM_NET_KIND = {a: a for a in ARMS}
#: ``spec/controllers.yaml`` keys deployed from each trained arm.
ARM_CONTROLLER_KEYS = {
    "protected": ("protected", "fixed_price"),
    "unconstrained": ("unprotected", "unprotected_usage", "unprotected_envelope"),
    "meas_only": ("meas_only",),
    "kappa0": ("kappa0",),
}


# ----------------------------------------------------------------------------------------------- configurations
_INFO = PARAMS["information"]
DEFAULT_D = int(_INFO["measurement_delay_s"])
DEFAULT_RHO = float(PARAMS["load"]["rho"])
DEFAULT_LEAD = float(_INFO["video_lead_time_s"])
SEEDS_MAIN = tuple(range(int(PARAMS["training"]["seeds_main"])))
SEEDS_SWEEP = tuple(range(int(PARAMS["training"]["seeds_sweeps"])))
DELAY_SWEEP = tuple(int(x) for x in _INFO["measurement_delay_sweep_s"] if int(x) != DEFAULT_D)
#: Offered-load cells (experiment_design.md E1: rho in {0.5, 0.7, 0.9, 1.1}; 0.9 is the default).
LOAD_SWEEP = (0.5, 0.7, 1.1)
LEAD_SWEEP = tuple(float(x) for x in _INFO["video_lead_sweep_s"] if float(x) != DEFAULT_LEAD)
SCALING_N = tuple(int(x) for x in PARAMS["n_tenants"]["scaling"])


@dataclass(frozen=True)
class RunConfig:
    """One training configuration. ``calibration`` names ``data/calibration/<name>.json`` (TRAIN fit)."""

    name: str
    sweep: str                  # 'default' | 'delay' | 'load' | 'lead' | 'scaling'
    n: int = 3
    d: int = DEFAULT_D          # measurement delay (intervals of 1 s)
    rho: float = DEFAULT_RHO
    lead_s: float = DEFAULT_LEAD
    seeds: tuple = SEEDS_SWEEP
    calibration: str = "n3_default"

    @property
    def data_key(self) -> str:
        """Configurations with the same key compose identical windows (d enters only the contexts)."""
        return self.calibration

    @property
    def static_config(self) -> str:
        """Configuration whose static fit this one uses (the d sweep cells share the default's)."""
        return "default" if self.sweep == "delay" else self.name


def _fmt(x: float) -> str:
    return f"{x:g}"


def calibration_name(n: int = 3, rho: float = DEFAULT_RHO, lead_s: float = DEFAULT_LEAD) -> str:
    """Calibration file name of a composition: ``n<n>_default`` or ``n3_rho<rho>`` / ``n3_lead<lead>``."""
    if rho == DEFAULT_RHO and lead_s == DEFAULT_LEAD:
        return f"n{n}_default"
    if n != 3 or (rho != DEFAULT_RHO and lead_s != DEFAULT_LEAD):
        raise ValueError("sweep calibrations change one factor of the n = 3 default")
    return f"n3_rho{_fmt(rho)}" if rho != DEFAULT_RHO else f"n3_lead{_fmt(lead_s)}"


def _build_configs() -> dict:
    cfgs = [RunConfig("default", "default", seeds=SEEDS_MAIN)]
    cfgs += [RunConfig(f"d{d}", "delay", d=d) for d in DELAY_SWEEP]
    cfgs += [RunConfig(f"rho{_fmt(r)}", "load", rho=r, calibration=calibration_name(rho=r)) for r in LOAD_SWEEP]
    cfgs += [RunConfig(f"lead{_fmt(h)}", "lead", lead_s=h, calibration=calibration_name(lead_s=h))
             for h in LEAD_SWEEP]
    cfgs += [RunConfig(f"n{n}", "scaling", n=n, calibration=calibration_name(n=n)) for n in SCALING_N]
    return {c.name: c for c in cfgs}


#: Ordered: default first, then the sweep cells (delay, load, lead, scaling).
CONFIGS: dict[str, RunConfig] = _build_configs()
DEFAULT = "default"

#: Calibrations the configurations need beyond the existing n3/n6/n9 defaults: ``calibrate(**kwargs)`` on TRAIN.
CALIB_SPECS: dict[str, dict] = {
    c.calibration: dict(n=c.n, rho=c.rho, lead_s=c.lead_s)
    for c in CONFIGS.values() if c.sweep in ("load", "lead")
}


# ----------------------------------------------------------------------------------------------- hyper-parameters
TUNING_GRID = {"lr": (1e-4, 3e-4), "sigma0": (0.3, 0.6), "M": (4, 8)}
#: The learning rate decays linearly from lr to lr * LR_END_RATIO (the PPOConfig default 3e-4 -> 3e-5).
LR_END_RATIO = 0.1
#: Lookahead horizon of the training reward (D6).
HORIZON = 3
#: DEV checkpoint-selection period (iterations).
CHECKPOINT_EVERY = 50
ITERATIONS = int(PARAMS["training"]["iterations"])
WINDOWS_PER_ITER = int(PARAMS["experiment_settings"]["PpoWindows"])
CLIP = float(PARAMS["training"]["clip"])
#: Smoke run: 2 PPO iterations on the smoke windows, DEV evaluation after each.
SMOKE_PPO = dict(iterations=2, windows_per_iter=2, eval_every=1)
#: Static fit: number of best TRAIN candidates whose DEV (selection-half) welfare chooses the deployed vector
#: (= 8 tuning cells x 9 DEV checkpoints, the candidates a learned arm's DEV selection scores per seed).
STATIC_DEV_CANDIDATES = 72


@dataclass(frozen=True)
class Cell:
    """One point of the tuning grid."""

    lr: float
    sigma0: float
    M: int

    @property
    def id(self) -> str:
        return f"lr{self.lr:.0e}_s{self.sigma0:g}_M{self.M}"

    def as_dict(self) -> dict:
        return dict(lr=self.lr, sigma0=self.sigma0, M=self.M)


def tuning_cells() -> list[Cell]:
    """The grid in a fixed order (lr, then sigma0, then M); ties in the selection keep the earlier cell."""
    return [Cell(lr, s, M) for lr, s, M in itertools.product(TUNING_GRID["lr"], TUNING_GRID["sigma0"],
                                                            TUNING_GRID["M"])]


def cell_by_id(cell_id: str) -> Cell:
    for c in tuning_cells():
        if c.id == cell_id:
            return c
    raise KeyError(f"unknown tuning cell {cell_id!r}")


def ppo_config(cell: Cell, *, smoke: bool = False):
    """``control.ppo.PPOConfig`` of a tuning cell (D6: horizon 3; DEV checkpoint every 50 iterations)."""
    from slicemech.control.ppo import PPOConfig
    kw = dict(iterations=ITERATIONS, windows_per_iter=WINDOWS_PER_ITER, eval_every=CHECKPOINT_EVERY)
    if smoke:
        kw.update(SMOKE_PPO)
    return PPOConfig(M=int(cell.M), clip=CLIP, lr_start=float(cell.lr), lr_end=float(cell.lr) * LR_END_RATIO,
                     sigma0=float(cell.sigma0), horizon=HORIZON, **kw)


# ----------------------------------------------------------------------------------------------- data
#: Smoke windows: two TRAIN windows (the integration smoke run's) and two DEV windows (first offset of two dates).
SMOKE_TRAIN_WINDOWS = (("2024-01-01", 60), ("2024-01-14", 21660))
SMOKE_DEV_WINDOWS = (("2024-02-28", 60), ("2024-03-02", 60))


def require_split(split: str, *, test_freeze=None) -> str:
    """Validate a ``--split`` value. 'test' passes only after the pre-registration freeze
    (``data.puffer_fetch.require_test_read_gate``: PermissionError while no frozen file is given or recorded)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "test":
        from slicemech.data import puffer_fetch
        puffer_fetch.require_test_read_gate(test_freeze)
    return split


def calibration_path(name: str, calib_root=None) -> pathlib.Path:
    return pathlib.Path(calib_root or CALIB_DIR) / f"{name}.json"


def load_calibration(name: str, calib_root=None):
    """``Calibration`` from ``<calib_root or data/calibration>/<name>.json``."""
    from slicemech.data.calibrate import Calibration
    p = calibration_path(name, calib_root)
    if not p.is_file():
        hint = (" (sweep calibration: run `experiments/train.py --only kind=calib` first)"
                if name in CALIB_SPECS else "")
        raise FileNotFoundError(f"calibration {p} not found{hint}")
    return Calibration.from_dict(sio.read_json(p))


DEV_ROLES = ("select", "estimate")


def dev_role(date: str) -> str:
    """'select' (even position in ``puffer.split_dates('dev')``) or 'estimate' (odd position) of a DEV date: the
    selection half chooses checkpoints, cells and the static vector; the estimation half gives the DEV estimates of
    the primaries (power analysis, H5b sign) and is never used for a choice (module docstring)."""
    from slicemech.data import puffer
    dates = list(puffer.split_dates("dev"))
    if date not in dates:
        raise ValueError(f"{date} is not a DEV date")
    return DEV_ROLES[dates.index(date) % 2]


def dev_dates(role: str) -> list[str]:
    """DEV dates of one half (:func:`dev_role`)."""
    from slicemech.data import puffer
    if role not in DEV_ROLES:
        raise ValueError(f"role must be one of {DEV_ROLES}")
    return [d for d in puffer.split_dates("dev") if dev_role(d) == role]


def window_keys(split: str, *, smoke: bool = False, role: str | None = None) -> list[tuple[str, int]]:
    """(date, offset_s) of the windows a training job uses on ``split`` (every pre-declared window unless smoke);
    ``role`` ('select' | 'estimate', DEV only) keeps one DEV half (:func:`dev_role`; ignored by smoke runs)."""
    from slicemech.data import compose, puffer
    if smoke:
        if split == "train":
            return [tuple(k) for k in SMOKE_TRAIN_WINDOWS]
        if split == "dev":
            return [tuple(k) for k in SMOKE_DEV_WINDOWS]
        raise ValueError("smoke runs have no TEST windows")
    if role is not None and split != "dev":
        raise ValueError("only DEV has selection and estimation halves")
    dates = dev_dates(role) if role is not None else puffer.split_dates(split)
    return [(d, int(o)) for d in dates for o in compose.window_offsets(split)]


def compose_windows(cal, split: str, keys, *, test_freeze=None) -> list:
    """Compose ``keys`` of ``split`` with calibration ``cal`` (D2 per-date counts; TEST only after the freeze)."""
    require_split(split, test_freeze=test_freeze)
    extra = dict(test_freeze=test_freeze) if split == "test" else {}
    return [cal.window(split, d, int(o), **extra) for d, o in keys]


def mechanism(cal):
    """``Mechanism`` recorded in the calibration (B, gamma, eta, Delta, integer)."""
    from slicemech.config import Mechanism
    return Mechanism(**cal.meta["mech"]) if "mech" in cal.meta else Mechanism()


# ----------------------------------------------------------------------------------------------- model files
def model_dir(config: str, arm: str, seed: int, *, root=None) -> pathlib.Path:
    return pathlib.Path(root or MODELS_DIR) / config / arm / f"seed{int(seed)}"


def tune_dir(arm: str, cell_id: str, seed: int, *, root=None) -> pathlib.Path:
    return pathlib.Path(root or MODELS_DIR) / DEFAULT / arm / "tune" / cell_id / f"seed{int(seed)}"


def static_dir(config: str, *, root=None) -> pathlib.Path:
    return pathlib.Path(root or MODELS_DIR) / CONFIGS[config].static_config / "static"


def selection_path(*, root=None) -> pathlib.Path:
    return pathlib.Path(root or MODELS_DIR) / SELECTION_FILE


def read_done(path) -> dict | None:
    """A finished job's record (``config.json`` or ``static.json``) or None."""
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    rec = sio.read_json(p)
    return rec if rec.get("status") == "done" else None


def spec_digest(spec: dict) -> str:
    return sio.digest(spec)


def load_model(config: str, arm: str, seed: int, *, root=None, check: bool = True):
    """``(net, record)`` of a deployed model: ``net`` from ``nets.load_net`` (carries its operator forecaster),
    ``record`` the job's ``config.json``. ``check`` verifies the file and parameter digests."""
    from slicemech.control import nets
    d = model_dir(config, arm, seed, root=root)
    rec = read_done(d / CONFIG_FILE)
    if rec is None:
        raise FileNotFoundError(f"no finished model at {d}")
    path = d / MODEL_FILE
    if check and sio.file_sha256(path) != rec["result"]["model_sha256"]:
        raise ValueError(f"{path} differs from the SHA-256 recorded in its config.json")
    net = nets.load_net(path)
    if check and nets.param_digest(net) != rec["result"]["param_digest"]:
        raise ValueError(f"{path}: parameter digest differs from the record")
    net.eval()
    return net, rec


def load_static(config: str, *, root=None):
    """The static fit of a configuration as a ``control.static.StaticFit`` (usable as an episode controller)."""
    import numpy as np
    from slicemech.control.static import StaticFit
    d = static_dir(config, root=root)
    rec = read_done(d / STATIC_FILE)
    if rec is None:
        raise FileNotFoundError(f"no finished static fit at {d}")
    cand = np.load(d / STATIC_CANDIDATES)
    return StaticFit(np.asarray(rec["w"], float), float(rec["welfare_train"]), rec["method"],
                     cand["candidates"], cand["scores"], float(rec["seconds"]), dict(rec["info"]))


def load_selection(*, root=None) -> dict:
    rec = read_done(selection_path(root=root))
    if rec is None:
        raise FileNotFoundError(f"no DEV hyper-parameter selection at {selection_path(root=root)}")
    return rec


def selected_cell(arm: str, *, root=None) -> Cell:
    return cell_by_id(load_selection(root=root)["arms"][arm]["cell"])


#: TRAIN usage-tariff caches (``e2_manipulation.usage_tariff``; one per protected checkpoint SHA-256).
USAGE_TARIFF_DIR = RESULTS_DIR / "raw" / "train" / "e2" / "usage_tariff"
#: TRAIN usage-tariff fits of the E1 sweep configurations (pre-freeze audit E5-7/F8): one
#: ``<config>/usage_nu.json`` per non-default configuration, fitted on TRAIN before the freeze and loaded (never
#: refitted) on TEST. The default configuration uses the shared E2-E4 cache in ``USAGE_TARIFF_DIR``.
SWEEP_TARIFF_DIR = RESULTS_DIR / "raw" / "train" / "e1"
SWEEP_TARIFF_FILE = "usage_nu.json"
#: E7 NR rate calibration (TRAIN; ``e7_nr --stage calibrate``): chosen radius, NR per-PRB rates (fluid_cal) and the
#: ns-3 binary/library SHA-256 they were measured with. Frozen with the models (pre-freeze audit F6).
NR_CALIBRATION_FILE = RESULTS_DIR / "raw" / "e7" / "nr_calibration.json"
#: Files that a driver reads together with a frozen asset: a model's job record (E5 reads lr, M and log_sigma from
#: it) and a static fit's TRAIN candidates (``load_static``). :func:`verify_frozen` checks them with their asset.
COMPANIONS = {MODEL_FILE: (CONFIG_FILE,), STATIC_FILE: (STATIC_CANDIDATES,)}


def sweep_tariff_path(config: str) -> pathlib.Path:
    """Frozen TRAIN usage-tariff file of an E1 sweep configuration (:data:`SWEEP_TARIFF_DIR`)."""
    if config not in CONFIGS or config == DEFAULT:
        raise ValueError(f"{config!r} is not a sweep configuration (the default uses USAGE_TARIFF_DIR)")
    return SWEEP_TARIFF_DIR / config / SWEEP_TARIFF_FILE


def checkpoint_paths(*, root=None, tariffs: bool = True) -> list[pathlib.Path]:
    """Every asset an experiment driver loads on TEST, for ``puffer_fetch.freeze_block``: each deployed model file
    with its ``config.json`` job record, each static fit with its ``candidates.npz`` and ``selection.json`` below
    ``root``. Tuning runs of unselected cells are excluded (their selected copies are listed under
    ``default/<arm>/seed<k>``). With ``tariffs`` (default; full registry only) the TRAIN assets outside the models
    root are listed too: the E2-E4 usage-tariff caches, the E1 sweep tariffs (:func:`sweep_tariff_path`) and the E7
    NR calibration (:data:`NR_CALIBRATION_FILE`), so every asset can be checked against the freeze
    (:func:`verify_frozen`; pre-freeze audit E5-7/F8 and F6)."""
    base = pathlib.Path(root or MODELS_DIR)
    out = []
    for p in sorted(base.glob(f"*/*/seed*/{MODEL_FILE}")):
        if "tune" not in p.parts:
            out += [p] + [p.parent / c for c in COMPANIONS[MODEL_FILE] if (p.parent / c).is_file()]
    for p in sorted(base.glob(f"*/static/{STATIC_FILE}")):
        out += [p] + [p.parent / c for c in COMPANIONS[STATIC_FILE] if (p.parent / c).is_file()]
    sel = selection_path(root=root)
    out += [sel] if sel.is_file() else []
    if tariffs and root is None:
        if USAGE_TARIFF_DIR.is_dir():
            out += sorted(USAGE_TARIFF_DIR.glob("nu_protected_seed*.json"))
        out += [p for p in (sweep_tariff_path(c) for c in CONFIGS if c != DEFAULT) if p.is_file()]
        out += [NR_CALIBRATION_FILE] if NR_CALIBRATION_FILE.is_file() else []
    return out


def _rel(p: pathlib.Path) -> str:
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def verify_frozen(paths, freeze_file) -> list[str]:
    """TEST gate of every asset a driver loads (round-3 review: only E0, E1 and E7 used to check): each path must be
    listed in the ``freeze`` block of ``freeze_file`` with its current SHA-256 (PermissionError otherwise).

    A ``model.pt`` or ``static.json`` brings its :data:`COMPANIONS` (``config.json``; ``candidates.npz``): each
    companion that exists, or that the freeze block lists, is checked too (pre-freeze audit E5-7/F8: E5 reads the
    hyper-parameters from the unfrozen ``config.json``). Returns the relative paths checked."""
    from slicemech.data.puffer_fetch import parse_freeze
    ck = parse_freeze(pathlib.Path(freeze_file).read_text(encoding="utf-8"))["checkpoints"]
    todo = []
    for p in paths:
        p = pathlib.Path(p).resolve()
        todo.append(p)
        for c in COMPANIONS.get(p.name, ()):
            q = p.parent / c
            if q.is_file() or _rel(q) in ck:
                todo.append(q)
    out = []
    for p in todo:
        rel = _rel(p)
        if rel in out:
            continue
        if not p.is_file() or ck.get(rel) != sio.file_sha256(p):
            raise PermissionError(f"{rel} is not frozen (missing from the freeze block or changed since the freeze)")
        out.append(rel)
    return out


def changed_options(args, parser, names) -> list[str]:
    """Command-line options among ``names`` whose parsed value differs from the parser default. On TEST every
    protocol choice (dates, seeds, arms, model locations, hyper-parameters, grids) must come from the frozen code,
    so drivers refuse these (round-3 review)."""
    bad = []
    for name in names:
        dest = name.lstrip("-").replace("-", "_")
        if not hasattr(args, dest):
            continue
        if getattr(args, dest) != parser.get_default(dest):
            bad.append(name)
    return bad


def refuse_test_overrides(split: str, args, parser, names) -> None:
    """SystemExit when ``split`` is TEST and any option in ``names`` was changed (:func:`changed_options`)."""
    if split != "test":
        return
    bad = changed_options(args, parser, names)
    if bad:
        raise SystemExit(f"refusing --split test with {', '.join(bad)}: on TEST every protocol choice comes from the "
                         "frozen code (round-3 review); a post-freeze change is a logged deviation, not an option")


def package_sha256() -> str:
    """SHA-256 over (relative path, file SHA-256) of every source file of ``code/slicemech`` (provenance); the same
    exclusions as the frozen code hash (``sio.source_files``: dot-files and dot-directories such as a lint cache,
    ``__pycache__``, ``*.pyc``)."""
    h = hashlib.sha256()
    base = CODE_DIR / "slicemech"
    for f in sio.source_files(base):
        h.update(f"{f.relative_to(base).as_posix()}\0{sio.file_sha256(f)}\n".encode())
    return h.hexdigest()


def config_record(cfg: RunConfig) -> dict:
    """JSON-able description of a configuration (without its seed list)."""
    d = dataclasses.asdict(cfg)
    d.pop("seeds")
    d["lead_s"] = float(d["lead_s"])
    d["rho"] = float(d["rho"])
    return d


def finite(x) -> bool:
    return x is not None and math.isfinite(float(x))
