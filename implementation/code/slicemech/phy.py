"""Physical-layer constants: CQI -> spectral efficiency -> per-PRB rate.

The per-PRB rate model (experiment_design.md §2, "Rates from Raca CQI") is

    r = SE(CQI) * N_RE * slots_per_s * (1 - BLER) / 1e6      [Mbit/s per PRB]

with SE from 3GPP TS 38.214 Table 5.2.2.1-2 (4-bit CQI table 1, up to 64QAM), N_RE = 12*12 - 12 DMRS = 132
resource elements per PRB per slot, 1000 slots per second (numerology 0, 15 kHz) and BLER = 0.1, single layer.
The cell constants are read from spec/params.yaml (network.re_per_prb_slot, network.bler).
"""
from __future__ import annotations

import numpy as np

from .config import PARAMS

#: Resource elements per PRB per slot carrying PDSCH data (12 subcarriers x 12 symbols minus 12 DMRS REs).
RE_PER_PRB: int = int(PARAMS["network"]["re_per_prb_slot"])
#: Target block error rate of link adaptation; delivered fraction is (1 - BLER).
BLER: float = float(PARAMS["network"]["bler"])
#: Slots per second at numerology 0 (15 kHz subcarrier spacing, 1 ms slots).
SLOTS_PER_S: int = 1000 // int(PARAMS["network"]["slot_ms"])

# TS 38.214 Table 5.2.2.1-2 ("4-bit CQI Table 1"). Index 0 is "out of range" and is not a valid measured CQI.
#: Modulation order Q_m for CQI 1..15 (index = CQI).
CQI_QM = np.array([0, 2, 2, 2, 2, 2, 2, 4, 4, 4, 6, 6, 6, 6, 6, 6], dtype=np.int64)
#: Code rate x 1024 for CQI 1..15 (index = CQI).
CQI_CODE_RATE_X1024 = np.array([0, 78, 120, 193, 308, 449, 602, 378, 490, 616, 466, 567, 666, 772, 873, 948],
                               dtype=np.int64)
#: Spectral efficiency (information bits per RE) exactly as printed in the table (4 decimals), index = CQI.
CQI_EFFICIENCY = np.array([0.0, 0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063,
                           2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547])


def _check_cqi(cqi) -> np.ndarray:
    c = np.asarray(cqi)
    if c.size and (not np.all(np.isfinite(c)) or np.any(c != np.floor(c)) or np.any(c < 1) or np.any(c > 15)):
        raise ValueError("CQI must be an observed integer in 1..15 (TS 38.214 Table 5.2.2.1-2)")
    return c.astype(np.int64)


def spectral_efficiency(cqi) -> np.ndarray:
    """Spectral efficiency (bit per resource element) of CQI values 1..15 (array-like, any shape)."""
    return CQI_EFFICIENCY[_check_cqi(cqi)]


def cqi_to_rate(cqi, re_per_prb: int = RE_PER_PRB, bler: float = BLER, slots_per_s: int = SLOTS_PER_S) -> np.ndarray:
    """Per-PRB rate (Mbit/s per PRB) of CQI values: SE * re_per_prb * slots_per_s * (1 - bler) / 1e6.

    Range with the defaults: 0.0181 (CQI 1) .. 0.6599 (CQI 15) Mbit/s per PRB.
    """
    if not 0 <= bler < 1 or re_per_prb <= 0 or slots_per_s <= 0:
        raise ValueError("need 0 <= bler < 1, re_per_prb > 0, slots_per_s > 0")
    return spectral_efficiency(cqi) * re_per_prb * slots_per_s * (1.0 - bler) / 1e6


#: Per-PRB rate (Mbit/s per PRB) for CQI 1..15 with the default cell constants; index = CQI (index 0 = nan).
RATE_TABLE = np.r_[np.nan, cqi_to_rate(np.arange(1, 16))]
