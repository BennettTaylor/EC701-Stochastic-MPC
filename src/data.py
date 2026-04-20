"""
Note that Claude was used to assist with writing this code.

Returns five tensor:

    L      : load [kW]                    shape (T,)
    pi_e   : electricity price [$/kWh]    shape (T,)
    pi_f   : FR capacity price [$/kW-h]   shape (T,)
    alpha  : hourly-avg FR signal [-1,1]  shape (T,)
    pi_D   : demand charge [$/kW/month]   scalar

Sources
-------
L      : data/Commercial-College.csv  (8760 h of campus load)
pi_e   : NYISO day-ahead hourly LMP, NYC zone, 2023  [gridstatus]
pi_f   : NYISO day-ahead hourly Regulation Capacity price, 2023  [gridstatus]
alpha  : synthesized (Ornstein-Uhlenbeck, mean-zero, bounded |alpha|<=1).
         Matches the statistical description the paper gives of the
         hourly-averaged PJM FR signal (Sec. II, Fig. 2). No ISO other
         than PJM publishes the raw 2-second regulation dispatch signal
         in the public domain, so either synthesis or a PJM proxy is
         required for any non-PJM implementation.
pi_D   : $20/kW-month - Con Edison SC-9 Rate III / NY commercial typical.
         Adjust to match your target utility tariff.

Units: NYISO publishes prices in $/MWh. We divide by 1000 to match the
paper's $/kWh (energy) and $/kW (FR capacity per hour of commitment).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT  = os.path.dirname(THIS_DIR)
DATA_DIR   = os.path.join(PROJ_ROOT, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")

LOAD_CSV   = os.path.join(DATA_DIR, "Commercial-College.csv")
LMP_CSV    = os.path.join(RAW_DIR, "nyiso_lmp_2023.csv")
AS_CSV     = os.path.join(RAW_DIR, "nyiso_as_2023.csv")

PI_D_DEFAULT = 20.0   # $/kW/month  (Con Edison SC-9 Rate III typical; NY)
# Reference tariffs:
#   PJM / paper campus          ~ $5.3  / kW-month
#   PG&E E-20 (CA large C&I)    ~ $19   / kW-month
#   Con Edison SC-9 Rate III    ~ $20   / kW-month
#   NYSEG/NatGrid large C&I     ~ $10-15/ kW-month


def load_campus_load(path: str = LOAD_CSV) -> np.ndarray:
    df = pd.read_csv(path)
    L = df[df.columns[0]].to_numpy(dtype=float)
    assert L.size == 8760, f"expected 8760 hourly rows, got {L.size}"
    return L


def load_nyiso_prices(
    lmp_csv: str = LMP_CSV,
    as_csv: str = AS_CSV,
    zone: str = "N.Y.C.",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pi_e [$/kWh], pi_f [$/kW-h]) aligned hour-by-hour for the year."""
    lmp = pd.read_csv(lmp_csv, parse_dates=["Interval Start"])
    lmp = lmp.sort_values("Interval Start").reset_index(drop=True)
    pi_e = lmp["LMP"].to_numpy(dtype=float) / 1000.0

    asp = pd.read_csv(as_csv, parse_dates=["Interval Start"])
    asp = asp[asp["Zone"] == zone].sort_values("Interval Start").reset_index(drop=True)
    pi_f = asp["Regulation Capacity"].to_numpy(dtype=float) / 1000.0

    n = min(pi_e.size, pi_f.size, 8760)
    return pi_e[:n], pi_f[:n]


def synth_alpha(T: int, seed: int = 42, theta: float = 0.25, sigma: float = 0.22) -> np.ndarray:
    """
    OU process on [-1, 1], mean-zero. Calibrated to approximate the
    hourly-average PJM FR signal in the paper's Fig. 2 (typical range
    [-0.3, +0.3], short-range memory).
    """
    rng = np.random.default_rng(seed)
    a = np.zeros(T)
    for t in range(1, T):
        a[t] = a[t - 1] - theta * a[t - 1] + sigma * rng.standard_normal()
    return np.clip(a, -1.0, 1.0)


@dataclass
class Dataset:
    L: np.ndarray        # kW
    pi_e: np.ndarray     # $/kWh
    pi_f: np.ndarray     # $/kW-h
    alpha: np.ndarray    # [-1, 1]
    pi_D: float          # $/kW/month
    meta: dict


def load_dataset(
    sim_time: int | None = None,
    pi_D: float = PI_D_DEFAULT,
    alpha_seed: int = 42,
) -> Dataset:
    """
    Assemble full simulation dataset.

    sim_time: truncate all series to this many hours from the start.
              If None, use the full 8760-hour year.
    """
    L = load_campus_load()
    pi_e, pi_f = load_nyiso_prices()

    T = min(L.size, pi_e.size, pi_f.size)
    if sim_time is not None:
        T = min(T, sim_time)

    L, pi_e, pi_f = L[:T], pi_e[:T], pi_f[:T]
    alpha = synth_alpha(T, seed=alpha_seed)

    meta = dict(
        T=T,
        lmp_source="NYISO DAM hourly, N.Y.C. zone, 2023 (via gridstatus)",
        as_source="NYISO DAM hourly Regulation Capacity, 2023 (via gridstatus)",
        alpha_source="synthetic OU process (no public non-PJM raw signal)",
        pi_D_source="ConEd SC-9 Rate III approx ($20/kW-month)",
        load_source="Commercial-College.csv (user-provided, 8760 h campus load)",
    )
    return Dataset(L=L, pi_e=pi_e, pi_f=pi_f, alpha=alpha, pi_D=pi_D, meta=meta)


if __name__ == "__main__":
    ds = load_dataset()
    print(f"T            : {ds.meta['T']}")
    print(f"L    [kW]    : mean={ds.L.mean():.0f}  min={ds.L.min():.0f}  max={ds.L.max():.0f}")
    print(f"pi_e [$/kWh] : mean={ds.pi_e.mean():.4f}  min={ds.pi_e.min():.4f}  max={ds.pi_e.max():.4f}")
    print(f"pi_f [$/kW]  : mean={ds.pi_f.mean():.4f}  min={ds.pi_f.min():.4f}  max={ds.pi_f.max():.4f}")
    print(f"alpha        : mean={ds.alpha.mean():+.3f}  std={ds.alpha.std():.3f}  "
          f"min={ds.alpha.min():+.2f}  max={ds.alpha.max():+.2f}")
    print(f"pi_D [$/kW/m]: {ds.pi_D}")
    print()
    for k, v in ds.meta.items():
        print(f"  {k:14s}: {v}")
