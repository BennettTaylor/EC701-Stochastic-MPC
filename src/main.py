"""
Main experiment driver.

Two experiments produce all the CSVs consumed by generate_figures.py:

  1. Trajectory experiment
     Run each controller (det / stoch / perfect) ONCE over one billing period
     against the real NYISO 2023 realization, for a chosen (σ, N, T).
     Output: results/trajectory__<method>__sigma-<case>.csv
     (per-hour P, F, E, d, cost_e, rev_fr and the real inputs).

  2. Fig-8 experiment  (paper Fig. 8 replication, det+stoch+perfect)
     For each σ ∈ {1, M/N}:
       a. Draw R joint realizations (L, π^e, π^f, α) from the fitted
          distributions.
       b. For each realization ξ, PLAY each RH-MPC policy against ξ as the
          true environment — the controller re-solves every hour using its
          own forecasts for the horizon ahead, but state propagates under ξ.
          • deterministic MPC: plans on fitted means, acts under ξ.
          • stochastic MPC:    plans on |Ξ̄|=S fresh scenarios, acts under ξ.
          • perfect-info MPC:  plans on ξ itself (oracle), acts under ξ.
       c. Score each policy's realized (P, F) trajectory on ξ.
     Output: results/fig8_costs.csv  (long format, one row per
     method × σ × realization).

     This is Method B: per-realization MPC, which is what the paper's Fig. 8
     depicts.  A faster --fig8-mode=fixed approximation (commit once on the
     real trace, score on each ξ) is available for quick sanity runs.

Usage
-----
    python main.py                    # run both experiments with defaults
    python main.py --trajectory       # only the trajectory run
    python main.py --fig8             # only the Fig 8 experiment
    python main.py --fig8 --n-real 50 --n-scen 30 --fig8-horizon 168
    python main.py --fig8 --fig8-mode fixed       # fast (commit once on real trace)
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from data import load_dataset
from scenarios import fit_all
from mpc import cost_on_realization, DEFAULT_PARAMS
from deterministic_mpc import DeterministicMPC
from stochastic_mpc    import StochasticMPC
from perfect_info_mpc  import PerfectInfoMPC


PROJ_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────────
def write_trajectory_csv(path: str, sim) -> None:
    """Per-hour time series for one controller run (SimResult)."""
    T = sim.P.size
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t", "L_kW", "pi_e_per_kWh", "pi_f_per_kW", "alpha",
            "P_kW", "F_kW", "E_kWh", "d_kW", "cost_e_usd", "rev_fr_usd",
        ])
        for t in range(T):
            w.writerow([
                t,
                round(float(sim.L[t]),     4),
                round(float(sim.pi_e[t]),  6),
                round(float(sim.pi_f[t]),  6),
                round(float(sim.alpha[t]), 6),
                round(float(sim.P[t]),     4),
                round(float(sim.F[t]),     4),
                round(float(sim.E[t]),     4),
                round(float(sim.d[t]),     4),
                round(float(sim.cost_e[t]),6),
                round(float(sim.rev_fr[t]),6),
            ])


def write_trajectory_summary_csv(path: str, rows: list[dict]) -> None:
    """One-line-per-run summary (total energy, FR, demand, net)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_fig8_csv(path: str, rows: list[dict]) -> None:
    """Long-format: one row per (method, sigma_case, realization)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ─────────────────────────────────────────────────────────────────────────────
# Realization sampler
# ─────────────────────────────────────────────────────────────────────────────
def sample_realization(lw_load, lw_price, ln_pif, gn_alpha, T: int, seed: int):
    """One joint draw of (L, π^e, π^f, α) over T hours from the fitted distributions."""
    rng = np.random.default_rng(seed)
    s0, s1, s2, s3 = rng.integers(1 << 31, size=4)
    L   = lw_load .sample_horizon(0, T, 1, seed=int(s0))[:, 0]
    pie = lw_price.sample_horizon(0, T, 1, seed=int(s1))[:, 0]
    pif = ln_pif.sample(shape=T, seed=int(s2))
    a   = gn_alpha.sample(shape=T, seed=int(s3))
    L   = np.clip(L,   1.0,  None)
    pie = np.clip(pie, 1e-4, None)
    pif = np.clip(pif, 1e-5, None)
    a   = np.clip(a,   -1.0, 1.0)
    return L, pie, pif, a


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — single-month trajectory per controller
# ─────────────────────────────────────────────────────────────────────────────
def run_trajectory_experiment(
    ds, lw_load, lw_price, ln_pif, gn_alpha,
    T: int, N: int, pi_D: float, sigma: float, sigma_case: str,
    n_scen: int,
):
    print("\n" + "=" * 72)
    print(f" TRAJECTORY experiment   T={T}h  N={N}h  σ={sigma:g} ({sigma_case})  |Ξ̄|={n_scen}")
    print("=" * 72)

    common = dict(horizon=N, T=T, pi_D=pi_D, sigma=sigma)

    controllers = [
        DeterministicMPC(lw_load, lw_price, ln_pif, gn_alpha, **common),
        StochasticMPC(lw_load, lw_price, ln_pif, gn_alpha, n_scen=n_scen, **common),
        PerfectInfoMPC(ds.L, ds.pi_e, ds.pi_f, ds.alpha, **common),
    ]

    summary_rows = []
    for ctl in controllers:
        t0 = time.time()
        print(f"\n  running {ctl.name:14s} ...", flush=True)
        sim = ctl.simulate(ds.L, ds.pi_e, ds.pi_f, ds.alpha, verbose=False)
        dt = time.time() - t0
        print(f"    done in {dt:.1f}s   peak={sim.D_final:7.0f} kW   "
              f"net=${sim.totals['net']:,.0f}  "
              f"(energy ${sim.totals['energy']:,.0f}  "
              f"fr ${sim.totals['fr']:,.0f}  "
              f"demand ${sim.totals['demand']:,.0f})")

        tag  = f"{ctl.name}__sigma-{sigma_case}"
        path = os.path.join(RESULTS_DIR, f"trajectory__{tag}.csv")
        write_trajectory_csv(path, sim)
        print(f"    -> {os.path.relpath(path, PROJ_ROOT)}")

        summary_rows.append(dict(
            method=ctl.name, sigma_case=sigma_case, sigma=sigma,
            N=N, T=T, n_scen=(n_scen if ctl.name == "stochastic" else 0),
            wall_s=round(dt, 2),
            energy=round(sim.totals["energy"], 2),
            fr_revenue=round(sim.totals["fr"], 2),
            demand=round(sim.totals["demand"], 2),
            net=round(sim.totals["net"], 2),
            peak_kW=round(sim.D_final, 2),
        ))

    summary_path = os.path.join(RESULTS_DIR, f"trajectory_summary__sigma-{sigma_case}.csv")
    write_trajectory_summary_csv(summary_path, summary_rows)
    print(f"\n  summary -> {os.path.relpath(summary_path, PROJ_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Fig 8 replication
# ─────────────────────────────────────────────────────────────────────────────
def run_fig8_experiment(
    ds, lw_load, lw_price, ln_pif, gn_alpha,
    T: int, N: int, pi_D: float, n_scen: int, n_real: int,
    sigma_cases: dict[str, float],
    mode: str = "per-realization",
    seed_real_base: int = 10_000,
):
    print("\n" + "=" * 72)
    print(f" FIG 8 experiment   T={T}h  N={N}h  |Ξ̄|={n_scen}  |Ξ|={n_real}  mode={mode}")
    print("=" * 72)

    # Pre-sample the evaluation realizations ONCE; reuse for all (σ, method) pairs.
    print(f"\n  sampling {n_real} evaluation realizations over {T}h ...")
    realizations = [
        sample_realization(lw_load, lw_price, ln_pif, gn_alpha, T,
                           seed=seed_real_base + r)
        for r in range(n_real)
    ]

    out_rows: list[dict] = []

    for case_name, sigma in sigma_cases.items():
        print("\n" + "-" * 72)
        print(f"  CASE σ = {sigma:g}   ({case_name})")
        print("-" * 72)

        common = dict(horizon=N, T=T, pi_D=pi_D, sigma=sigma)

        # In per-realization mode the same controllers are re-simulated with ξ
        # as the environment each time (so their priors are fixed but their
        # commitments differ per realization).  In fixed mode they solve once
        # against the real trace and their commitments are reused.
        det   = DeterministicMPC(lw_load, lw_price, ln_pif, gn_alpha, **common)
        stoch = StochasticMPC(lw_load, lw_price, ln_pif, gn_alpha,
                              n_scen=n_scen, **common)

        if mode == "fixed":
            t0 = time.time()
            print("    running deterministic MPC on real trace ...", flush=True)
            det_fixed = det.simulate(ds.L, ds.pi_e, ds.pi_f, ds.alpha)
            print(f"      done in {time.time()-t0:.1f}s, peak={det_fixed.D_final:.0f} kW")

            t0 = time.time()
            print(f"    running stochastic MPC on real trace (|Ξ̄|={n_scen}) ...", flush=True)
            stoch_fixed = stoch.simulate(ds.L, ds.pi_e, ds.pi_f, ds.alpha)
            print(f"      done in {time.time()-t0:.1f}s, peak={stoch_fixed.D_final:.0f} kW, "
                  f"F mean={stoch_fixed.F.mean():.0f} kW")

        print(f"    playing policies on {n_real} realizations ...", flush=True)
        t0 = time.time()
        for r, (L_xi, pie_xi, pif_xi, a_xi) in enumerate(realizations):
            if mode == "fixed":
                # Reuse the single commitment from the real trace
                det_P,  det_F  = det_fixed.P,  det_fixed.F
                stoch_P, stoch_F = stoch_fixed.P, stoch_fixed.F
            else:
                # Re-simulate det and stoch with ξ as the environment
                det_sim = det.simulate(L_xi, pie_xi, pif_xi, a_xi)
                det_P, det_F = det_sim.P, det_sim.F

                stoch_sim = stoch.simulate(L_xi, pie_xi, pif_xi, a_xi)
                stoch_P, stoch_F = stoch_sim.P, stoch_sim.F

            # Deterministic
            phi_det, det_info = cost_on_realization(
                det_P, det_F, L_xi, pie_xi, pif_xi, a_xi, pi_D)
            out_rows.append(dict(
                method="deterministic", sigma_case=case_name, sigma=sigma,
                realization=r, phi=round(phi_det, 2),
                energy=round(det_info["energy"], 2),
                fr_revenue=round(det_info["fr"], 2),
                demand=round(det_info["demand"], 2),
                peak_kW=round(det_info["peak"], 2),
            ))

            # Stochastic
            phi_stoch, stoch_info = cost_on_realization(
                stoch_P, stoch_F, L_xi, pie_xi, pif_xi, a_xi, pi_D)
            out_rows.append(dict(
                method="stochastic", sigma_case=case_name, sigma=sigma,
                realization=r, phi=round(phi_stoch, 2),
                energy=round(stoch_info["energy"], 2),
                fr_revenue=round(stoch_info["fr"], 2),
                demand=round(stoch_info["demand"], 2),
                peak_kW=round(stoch_info["peak"], 2),
            ))

            # Perfect info: always per-realization (plans on ξ, acts on ξ)
            perf = PerfectInfoMPC(L_xi, pie_xi, pif_xi, a_xi, **common)
            perf_sim = perf.simulate(L_xi, pie_xi, pif_xi, a_xi)
            phi_perf, perf_info = cost_on_realization(
                perf_sim.P, perf_sim.F, L_xi, pie_xi, pif_xi, a_xi, pi_D)
            out_rows.append(dict(
                method="perfect_info", sigma_case=case_name, sigma=sigma,
                realization=r, phi=round(phi_perf, 2),
                energy=round(perf_info["energy"], 2),
                fr_revenue=round(perf_info["fr"], 2),
                demand=round(perf_info["demand"], 2),
                peak_kW=round(perf_info["peak"], 2),
            ))

            if (r + 1) % 5 == 0 or r == n_real - 1:
                dt = time.time() - t0
                print(f"      [{r+1:3d}/{n_real}]  cum {dt:.0f}s  "
                      f"avg {dt/(r+1):.1f}s/realization")

        # Summary for this case
        def stats(method: str):
            vals = np.array([row["phi"] for row in out_rows
                             if row["method"] == method and row["sigma_case"] == case_name])
            return vals.mean(), vals.std(), np.quantile(vals, 0.9)

        print(f"\n    summary σ={case_name}:")
        for m in ("deterministic", "stochastic", "perfect_info"):
            mu, sd, p90 = stats(m)
            print(f"      {m:14s}  mean=${mu:>9,.0f}  std=${sd:>7,.0f}  p90=${p90:>9,.0f}")

    # ── Write the long-format CSV ──────────────────────────────────────────
    path = os.path.join(RESULTS_DIR, "fig8_costs.csv")
    write_fig8_csv(path, out_rows)
    print(f"\n  -> {os.path.relpath(path, PROJ_ROOT)}  ({len(out_rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", action="store_true",
                    help="Run the single-month trajectory experiment.")
    ap.add_argument("--fig8", action="store_true",
                    help="Run the Fig 8 replication experiment.")
    # Trajectory-specific
    ap.add_argument("--traj-horizon", type=int, default=24,
                    help="MPC horizon for trajectory runs (default 24).")
    # Fig 8 specific
    ap.add_argument("--fig8-horizon", type=int, default=168,
                    help="MPC horizon for Fig 8 runs (default 168 = 7 days, per paper).")
    ap.add_argument("--n-real", type=int, default=30,
                    help="Number of evaluation realizations |Ξ| (default 30).")
    ap.add_argument("--fig8-mode", choices=("per-realization", "fixed"),
                    default="per-realization",
                    help="per-realization: replan det/stoch against each ξ "
                         "(faithful to paper). fixed: commit once on real "
                         "trace, score on ξ (≈60× faster, approximate).")
    # Shared
    ap.add_argument("--T", type=int, default=720,
                    help="Simulation length in hours (default 720 = 1 month).")
    ap.add_argument("--n-scen", type=int, default=20,
                    help="Number of stochastic scenarios |Ξ̄| per MPC step (default 20).")

    args = ap.parse_args()

    # If neither flag set, run both.
    if not args.trajectory and not args.fig8:
        args.trajectory = True
        args.fig8 = True

    # ── Load data + fit generators once ────────────────────────────────────
    print("Loading data …")
    ds_full = load_dataset()                      # full year (for fitting)
    ds_sim  = load_dataset(sim_time=args.T)       # truncated for the simulation
    pi_D    = ds_full.pi_D

    print("Fitting scenario generators on full year …")
    lw_load, lw_price, ln_pif, gn_alpha = fit_all(ds_full)
    print(f"  load   LW shrinkage = {lw_load.lw.shrinkage_intensity:.3f}")
    print(f"  price  LW shrinkage = {lw_price.lw.shrinkage_intensity:.3f}")
    print(f"  π^f    log-normal   μ_log={ln_pif.mu_log:+.3f}  σ_log={ln_pif.sigma_log:.3f}")
    print(f"  α      Gaussian     μ={gn_alpha.mu:+.4f}  σ={gn_alpha.sigma:.4f}")

    M = 720
    sigma_cases = {"undiscounted": 1.0, "discounted": M / args.fig8_horizon}

    if args.trajectory:
        # Trajectory uses σ=1 (easier to interpret);
        # σ=M/N is reserved for Fig 8 comparisons.
        run_trajectory_experiment(
            ds_sim, lw_load, lw_price, ln_pif, gn_alpha,
            T=args.T, N=args.traj_horizon, pi_D=pi_D,
            sigma=1.0, sigma_case="undiscounted",
            n_scen=args.n_scen,
        )

    if args.fig8:
        run_fig8_experiment(
            ds_sim, lw_load, lw_price, ln_pif, gn_alpha,
            T=args.T, N=args.fig8_horizon, pi_D=pi_D,
            n_scen=args.n_scen, n_real=args.n_real,
            sigma_cases=sigma_cases,
            mode=args.fig8_mode,
        )


if __name__ == "__main__":
    main()
