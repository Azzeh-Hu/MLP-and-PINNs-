"""
Verification of the FEM reference (run once; takes ~15-30 min).

1. Analytic   : L0 steady mid-span rise vs fin solution dT = Q R/(2h)    (from summaries)
2. Target     : L1 steady mid-span T vs the target used to set the current (from summaries)
3. Mesh       : L2 on the hardest case, nominal vs 2x refined in r and z (and Nr x2 alone)
   Symmetry   : max |T(z, t) - T(L - z, t)| of the nominal run (mesh/BC symmetry)
4. Time step  : L2, dt vs dt/2  (backward Euler, first order)
5. Limit      : L2 with tau_xi -> 0 must reproduce L1 during heating
Differences are max |dT| over the exported (z, t) grid. Writes results/fem/verification/report.json
"""
import json

import numpy as np

import L1_apparent_heat_capacity as L1
import L2_coupled_kinetics as L2
from core import params as P
from core.runner import ROOT, simulate

CASE = "D0305_C4"                      # mid-span crosses the whole A_s..A_f range
OUT = ROOT / "verification"


def run(model, **kw):
    case = next(c for c in P.cases() if CASE in c["name"])
    field, *_ = simulate(model, case, **kw)
    return field


def max_diff(a, b, t_max=None):
    keep = np.ones(len(a), bool) if t_max is None else a[:, 0] <= t_max + 1e-9
    return float(np.abs(a[keep, 3] - b[keep, 3]).max())


def from_summaries():
    out = {}
    for c in P.cases():
        s0 = json.loads((ROOT / "L0_linear" / c["name"] / "summary.json").read_text())
        s1 = json.loads((ROOT / "L1_apparent_heat_capacity" / c["name"] / "summary.json").read_text())
        out[c["name"]] = dict(
            L0_fin_error_K=round(s0["T_mid_at_toff_C"] - (P.T_INF - 273.15) - s0["dT_fin_analytic_K"], 4),
            L1_target_error_K=round(s1["T_mid_at_toff_C"] - s1["T_target_C"], 4),
            energy_error_rel={lv: s["energy_error_rel"] for lv, s in (("L0", s0), ("L1", s1))})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"analytic_and_target": from_summaries()}

    base = run(L2.Model)
    fine = run(L2.Model, nr=2 * P.NR, dz_fine=0.5 * P.MESH["dz_fine"], dz_core=0.5 * P.MESH["dz_core"])
    report["mesh_2x_max_dT_K"] = max_diff(base, fine)
    report["mesh_Nr2x_max_dT_K"] = max_diff(base, run(L2.Model, nr=2 * P.NR))
    z = np.unique(base[:, 1])
    Ts = base[:, 3].reshape(-1, len(z))
    report["symmetry_max_dT_K"] = float(np.abs(Ts - Ts[:, ::-1]).max())

    dt0 = P.DT
    P.DT = 0.5 * dt0
    half_dt = run(L2.Model)
    P.DT = dt0
    report["dt_half_max_dT_K"] = max_diff(base, half_dt)

    l1 = run(L1.Model)
    for tau in (P.TAU_XI, 0.01):
        l2 = run(lambda g, c: L2.Model(g, c, tau_xi=tau))
        report[f"L2_tau{tau}_vs_L1_heating_max_dT_K"] = max_diff(l2, l1, t_max=P.T_OFF)

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "analytic_and_target"}, indent=2))


if __name__ == "__main__":
    main()
