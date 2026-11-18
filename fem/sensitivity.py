"""
One-at-a-time sensitivity of the phase-aware model (L1, heating phase) to uncertain inputs
and to two model-form choices. The current is held at its nominal value (same experiment);
only the varied input changes.

Ranges (low, high), see notes; [verify] = to be checked against the purchased wire / literature:
    k        14 - 28 W/m/K      Faulkner 2000 (M / A); nominal 18 (datasheet)
    c_p      480 - 837 J/kg/K   typical DSC value [verify] / datasheet nominal
    dH       15 - 28 J/g        typical NiTi DSC range [verify]; nominal 24.2
    h        x0.8 - x1.2        Churchill-Chu correlation uncertainty
    eps      0.1 - 0.6          bare / oxidised wire [verify]; changes h_rad
    A_s,A_f  -5 K - +5 K        batch-to-batch shift [verify with DSC]
    rho_e    x0.95 - x1.05
Model-form variants: local h(T) instead of constant h; clamp conductance 1e3 / 1e4 W/m2/K
instead of T = T_inf at the clamps.

Metrics vs nominal: mid-span T at t_off, max |dT| over the field, max |dT| within 25 mm of
a clamp. Writes results/fem/sensitivity/<case>.csv and <case>.png.
Run: python sensitivity.py [--case D0305_C3]
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import L1_apparent_heat_capacity as L1
from core import params as P
from core.convection import h_total
from core.runner import ROOT, simulate

OUT = ROOT / "sensitivity"


def variations(case):
    dT = case["T_target"] - P.T_INF
    h_eps = lambda e: h_total(dT, case["D"], P.T_INF, e)
    shift = lambda d: {"AS": P.AS + d, "AF": P.AF + d, "MS": P.MS + d, "MF": P.MF + d}
    rho_e = lambda f: {"RHO_E": {k: f * v for k, v in P.RHO_E.items()}}
    return [  # (name, low/high label, params override, case override, model kwargs)
        ("k", "14", {"K": 14.0}, {}, {}), ("k", "28", {"K": 28.0}, {}, {}),
        ("c_p", "480", {"CP": 480.0}, {}, {}), ("c_p", "837", {"CP": 836.8}, {}, {}),
        ("dH", "15 J/g", {"DH": 15e3}, {}, {}), ("dH", "28 J/g", {"DH": 28e3}, {}, {}),
        ("h", "x0.8", {}, {"h": 0.8 * case["h"]}, {}), ("h", "x1.2", {}, {"h": 1.2 * case["h"]}, {}),
        ("eps", "0.1", {}, {"h": h_eps(0.1)}, {}), ("eps", "0.6", {}, {"h": h_eps(0.6)}, {}),
        ("A_s, A_f", "-5 K", shift(-5.0), {}, {}), ("A_s, A_f", "+5 K", shift(5.0), {}, {}),
        ("rho_e", "x0.95", rho_e(0.95), {}, {}), ("rho_e", "x1.05", rho_e(1.05), {}, {}),
        ("h(T) local", "on", {}, {}, {"h_local": True}),
        ("clamp h_c", "1e3", {}, {}, {"clamp_h": 1e3}), ("clamp h_c", "1e4", {}, {}, {"clamp_h": 1e4}),
    ]


def run(case, p_over, c_over, kw):
    saved = {k: getattr(P, k) for k in p_over}
    for k, v in p_over.items():
        setattr(P, k, v)
    try:
        field, *_ = simulate(lambda g, c: L1.Model(g, c, **kw), case | c_over)
    finally:
        for k, v in saved.items():
            setattr(P, k, v)
    return field


def metrics(f, f0):
    z = f[:, 1]
    mid = np.isclose(z, 0.5 * P.L) & np.isclose(f[:, 0], P.T_OFF)
    clamp = (z <= 0.025) | (z >= P.L - 0.025)
    d = f[:, 3] - f0[:, 3]
    return float(d[mid][0]), float(np.abs(d).max()), float(np.abs(d[clamp]).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="D0305_C3")
    args = ap.parse_args()
    case = next(c for c in P.cases() if args.case in c["name"])
    P.T_END = P.T_OFF                                  # heating phase only
    OUT.mkdir(parents=True, exist_ok=True)

    f0 = run(case, {}, {}, {})
    rows = []
    for name, lab, p_over, c_over, kw in variations(case):
        rows.append([name, lab, *np.round(metrics(run(case, p_over, c_over, kw), f0), 3)])
        print(rows[-1], flush=True)
    with open(OUT / f"{case['name']}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["input", "value", "dT_mid_toff_K", "max_abs_dT_K",
                                   "max_abs_dT_clamp25mm_K"]] + rows)

    names = list(dict.fromkeys(r[0] for r in rows))
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(names) + 1), layout="constrained")
    for i, n in enumerate(names):
        for r in (r for r in rows if r[0] == n):
            ax.barh(i, r[2], color="C3" if r[2] > 0 else "C0", alpha=0.8)
            ax.text(r[2], i, f" {r[1]} ", va="center", fontsize=7,
                    ha="left" if r[2] >= 0 else "right")
    ax.set_yticks(range(len(names)), names)
    ax.axvline(0, c="k", lw=0.8)
    ax.invert_yaxis()
    ax.set(xlabel="change in mid-span T at t_off (K)",
           title=f"L1 sensitivity, {case['name']} (nominal {case['T_target'] - 273.15:.0f} °C)")
    fig.savefig(OUT / f"{case['name']}.png", dpi=200)


if __name__ == "__main__":
    main()
