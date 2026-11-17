"""
Compare L0 / L1 / L2 for every case (run after the three level scripts).

Model errors, with L2 as the reference:
    phase gap      e_0 = T_L0 - T_L2    (cost of the single-phase assumption)
    hysteresis gap e_1 = T_L1 - T_L2    (cost of equilibrium, no-hysteresis transformation)
Each is reported over the whole domain and split into the transformation region
Omega_tr = {(z, t): A_s <= T_L2 <= A_f on heating, M_f <= T_L2 <= M_s on cooling} and its complement.

Writes results/fem/comparison/: <case>.png per case, model_errors.csv, overview.png
"""
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import params as P
from core.runner import ROOT

LEVELS = ("L0_linear", "L1_apparent_heat_capacity", "L2_coupled_kinetics")
OUT = ROOT / "comparison"
C = lambda T: T - 273.15


def load(level, case):
    f = np.loadtxt(ROOT / level / case / "data" / "field.csv.gz", delimiter=",", skiprows=1)
    t, z = np.unique(f[:, 0]), np.unique(f[:, 1])
    shape = (len(t), len(z))
    xi = f[:, 4].reshape(shape) if f.shape[1] == 5 else None
    return t, z, f[:, 3].reshape(shape), xi                      # surface T


def stats(e, mask):
    sel = lambda m: (float(np.sqrt(np.mean(e[m] ** 2))), float(np.abs(e[m]).max())) if m.any() else (0.0, 0.0)
    return sel(np.ones_like(mask)) + sel(mask) + sel(~mask)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [c["name"] for c in P.cases()
             if all((ROOT / lv / c["name"] / "data" / "field.csv.gz").exists() for lv in LEVELS)]
    rows = []
    for case in cases:
        (t, z, T0, _), (_, _, T1, xi1), (_, _, T2, xi2) = (load(lv, case) for lv in LEVELS)
        heating = (t <= P.T_OFF)[:, None]
        tr = np.where(heating, (T2 >= P.AS) & (T2 <= P.AF), (T2 >= P.MF) & (T2 <= P.MS))
        e0, e1 = T0 - T2, T1 - T2
        rows.append([case, *np.round(stats(e0, tr), 3), *np.round(stats(e1, tr), 3)])

        half = z <= 0.5 * P.L + 1e-9
        m = int(np.argmin(abs(z - 0.5 * P.L)))
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        for T, lab in ((T0, "L0 linear"), (T1, "L1 equilibrium"), (T2, "L2 kinetics")):
            ax[0].plot(t, C(T[:, m]), label=lab)
        for T, s in ((P.AS, "--"), (P.AF, "--"), (P.MS, ":"), (P.MF, ":")):
            ax[0].axhline(C(T), ls=s, c="0.6", lw=0.8)
        ax[0].set(xlabel="t (s)", ylabel="T mid-span (°C)", title=case)
        ax[0].legend(fontsize=7)
        ax[1].plot(t, xi1[:, m], label="L1"); ax[1].plot(t, xi2[:, m], label="L2")
        ax[1].set(xlabel="t (s)", ylabel="ξ mid-span", ylim=(-0.02, 1.02)); ax[1].legend(fontsize=7)
        lim = max(abs(e0[:, half]).max(), 1e-3)
        pm = ax[2].pcolormesh(1e3 * z[half], t, e0[:, half], cmap="RdBu_r", vmin=-lim, vmax=lim,
                              shading="auto")
        fig.colorbar(pm, ax=ax[2], label="$T_{L0}-T_{L2}$ (K)")
        ax[2].set(xlabel="z (mm)", ylabel="t (s)", title="phase gap")
        for a in ax[:2]:
            a.axvline(P.T_OFF, c="k", lw=0.8)
        fig.tight_layout(); fig.savefig(OUT / f"{case}.png", dpi=200); plt.close(fig)

    head = ["case"] + [f"{g}_{s}_{w}" for g in ("gap_L0", "gap_L1")
                       for w in ("all", "transf", "outside") for s in ("rms_K", "max_K")]
    with open(OUT / "model_errors.csv", "w", newline="") as f:
        csv.writer(f).writerows([head] + rows)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for dkey, mk in zip(P.DIAMETERS, "os"):
        sel = [r for r in rows if r[0].startswith(dkey)]
        lv = [r[0].split("_")[1] for r in sel]
        ax.plot(lv, [r[2] for r in sel], mk + "-", label=f"{dkey}: max |L0-L2|")
        ax.plot(lv, [r[8] for r in sel], mk + "--", label=f"{dkey}: max |L1-L2|")
    ax.set(xlabel="current level", ylabel="max model error (K)")
    ax.legend(fontsize=7)
    fig.savefig(OUT / "overview.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({r[0]: {"gap_L0_max_K": r[2], "gap_L1_max_K": r[8]} for r in rows}, indent=1))


if __name__ == "__main__":
    main()
