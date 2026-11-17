"""Standard per-case figures (identical layout for every level)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import params as P

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200})
C = lambda T: T - 273.15                                     # K -> degC


def _transformation_lines(ax, horizontal=True):
    for T, s, lab in ((P.AS, "--", "$A_s$"), (P.AF, "--", "$A_f$"),
                      (P.MS, ":", "$M_s$"), (P.MF, ":", "$M_f$")):
        (ax.axhline if horizontal else ax.axvline)(C(T), ls=s, lw=0.8, c="0.5")
        if horizontal:
            ax.text(ax.get_xlim()[1], C(T), lab + " ", ha="right", va="bottom",
                    fontsize=7, c="0.4")


def case_figures(d, field, t_s, T_s, z_s, case, level):
    title = f"{level} | D = {1e3 * case['D']:.3f} mm, I = {case['I']:.3f} A ({case['regime']})"
    t_all = np.unique(field[:, 0])
    z_all = np.unique(field[:, 1])
    nt, nz = len(t_all), len(z_all)
    Ta = field[:, 2].reshape(nt, nz)
    half = z_all <= 0.5 * P.L + 1e-9

    # 1. sensor transients
    fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
    for j, z in enumerate(z_s):
        ax.plot(t_s, C(T_s[:, j]), lw=1.2, label=f"z = {1e3 * z:.0f} mm")
    ax.axvline(P.T_OFF, c="k", lw=0.8)
    ax.set(xlabel="t (s)", ylabel="T surface (°C)", title=title, xlim=(0, P.T_END))
    _transformation_lines(ax)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(d / "figures" / "transient_sensors.png"); plt.close(fig)

    # 2. axial profiles on the half wire (+ clamp zoom)
    tau = case["tau_th"]
    times = [t for t in (0.5 * tau, tau, 2 * tau, P.T_OFF, P.T_OFF + tau, P.T_OFF + 3 * tau)
             if t <= P.T_END]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for t in times:
        i = int(np.argmin(abs(t_all - t)))
        for ax in axs:
            ax.plot(1e3 * z_all[half], C(Ta[i, half]), lw=1.2,
                    ls="-" if t <= P.T_OFF else "--", label=f"t = {t_all[i]:.0f} s")
    axs[1].set_xlim(0, 5e3 * case["L_b"])
    axs[1].axvline(1e3 * case["L_b"], c="0.5", lw=0.8)
    axs[1].text(1e3 * case["L_b"], axs[1].get_ylim()[0], " $L_b$", fontsize=7)
    for ax in axs:
        ax.set(xlabel="z (mm)", ylabel="T axis (°C)")
        _transformation_lines(ax)
    axs[0].set_title(title, fontsize=8); axs[1].set_title("clamp region", fontsize=8)
    axs[0].legend(fontsize=7)
    fig.savefig(d / "figures" / "axial_profiles.png"); plt.close(fig)

    # 3. space-time maps (half wire)
    panels = [("T axis (°C)", C(Ta), "inferno")]
    if field.shape[1] == 5:
        panels.append(("austenite fraction ξ", field[:, 4].reshape(nt, nz), "viridis"))
    fig, axs = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.4), squeeze=False,
                            layout="constrained")
    for ax, (lab, F, cmap) in zip(axs[0], panels):
        m = ax.pcolormesh(1e3 * z_all[half], t_all, F[:, half], cmap=cmap, shading="auto")
        fig.colorbar(m, ax=ax, label=lab)
        ax.axhline(P.T_OFF, c="w", lw=0.8, ls="--")
        ax.set(xlabel="z (mm)", ylabel="t (s)")
    axs[0, 0].set_title(title, fontsize=8)
    fig.savefig(d / "figures" / "space_time_maps.png"); plt.close(fig)
