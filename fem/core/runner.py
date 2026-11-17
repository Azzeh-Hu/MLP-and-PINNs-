"""Shared time loop and export for all levels.

Per case, writes to results/fem/<level>/<case>/:
    data/field.csv.gz       t, z, T_axis, T_surf [, xi_surf]  every OUT_DT on an OUT_DZ grid (gzip)
    data/energy.csv         t, stored energy change, P_J, P_surf, P_clamp, imbalance (every DT)
    sensors/<layout>_N{n}.csv surface sensors (P1_uniform / P2_clamp / P3_half), every DT, noise-free
    figures/*.png           standard per-case figures
    summary.json            inputs, derived scales, peak values, solver statistics
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import mesh, params as P, plots

ROOT = Path(__file__).resolve().parents[2] / "results" / "fem"


def all_sensor_z():
    return np.unique(np.concatenate([P.sensor_positions(lay, n)
                                     for lay in P.SENSOR_LAYOUTS for n in P.SENSOR_COUNTS]))


def simulate(make_model, case, **mesh_kw):
    """Run one case; return the output field, sensor series and solver info (no files)."""
    t0 = time.perf_counter()
    mkw = dict(P.MESH, nr=P.NR) | mesh_kw
    geo = mesh.build(case["R"], P.L, **mkw)
    model = make_model(geo, case)

    z_out = np.arange(0.0, P.L + 0.5 * P.OUT_DZ, P.OUT_DZ)
    z_sens = all_sensor_z()
    axis = mesh.Probe(geo.msh, np.zeros_like(z_out), z_out)
    surf = mesh.Probe(geo.msh, np.full_like(z_out, case["R"]), z_out)
    sens = mesh.Probe(geo.msh, np.full_like(z_sens, case["R"]), z_sens)

    n_steps = int(round(P.T_END / P.DT))
    out_every = int(round(P.OUT_DT / P.DT))
    t_sens, T_sens, rows, iters = [], [], [], []

    def record(n):
        t = n * P.DT
        t_sens.append(t)
        T_sens.append(model.fields(sens)["T"])
        if n % out_every == 0:
            fa, fs = model.fields(axis), model.fields(surf)
            cols = [np.full_like(z_out, t), z_out, fa["T"], fs["T"]]
            if "xi" in fs:
                cols.append(fs["xi"])
            rows.append(np.column_stack(cols))

    record(0)
    for n in range(1, n_steps + 1):
        iters.append(model.step(source_on=n * P.DT <= P.T_OFF + 1e-9))
        model.energy.record(n * P.DT, P.DT)
        record(n)
    info = dict(model=model, mesh=mkw, iters=iters, z_sens=z_sens,
                wall=round(time.perf_counter() - t0, 1))
    return np.vstack(rows), np.array(t_sens), np.array(T_sens), info


def run_case(level, make_model, case, root=ROOT, **mesh_kw):
    field, t_sens, T_sens, info = simulate(make_model, case, **mesh_kw)
    model, iters, z_sens = info["model"], info["iters"], info["z_sens"]
    has_xi = field.shape[1] == 5
    d = Path(root) / level / case["name"]
    shutil.rmtree(d, ignore_errors=True)                 # no stale files from older runs
    for sub in ("data", "sensors", "figures"):
        (d / sub).mkdir(parents=True, exist_ok=True)

    head = "t_s,z_m,T_axis_K,T_surf_K" + (",xi_surf" if has_xi else "")
    np.savetxt(d / "data" / "field.csv.gz", field, delimiter=",", header=head,
               comments="", fmt=["%.1f", "%.4f", "%.4f", "%.4f"] + (["%.5f"] if has_xi else []))
    np.savetxt(d / "data" / "energy.csv", model.energy.table(), delimiter=",", comments="",
               header="t_s,dE_J,P_joule_W,P_surface_W,P_clamps_W,imbalance_J", fmt="%.6e")
    for lay in P.SENSOR_LAYOUTS:
        for n in P.SENSOR_COUNTS:
            zi = P.sensor_positions(lay, n)
            idx = [int(np.argmin(abs(z_sens - z))) for z in zi]
            h = "t_s," + ",".join(f"T_z{1e3 * z:.1f}mm_K" for z in zi)
            np.savetxt(d / "sensors" / f"{lay}_N{n}.csv",
                       np.column_stack([t_sens, T_sens[:, idx]]),
                       delimiter=",", header=h, comments="", fmt="%.4f")

    mid = np.isclose(field[:, 1], 0.5 * P.L)
    t_mid, T_mid = field[mid, 0], field[mid, 2]
    summary = dict(
        level=level, case=case["name"], regime=case["regime"],
        D_mm=1e3 * case["D"], I_A=case["I"], h_W_m2K=case["h"],
        T_target_C=case["T_target"] - 273.15,
        Bi=case["Bi"], L_b_mm=1e3 * case["L_b"], tau_th_s=case["tau_th"],
        T_mid_at_toff_C=float(np.interp(P.T_OFF, t_mid, T_mid) - 273.15),
        T_mid_max_C=float(T_mid.max() - 273.15),
        xi_mid_max=float(field[mid, 4].max()) if has_xi else None,
        newton_iters_total=int(sum(iters)), newton_iters_max=int(max(iters)),
        dofs=model.ndofs, wall_time_s=info["wall"],
        mesh=info["mesh"], dt_s=P.DT, t_off_s=P.T_OFF, t_end_s=P.T_END,
    ) | model.energy.summary() | model.checks()
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    replot(level, case, root)
    return summary


def replot(level, case, root=ROOT):
    """(Re)draw the per-case figures from the exported files only."""
    d = Path(root) / level / case["name"]
    field = np.loadtxt(d / "data" / "field.csv.gz", delimiter=",", skiprows=1)
    s5 = np.loadtxt(d / "sensors" / "P2_clamp_N5.csv", delimiter=",", skiprows=1)
    plots.case_figures(d, field, s5[:, 0], s5[:, 1:], P.sensor_positions("P2_clamp", 5),
                       case, level)


def main(level, make_model):
    ap = argparse.ArgumentParser(description=f"Run FEM level {level}")
    ap.add_argument("--only", default="", help="run cases whose name contains this text")
    ap.add_argument("--replot", action="store_true", help="redraw figures from saved data")
    args = ap.parse_args()
    for case in P.cases():
        if args.replot and args.only in case["name"]:
            replot(level, case)
        elif args.only in case["name"]:
            s = run_case(level, make_model, case)
            print(f"{level} | {s['case']:<22} T_mid(t_off) = {s['T_mid_at_toff_C']:7.2f} C  "
                  f"(target {s['T_target_C']:6.1f})  energy err {s['energy_error_rel']:.1e}  "
                  f"Newton max {s['newton_iters_max']}  {s['wall_time_s']:6.1f} s", flush=True)
