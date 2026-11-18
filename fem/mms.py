"""
Code verification by the method of manufactured solutions (MMS) for L0, L1 and L2.

The model classes are used unchanged; a source f (volume) and g (surface) are added through
their `extra` hook so that a chosen T_m(r, z, t) (and xi_m for L2) solves the discrete problem:
    T_m  = T_inf + A t/t_end * s(r, z),     s = sin(pi z/L)(1 + 0.1 cos(pi r/R))
    xi_m = 0.8 t/t_end * s / 1.1
s satisfies the model's own conditions (T = T_inf at z = 0, L; dT/dr = 0 at r = 0; T(0) = T_inf).
f uses the backward-Euler difference of the exact solution, (H(T_m^{n+1}) - H(T_m^n))/dt, so
the time-stepping error cancels and the test isolates the spatial discretisation of every
term, including the nonlinear ones: latent heat, rho_e(xi), and the max() kinetics.
Expected: L2-norm error of order 2 under mesh refinement (P1 elements).

Writes results/fem/verification/mms.json
"""
import json

import numpy as np
import ufl
from dolfinx import fem

import L0_linear as L0
import L1_apparent_heat_capacity as L1
import L2_coupled_kinetics as L2
from core import mesh, params as P
from core.mesh import SURFACE
from core.runner import ROOT

L_MMS = 10e-3                  # short wire: the test is local, the length is irrelevant
N_STEPS = 10
A = 90.0                       # K: T_m sweeps through A_s..A_f
LEVELS = [(2, 10), (4, 20), (8, 40)]   # (nr, nz)


def manufactured(geo, t):
    x = ufl.SpatialCoordinate(geo.msh)
    r, z = x[0], x[1]
    s = ufl.sin(np.pi * z / L_MMS) * (1 + 0.1 * ufl.cos(np.pi * r / geo.R))
    t_end = N_STEPS * P.DT
    return P.T_INF + A * t / t_end * s, 0.8 * t / t_end * s / 1.1


def flux_terms(geo, Tm, case, v):
    """-div(k grad T_m) volume term and the Robin mismatch g on r = R (r-weighted)."""
    r = geo.r
    lap = Tm.dx(0).dx(0) + Tm.dx(0) / r + Tm.dx(1).dx(1)        # axisymmetric Laplacian
    g = -P.K * Tm.dx(0) - case["h"] * (Tm - P.T_INF)            # -k dT/dn = h(T - T_inf) + g
    return -P.K * lap * v * r * geo.dx, g * v * r * geo.ds(SURFACE)


def run_level(level, nr, nz, case):
    dz = L_MMS / nz
    geo = mesh.build(case["R"], L_MMS, nr, dz_fine=dz, z_fine=dz, dz_core=dz, growth=1.1)
    t_new, t_old = fem.Constant(geo.msh, 0.0), fem.Constant(geo.msh, 0.0)
    Tm, xim = manufactured(geo, t_new)
    Tm0, xim0 = manufactured(geo, t_old)
    Q = lambda xi: case["I"] ** 2 * P.mix(P.RHO_E, xi) / (np.pi**2 * case["R"] ** 4)
    r, dx = geo.r, geo.dx

    def extra_T(T, v, stored, stored0, Qm):
        vol, surf = flux_terms(geo, Tm, case, v)
        f_time = P.RHO * (stored - stored0) / P.DT * v * r * dx
        return -(f_time + vol - Qm * v * r * dx) + surf

    if level == "L0":
        extra = lambda T, v: extra_T(T, v, P.CP * Tm, P.CP * Tm0, Q(0.0))
        model = L0.Model(geo, case, extra=extra)
    elif level == "L1":
        H = lambda T: P.CP * T + P.DH * P.xi_heating(T, exp=ufl.exp)
        extra = lambda T, v: extra_T(T, v, H(Tm), H(Tm0), Q(P.xi_heating(Tm, exp=ufl.exp)))
        model = L1.Model(geo, case, extra=extra)
    else:
        rate = (L2.ramp(P.xi_heating(Tm, exp=ufl.exp) - xim)
                - L2.ramp(xim - P.xi_cooling(Tm, exp=ufl.exp))) / P.TAU_XI
        def extra(T, xi, v, w):
            e = extra_T(T, v, P.CP * Tm + P.DH * xim, P.CP * Tm0 + P.DH * xim0, Q(xim))
            return e - ((xim - xim0) / P.DT - rate) * w * L2.node_dx(geo.msh)
        model = L2.Model(geo, case, extra=extra)

    for n in range(1, N_STEPS + 1):
        t_old.value, t_new.value = (n - 1) * P.DT, n * P.DT
        model.step(source_on=True)

    hi = ufl.Measure("dx", geo.msh, metadata={"quadrature_degree": 8})
    norm = lambda e: np.sqrt(fem.assemble_scalar(fem.form(e**2 * r * hi)))
    T_h = model.T if level != "L2" else ufl.split(model.u)[0]
    out = {"h_z_mm": 1e3 * dz, "err_T_rel": norm(T_h - Tm) / norm(Tm - P.T_INF)}
    if level == "L2":
        out["err_xi_rel"] = norm(ufl.split(model.u)[1] - xim) / norm(xim)
    return out


def main():
    case = next(c for c in P.cases() if "D0305_C3" in c["name"])
    report = {}
    for level in ("L0", "L1", "L2"):
        rows = [run_level(level, nr, nz, case) for nr, nz in LEVELS]
        for key in [k for k in rows[0] if k.startswith("err")]:
            e = np.array([row[key] for row in rows])
            for row, order in zip(rows[1:], np.log2(e[:-1] / e[1:])):
                row[key.replace("err", "order")] = round(float(order), 3)
        report[level] = rows
        print(level, json.dumps(rows))
    out = ROOT / "verification"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mms.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
