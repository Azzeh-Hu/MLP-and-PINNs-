"""
L1 — Phase-aware, equilibrium transformation (apparent heat capacity)
=====================================================================
Derived from the full energy balance solved in L2:
    rho [c_p dT/dt + dH dxi/dt] = (1/r) d/dr(r k dT/dr) + d/dz(k dT/dz) + I^2 rho_e(xi) / (pi^2 R^4)

Introduced (vs L0)
    - latent heat of the M -> A transformation, dH
    - phase-dependent resistivity rho_e(xi) = rho_e,M + xi (rho_e,A - rho_e,M)
Simplified
    - xi is in instantaneous equilibrium with T: xi = xi_eq(T) (logistic; 1 % at A_s, 99 % at A_f)
      => dH dxi/dt = dH xi_eq'(T) dT/dt, i.e. c_eff(T) = c_p + dH xi_eq'(T)  (apparent heat capacity)
    - no hysteresis: cooling retraces the heating curve (reverse transformation at A_s..A_f)
    - c_p, k phase-independent (see core/params.py)

Solved (enthalpy form of the apparent-heat-capacity method)
    rho [H(T) - H(T_n)] / dt = div(k grad T) + Q(T),   H(T) = c_p T + dH xi_eq(T)
    The enthalpy difference is the exact time integral of c_eff(T) dT, so latent heat is
    conserved per step even when a step crosses the transformation range (Tubini et al. 2021).
    BCs and IC as in L0 (identical h per case, so L1 - L0 is purely the phase effect).

Role : phase-aware reference for heating; L2 reduces to L1 on heating as tau_xi -> 0.
Check: steady mid-span T vs the target temperature used to set the current.
Run  : python L1_apparent_heat_capacity.py [--only D0305_C3]
"""
import numpy as np
import ufl
from dolfinx import fem

from core import params as P, runner
from core.mesh import SURFACE
from core.solver import Newton


class Model:
    def __init__(self, geo, case):
        V = fem.functionspace(geo.msh, ("Lagrange", 1))
        self.T, self.Tn = fem.Function(V), fem.Function(V)
        self.T.x.array[:] = self.Tn.x.array[:] = P.T_INF
        T, Tn, v, r = self.T, self.Tn, ufl.TestFunction(V), geo.r
        dx = geo.dx

        xi = lambda T: P.xi_heating(T, exp=ufl.exp)
        H = lambda T: P.CP * T + P.DH * xi(T)
        self.on = fem.Constant(geo.msh, 1.0)
        Q = self.on * case["I"] ** 2 * P.mix(P.RHO_E, xi(T)) / (np.pi**2 * case["R"] ** 4)
        F = (P.RHO * (H(T) - H(Tn)) / P.DT * v * r * dx
             + P.K * ufl.dot(ufl.grad(T), ufl.grad(v)) * r * dx
             - Q * v * r * dx
             + case["h"] * (T - P.T_INF) * v * r * geo.ds(SURFACE))

        bc = fem.dirichletbc(np.float64(P.T_INF),
                             fem.locate_dofs_topological(V, 1, geo.clamp_facets), V)
        self.newton = Newton(F, T, [bc], "L1_")
        self.case, self.ndofs = case, V.dofmap.index_map.size_global

    def step(self, source_on):
        self.on.value = 1.0 if source_on else 0.0
        it = self.newton.solve()
        self.Tn.x.array[:] = self.T.x.array
        return it

    def fields(self, probe):
        T = probe(self.T)[:, 0]
        return {"T": T, "xi": P.xi_heating(T)}

    def checks(self):
        return {"T_target_C": self.case["T_target"] - 273.15}


if __name__ == "__main__":
    runner.main("L1_apparent_heat_capacity", Model)
