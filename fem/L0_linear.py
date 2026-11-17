"""
L0 — Linear single-phase heat equation (baseline)
=================================================
Derived from the full energy balance solved in L2:
    rho [c_p dT/dt + dH dxi/dt] = (1/r) d/dr(r k dT/dr) + d/dz(k dT/dz) + I^2 rho_e(xi) / (pi^2 R^4)

Simplified / removed
    - no transformation: xi = 0 (martensite) for all t  ->  latent-heat term removed
    - rho_e = rho_e,M constant                            ->  source independent of T
    => linear PDE with constant coefficients

Solved
    rho c_p dT/dt = (1/r) d/dr(r k dT/dr) + d/dz(k dT/dz) + Q,     Q = I^2 rho_e,M / (pi^2 R^4)
    T = T_inf at z = 0, L          clamps (heat sinks)
    -k dT/dr = h (T - T_inf)       at r = R, h = h_conv + h_rad (constant per case)
    dT/dr = 0                      at r = 0 (natural in the r-weighted weak form)
    T = T_inf                      at t = 0;  Q = 0 for t > t_off

Role : same physics as the Paper-1 PINN; isolates PINN error from model error.
Check: steady mid-span rise vs the fin solution dT = Q R / (2 h)  (valid since L/2 >> L_b).
Run  : python L0_linear.py [--only D0305_C3]
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

        self.Q_on = case["I"] ** 2 * P.RHO_E["M"] / (np.pi**2 * case["R"] ** 4)
        self.Q = fem.Constant(geo.msh, self.Q_on)
        F = (P.RHO * P.CP * (T - Tn) / P.DT * v * r * dx
             + P.K * ufl.dot(ufl.grad(T), ufl.grad(v)) * r * dx
             - self.Q * v * r * dx
             + case["h"] * (T - P.T_INF) * v * r * geo.ds(SURFACE))

        bc = fem.dirichletbc(np.float64(P.T_INF),
                             fem.locate_dofs_topological(V, 1, geo.clamp_facets), V)
        self.newton = Newton(F, T, [bc], "L0_")
        self.case, self.ndofs = case, V.dofmap.index_map.size_global

    def step(self, source_on):
        self.Q.value = self.Q_on if source_on else 0.0
        it = self.newton.solve()
        self.Tn.x.array[:] = self.T.x.array
        return it

    def fields(self, probe):
        return {"T": probe(self.T)[:, 0]}

    def checks(self):
        return {"dT_fin_analytic_K": self.Q_on * self.case["R"] / (2.0 * self.case["h"])}


if __name__ == "__main__":
    runner.main("L0_linear", Model)
