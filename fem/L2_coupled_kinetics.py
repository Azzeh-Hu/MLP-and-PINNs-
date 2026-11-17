"""
L2 — Coupled temperature / phase-fraction model with kinetics and hysteresis
===========================================================================
Full model of this study (no further reduction of the thermal physics):
    rho [c_p dT/dt + dH dxi/dt] = (1/r) d/dr(r k dT/dr) + d/dz(k dT/dz) + I^2 rho_e(xi) / (pi^2 R^4)
    tau_xi dxi/dt = <xi_A(T) - xi>_+ - <xi - xi_M(T)>_+

    xi      austenite volume fraction, a second field solved monolithically with T (Newton)
    xi_A(T) heating equilibrium curve (1 % at A_s, 99 % at A_f)  -> forward M -> A
    xi_M(T) cooling equilibrium curve (99 % at M_s, 1 % at M_f)  -> reverse A -> M
    <s>_+   max(s, 0) (semismooth Newton); between the two curves xi stays exactly frozen
            (hysteresis loop); latent heat is absorbed on heating and released on cooling.
    tau_xi  relaxation time: the Allen-Cahn-type kinetics of phase-field models reduced to a
            macroscopic fraction without gradient term (Biner 2017, Ch. 6); tau_xi -> 0 gives
            rate-independent transformation, and on heating L2 -> L1.

Introduced (vs L1)
    - xi as an independent field with its own evolution law (no instantaneous equilibrium)
    - separate heating and cooling branches (thermal hysteresis A_s/A_f vs M_s/M_f)
Still neglected (-> L3)
    - stress: transformation temperatures are stress-free values (free-hanging wire)
    - R-phase, phase-dependent k and c_p, T-dependent h, thermocouple heat sinking
BCs/IC as in L0; xi(t = 0) = 0 (fully martensitic). xi needs no boundary condition.

Run  : python L2_coupled_kinetics.py [--only D0305_C3]
"""
import basix.ufl
import numpy as np
import ufl
from dolfinx import fem

from core import params as P, runner
from core.mesh import SURFACE
from core.solver import Newton

ramp = lambda s: ufl.max_value(s, 0.0)           # exact <s>_+: no drift inside the loop


class Model:
    def __init__(self, geo, case, tau_xi=P.TAU_XI):
        p1 = basix.ufl.element("Lagrange", geo.msh.basix_cell(), 1)
        W = fem.functionspace(geo.msh, basix.ufl.mixed_element([p1, p1]))
        self.u, self.un = fem.Function(W), fem.Function(W)
        for f in (self.u, self.un):
            f.sub(0).interpolate(lambda x: np.full(x.shape[1], P.T_INF))
            f.sub(1).interpolate(lambda x: np.zeros(x.shape[1]))
        T, xi = ufl.split(self.u)
        Tn, xin = ufl.split(self.un)
        v, w = ufl.TestFunctions(W)
        r, dx = geo.r, geo.dx

        xi_A = P.xi_heating(T, exp=ufl.exp)
        xi_M = P.xi_cooling(T, exp=ufl.exp)
        self.on = fem.Constant(geo.msh, 1.0)
        Q = self.on * case["I"] ** 2 * P.mix(P.RHO_E, xi) / (np.pi**2 * case["R"] ** 4)
        F_T = (P.RHO * (P.CP * (T - Tn) + P.DH * (xi - xin)) / P.DT * v * r * dx
               + P.K * ufl.dot(ufl.grad(T), ufl.grad(v)) * r * dx
               - Q * v * r * dx
               + case["h"] * (T - P.T_INF) * v * r * geo.ds(SURFACE))
        F_xi = ((xi - xin) / P.DT - (ramp(xi_A - xi) - ramp(xi - xi_M)) / tau_xi) * w * r * dx

        V0, self.map0 = W.sub(0).collapse()
        V1, self.map1 = W.sub(1).collapse()
        self.T_out, self.xi_out = fem.Function(V0), fem.Function(V1)
        T_bc = fem.Function(V0)
        T_bc.x.array[:] = P.T_INF
        dofs = fem.locate_dofs_topological((W.sub(0), V0), 1, geo.clamp_facets)
        self.newton = Newton(F_T + F_xi, self.u, [fem.dirichletbc(T_bc, dofs, W.sub(0))], "L2_")
        self.case, self.ndofs = case, 2 * V0.dofmap.index_map.size_global

    def step(self, source_on):
        self.on.value = 1.0 if source_on else 0.0
        it = self.newton.solve()
        self.un.x.array[:] = self.u.x.array
        return it

    def fields(self, probe):
        self.T_out.x.array[:] = self.u.x.array[self.map0]
        self.xi_out.x.array[:] = self.u.x.array[self.map1]
        return {"T": probe(self.T_out)[:, 0], "xi": probe(self.xi_out)[:, 0]}

    def checks(self):
        return {"tau_xi_s": P.TAU_XI}


if __name__ == "__main__":
    runner.main("L2_coupled_kinetics", Model)
