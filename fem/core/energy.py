"""Global energy balance of the whole wire, recorded every time step.

    E(t) - E(0) = int_0^t [P_J - P_surf - P_clamp] dt
    E       = int rho (c_p T + dH xi) dV         stored (sensible + latent)
    P_J     = int Q dV                           Joule input
    P_surf  = int h (T - T_inf) dA   at r = R    convection + radiation
    P_clamp = int -k dT/dn dA        at z = 0, L conduction into the clamps

P_clamp is computed independently from the temperature gradient (not from the solver
residual), so the imbalance measures how well the mesh resolves the clamp layer and how
consistently the latent-heat and source terms are discretised. Reported as
    energy_error_rel = max_t |imbalance(t)| / (Joule energy delivered over the cycle).
"""
import numpy as np
import ufl
from dolfinx import fem

from . import params as P
from .mesh import CLAMPS, SURFACE


class EnergyBalance:
    def __init__(self, geo, T, stored_density, source, h):
        r, n, w = geo.r, ufl.FacetNormal(geo.msh), 2.0 * np.pi
        self.forms = {k: fem.form(w * f) for k, f in dict(
            E=stored_density * r * geo.dx,
            P_J=source * r * geo.dx,
            P_surf=h * (T - P.T_INF) * r * geo.ds(SURFACE),
            P_clamp=-P.K * ufl.dot(ufl.grad(T), n) * r * geo.ds(CLAMPS),
        ).items()}
        self.rows = []
        self.cum_flux = 0.0
        self.record(0.0, dt=None)

    def _eval(self):
        return {k: fem.assemble_scalar(f) for k, f in self.forms.items()}

    def record(self, t, dt):
        v = self._eval()
        if dt is not None:                       # backward Euler: fluxes at the new time level
            self.cum_flux += dt * (v["P_J"] - v["P_surf"] - v["P_clamp"])
        else:
            self.E0 = v["E"]
        self.rows.append([t, v["E"] - self.E0, v["P_J"], v["P_surf"], v["P_clamp"],
                          v["E"] - self.E0 - self.cum_flux])

    def table(self):
        return np.array(self.rows)

    def summary(self):
        a = self.table()
        W_J = np.sum(np.diff(a[:, 0]) * a[1:, 2])
        return {"energy_joule_J": float(W_J),
                "energy_error_rel": float(np.abs(a[:, 5]).max() / W_J)}
