"""Newton solver settings shared by all levels (L0 converges in one iteration)."""
from dolfinx.fem.petsc import NonlinearProblem

OPTIONS = {
    "snes_type": "newtonls", "snes_linesearch_type": "none",
    "snes_rtol": 1e-10, "snes_atol": 1e-14, "snes_stol": 1e-12, "snes_max_it": 25,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",  # pivoting LU
}


class Newton:
    def __init__(self, F, u, bcs, prefix):
        self.problem = NonlinearProblem(F, u, bcs=bcs, petsc_options_prefix=prefix,
                                        petsc_options=OPTIONS)

    def solve(self):
        self.problem.solve()
        snes = self.problem.solver
        if snes.getConvergedReason() <= 0:
            raise RuntimeError(f"Newton failed (reason {snes.getConvergedReason()})")
        return snes.getIterationNumber()
