"""Axisymmetric (r, z) mesh, boundary tags and point probes.

Domain r in [0, R], z in [0, L]; x[0] = r, x[1] = z. Quadrilaterals, uniform in r,
graded in z: fine next to the clamps (end-cooling layer, L_b ~ 4-7 mm), coarse inside.
Boundary tags: 2 = wire surface r = R, 3 = clamp faces z = 0, L. The axis r = 0 needs no condition: the weak form
is weighted by r, so the symmetry condition dT/dr = 0 is natural.
"""
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from dolfinx import geometry
import ufl

SURFACE, CLAMPS = 2, 3
QUAD = {"quadrature_degree": 4}     # fixed: auto-estimate for exp()/sqrt() is ~6x slower


def graded_z(L, dz_fine, z_fine, dz_core, growth):
    edge = [dz_fine] * int(round(z_fine / dz_fine))
    while edge[-1] * growth < dz_core:
        edge.append(edge[-1] * growth)
    core = L - 2.0 * sum(edge)
    n = int(np.ceil(core / dz_core))
    z = np.concatenate([[0.0], np.cumsum(edge + [core / n] * n + edge[::-1])])
    z[-1] = L
    return z


@dataclass
class Geometry:
    msh: dmesh.Mesh
    dx: ufl.Measure
    ds: ufl.Measure
    r: ufl.core.expr.Expr
    clamp_facets: np.ndarray
    R: float
    L: float


def build(R, L, nr, dz_fine, z_fine, dz_core, growth):
    z = graded_z(L, dz_fine, z_fine, dz_core, growth)
    nz = len(z) - 1
    msh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [R, L]], [nr, nz],
                                 dmesh.CellType.quadrilateral)
    level = np.rint(msh.geometry.x[:, 1] / L * nz).astype(int)
    msh.geometry.x[:, 1] = z[level]

    fdim = msh.topology.dim - 1
    surf = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], R))
    clamps = dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], L))
    facets = np.concatenate([surf, clamps])
    order = np.argsort(facets)
    marks = np.concatenate([np.full_like(surf, SURFACE), np.full_like(clamps, CLAMPS)])
    tags = dmesh.meshtags(msh, fdim, facets[order], marks[order])
    return Geometry(msh, ufl.Measure("dx", msh, metadata=QUAD),
                    ufl.Measure("ds", msh, subdomain_data=tags, metadata=QUAD),
                    ufl.SpatialCoordinate(msh)[0], clamps, R, L)


class Probe:
    """Evaluate a Function at fixed (r, z) points; cell search done once."""

    def __init__(self, msh, r, z):
        pts = np.zeros((len(r), 3))
        pts[:, 0] = np.clip(r, 0.0, None)
        pts[:, 1] = z
        tree = geometry.bb_tree(msh, msh.topology.dim)
        hits = geometry.compute_colliding_cells(
            msh, geometry.compute_collisions_points(tree, pts), pts)
        self.cells = np.array([hits.links(i)[0] for i in range(len(pts))], dtype=np.int32)
        self.pts = pts

    def __call__(self, u):
        return u.eval(self.pts, self.cells).reshape(len(self.pts), -1)   # (n_points, value_size)
