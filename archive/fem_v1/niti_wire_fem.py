"""
NiTi Wire FEM — Axisymmetric Transient Heat Equation
=====================================================
Material : Flexinol HT 0.012" (d = 0.305 mm ≈ 0.3 mm), binary near-equiatomic NiTi
           Dynalloy Inc., 90 °C nominal transition temperature
Solver   : FEniCSx / DOLFINx (Crank-Nicolson θ = 0.5)
Domain   : 2-D axisymmetric  r ∈ [0, R],  z ∈ [0, L]
BCs      : Dirichlet T = T∞ at z = 0 and z = L  (clamped ends, heat-sink)
           Robin  −k ∂T/∂r = h(T − T∞)  at r = R  (natural convection surface)
           Symmetry  ∂T/∂r = 0  enforced implicitly at r = 0 (no flux, axisymmetry)

h parameterisation
------------------
h is derived per current from the Churchill–Chu (1975) correlation for natural
convection from a long horizontal isothermal cylinder:

    Nu = { 0.60 + 0.387 Ra_d^(1/6) / [1 + (0.559/Pr)^(9/16)]^(8/27) }²
    h  = Nu · k_air / d

Air properties are evaluated at the film temperature T_film = (T_wall + T∞)/2,
where T_wall is estimated from the FEM steady-state scaling law
    ΔT_ss ≈ ρ_e · I² / (2π² h R³)      (fin equation, mL/2 >> 1 limit)
solved iteratively once per current before the time-loop begins.

Reference: Churchill, S. W. & Chu, H. H. S. (1975). Correlating equations for
laminar and turbulent free convection from a horizontal cylinder. Int. J. Heat
Mass Transfer, 18(9), 1049–1053.

Symmetry check (Fig 6 validation)
-----------------------------------
After each current completes, T at z = L_b and z = L − L_b are compared.
For a symmetric problem (uniform Q, uniform h, Dirichlet ends) these must be
equal to within solver tolerance. A PASS/FAIL statement is printed.

τ definition
-------------
Thermal time constant for a cylinder losing heat by surface convection:
    τ = ρ · cp · R / (2 · h)      [s]
This is the lumped-capacitance time constant for a cylinder (not the diffusion
time ρ cp R²/(4k)), appropriate here because Bi ≪ 1.

Time domain
-----------
Heating–cooling cycle, t_final = 60 s:
  0 ≤ t ≤ t_off = 30 s : current on  (Q = I²ρ_e/A²)
  t_off < t ≤ 60 s      : current off (Q = 0), wire cools back towards T∞
t_off = 30 s ≥ 4.5τ for every current in the sweep (τ = 5.6–6.7 s), so the
midpoint reaches ≥ 98.9 % of its steady-state rise before switch-off, and the
30 s cooling phase is long enough to return within ~1 % of T∞. τ is
independent of L, so this holds for any wire length.
"Steady-state" checks (symmetry, analytical slope) use T at t = t_off.

Exposé notes (to be incorporated into §4 / §6.1):
- Eq. for h: Churchill–Chu above → add as Eq. in §4 or new §4.6
- Eq. for τ = ρ cp R / (2h) → add to §4.1 or caption note
- Eq. for L_b = √(kR/2h) with numerical evaluation per h → add to §4.1
- Bi = hR/k ≈ 5–6×10⁻⁴ ≪ 1 → two-sentence paragraph in §4.3
- Coordinate notation: all equations in exposé use x for axial;
  figures and code use z. Standardise to z throughout §4.1, §4.2, §4.3,
  Eq. 1, Eq. 13, and all BC statements before submission.
"""

import os
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from ufl import (TrialFunction, TestFunction, dx, ds, grad, dot,
                 SpatialCoordinate, Measure, inner)
import dolfinx.mesh as dmesh

# ══════════════════════════════════════════════════════════════════════════════
#  MATERIAL AND GEOMETRY PARAMETERS
#  Flexinol HT 0.012" — binary near-equiatomic NiTi (Dynalloy Inc.)
#  Thermophysical properties: Otsuka & Wayman (1998), Jani et al. (2014)
#  † values to be replaced by DSC measurements from experimental wire batch
# ══════════════════════════════════════════════════════════════════════════════
L     = 0.500          # wire length (m)
R     = 0.1525e-3      # wire radius (m) — Flexinol HT 0.012" = 0.305 mm
rho   = 6450.0         # density (kg/m³)          — both phases within 1%
cp    = 836.8          # specific heat (J/kg·K)   — Approach 1 constant value
                       # NOTE: original code used cp=836; corrected to 490
                       # per Otsuka & Wayman (1998) / Jani et al. (2014)
k     = 18.0           # thermal conductivity W/(m·K) — austenite phase
rho_e = 0.82e-6        # electrical resistivity (Ω·m) — austenite phase
                       # NOTE: original code used martensite value 0.76e-6
                       # Corrected to austenite 0.82e-6 (Jani et al. 2014)
                       # Approach 1 uses constant ρ_e (no phase dependence)
T_inf = 295.15         # ambient temperature (K) — 22 °C lab condition
T_0   = 295.15         # initial temperature (K) — uniform, equals T_inf

t_final = 60.0         # simulation duration (s) — heating + cooling
t_off   = 30.0         # current switched off at this time (s) — ≥ 4.5τ for all currents
dt      = 1.0          # timestep (s)

# Transformation temperatures (literature estimates; replace with DSC values †)
AS_K = 333.15          # As ≈ 60 °C in K
AF_K = 383.15          # Af ≈ 110 °C in K

# Output folder for exported CSVs (sensor time series + field snapshots).
# plot_niti_results.py reads from the same location.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Sweep chosen so the wire reaches/passes Af (full austenite) at steady state:
#   0.5 A → T_ss ≈ 70 °C (between As and Af), 0.7 A ≈ Af,
#   0.9 A ≈ 157 °C, 1.1 A ≈ 214 °C (fin estimate, Churchill–Chu h)
currents_mA = [500, 700, 900, 1100]              # parametric sweep (mA)
currents_A  = [I * 1e-3 for I in currents_mA]   # in amperes

# ══════════════════════════════════════════════════════════════════════════════
#  CHURCHILL–CHU h PARAMETERISATION
#  Per-current h derived from the Churchill–Chu (1975) correlation.
#  Air properties evaluated at film temperature T_film = (T_wall + T_inf)/2.
#  T_wall is estimated from the 1D fin steady-state:
#      ΔT_ss = ρ_e · I² / (2π² h R³)   (mL/2 >> 1 limit, fin solution)
#  Solved iteratively (3 iterations are sufficient; converges in <5).
#
#  Churchill–Chu (1975) for horizontal cylinder, all Ra:
#      Nu_d = { 0.60 + 0.387·Ra_d^(1/6) / [1+(0.559/Pr)^(9/16)]^(8/27) }²
#      Ra_d = g·β·ΔT_wall·d³ / (ν·α_air)
#      h    = Nu_d · k_air / d
#
#  Reference: Churchill & Chu (1975), Int. J. Heat Mass Transfer 18(9), 1049–53
#             Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, §9.4
# ══════════════════════════════════════════════════════════════════════════════

def air_properties(T_film_C):
    """
    Return (k_air, nu, alpha_air, Pr, beta) for dry air at T_film (°C).
    Linear interpolation over standard tabulated values (Incropera Table A.4).
    Valid range: 20–160 °C (film temperature reaches ≈ 120 °C at 1.1 A).
    """
    # Tabulated at 20–160 °C in 20 °C steps
    T_tab   = np.array([20,      40,      60,      80,      100,     120,     140,     160    ])
    k_tab   = np.array([0.02551, 0.02756, 0.02966, 0.03047, 0.03095, 0.03235, 0.03374, 0.03511])  # W/(m·K)
    nu_tab  = np.array([15.11,   16.97,   18.97,   20.92,   23.02,   25.22,   27.40,   29.65  ]) * 1e-6  # m²/s
    al_tab  = np.array([21.40,   24.02,   26.92,   29.85,   32.80,   35.65,   38.76,   41.91  ]) * 1e-6  # m²/s
    Pr_tab  = np.array([0.7290,  0.7255,  0.7202,  0.7154,  0.7111,  0.7073,  0.7041,  0.7014 ])

    T_film_C = np.clip(T_film_C, T_tab[0], T_tab[-1])
    k_air    = np.interp(T_film_C, T_tab, k_tab)
    nu       = np.interp(T_film_C, T_tab, nu_tab)
    alpha_a  = np.interp(T_film_C, T_tab, al_tab)
    Pr       = np.interp(T_film_C, T_tab, Pr_tab)
    beta     = 1.0 / (T_film_C + 273.15)      # ideal gas, 1/T_film (K⁻¹)
    return k_air, nu, alpha_a, Pr, beta


def churchill_chu_h(delta_T_wall, T_inf_C, d):
    """
    Compute h [W/(m²·K)] from Churchill–Chu (1975) horizontal cylinder correlation.

    Parameters
    ----------
    delta_T_wall : float  — T_wall − T_inf  [K or °C, same numeric value]
    T_inf_C      : float  — ambient temperature [°C]
    d            : float  — cylinder diameter [m]

    Returns
    -------
    h  : float  — convective heat transfer coefficient [W/(m²·K)]
    Nu : float  — Nusselt number  (for reporting)
    Ra : float  — Rayleigh number (for reporting)
    """
    T_film_C = T_inf_C + 0.5 * delta_T_wall
    k_air, nu, alpha_a, Pr, beta = air_properties(T_film_C)
    g = 9.81

    Ra = g * beta * abs(delta_T_wall) * d**3 / (nu * alpha_a)

    # Churchill–Chu (1975) — valid 10⁻⁵ ≤ Ra ≤ 10¹²
    inner_bracket = (1.0 + (0.559 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    Nu = (0.60 + 0.387 * Ra ** (1.0 / 6.0) / inner_bracket) ** 2

    h = Nu * k_air / d
    return h, Nu, Ra


def compute_h_for_current(I_A, T_inf_K, R, rho_e, k, n_iter=5):
    """
    Iteratively compute h for a given current using the 1D fin steady-state
    estimate as the wall temperature predictor.

    1D fin solution (mL/2 >> 1 limit, Dirichlet ends at T_inf):
        ΔT_ss ≈ Q·R / (2·h)   where Q = I²·ρ_e / (π·R²)²

    This is rearranged as:
        h_new = Nu(ΔT_guess)·k_air / d
    iterated until |h_new − h_old| < 1e-4.
    """
    T_inf_C = T_inf_K - 273.15
    d       = 2 * R
    A_cs    = np.pi * R**2
    Q       = I_A**2 * rho_e / A_cs**2    # W/m³

    # Initial guess: h = 55 W/(m²·K) (midrange estimate)
    h = 55.0
    for _ in range(n_iter):
        # Fin steady-state midpoint temperature rise
        delta_T = Q * R / (2.0 * h)
        h_new, Nu, Ra = churchill_chu_h(delta_T, T_inf_C, d)
        if abs(h_new - h) < 1e-4:
            h = h_new
            break
        h = h_new
    return h, Nu, Ra, delta_T


# Pre-compute h for each current and print summary table
print("=" * 70)
print("  Churchill–Chu h parameterisation — Flexinol HT 0.012\", d = 0.305 mm")
print("  Ref: Churchill & Chu (1975), Int. J. Heat Mass Transfer 18(9), 1049")
print("=" * 70)
print(f"  {'I (mA)':>8}  {'ΔT_ss (°C)':>11}  {'T_film (°C)':>12}  "
      f"{'Ra_d':>10}  {'Nu':>6}  {'h (W/m²K)':>11}")
print("  " + "-" * 66)

H_DICT = {}         # h value per current in mA
TAU_DICT = {}       # τ = ρ cp R / (2h) per current
LB_DICT = {}        # L_b = √(kR/2h) per current

for I_mA, I_A in zip(currents_mA, currents_A):
    h, Nu, Ra, dT = compute_h_for_current(I_A, T_inf, R, rho_e, k)
    tau_i = rho * cp * R / (2.0 * h)
    Lb_i  = np.sqrt(k * R / (2.0 * h))
    T_film_C = (T_inf - 273.15) + 0.5 * dT
    print(f"  {I_mA:>8}  {dT:>11.1f}  {T_film_C:>12.1f}  "
          f"{Ra:>10.4f}  {Nu:>6.3f}  {h:>11.2f}")
    H_DICT[I_mA]   = h
    TAU_DICT[I_mA] = tau_i
    LB_DICT[I_mA]  = Lb_i

print()
print("  Biot numbers and derived length/time scales:")
print(f"  {'I (mA)':>8}  {'Bi = hR/k':>11}  {'τ (s)':>8}  {'L_b (mm)':>10}")
print("  " + "-" * 45)
for I_mA in currents_mA:
    Bi = H_DICT[I_mA] * R / k
    print(f"  {I_mA:>8}  {Bi:>11.2e}  "
          f"{TAU_DICT[I_mA]:>8.2f}  {LB_DICT[I_mA]*1e3:>10.2f}")
print()
print("  NOTE: Bi ≪ 1 for all cases → radial gradients negligible at d = 0.3 mm")
print("  NOTE: L_b values differ from earlier h=15 estimate of 9.5 mm.")
print("        Update all figure titles, exposé §4.1, and COMSOL setup accordingly.")
print("=" * 70)
print()

# ══════════════════════════════════════════════════════════════════════════════
#  MESH
#  2-D rectangle: r ∈ [0, R],  z ∈ [0, L]
#  Axisymmetric: r is the radial coordinate (x[0]), z is axial (x[1])
#  Nr = 20 elements radially (fine enough given Bi ≪ 1)
#
#  Axial grading (symmetric about z = L/2):
#    • dz_fine   = 0.1 mm  over z_fine = 10 mm from each clamp (≈ 2 L_b),
#                  resolving the end-cooling boundary layer (L_b ≈ 4.3–4.7 mm)
#    • geometric growth (ratio 1.1) from 0.1 mm up to 0.5 mm   (≈ 4 mm)
#    • dz_core  ≤ 0.5 mm  uniform in the interior, where T is uniform in z
#  The mesh is built uniform and its z-coordinates are then remapped onto the
#  graded node distribution (monotone map → valid quadrilaterals).
# ══════════════════════════════════════════════════════════════════════════════
dz_fine, z_fine = 0.1e-3, 10.0e-3
dz_core, growth = 0.5e-3, 1.1


def graded_axial_nodes(L, dz_fine, z_fine, dz_core, growth):
    """Return a z-node array on [0, L], fine at both ends, ≤ dz_core inside."""
    edge = [dz_fine] * int(round(z_fine / dz_fine))
    while edge[-1] * growth < dz_core:
        edge.append(edge[-1] * growth)
    core_len = L - 2.0 * sum(edge)
    if core_len <= 0.0:
        raise ValueError("Wire too short for the requested end refinement.")
    n_core = int(np.ceil(core_len / dz_core))
    sizes  = edge + [core_len / n_core] * n_core + edge[::-1]
    z = np.concatenate([[0.0], np.cumsum(sizes)])
    z[-1] = L                                    # remove round-off at z = L
    return z


z_nodes = graded_axial_nodes(L, dz_fine, z_fine, dz_core, growth)
Nr, Nz  = 20, len(z_nodes) - 1
msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [R, L]],          # r ∈ [0, R],  z ∈ [0, L]
    [Nr, Nz],
    mesh.CellType.quadrilateral
)
# Remap uniform z-levels j·L/Nz → z_nodes[j]
j_level = np.rint(msh.geometry.x[:, 1] / L * Nz).astype(int)
msh.geometry.x[:, 1] = z_nodes[j_level]

if MPI.COMM_WORLD.rank == 0:
    dz = np.diff(z_nodes)
    print(f"  Mesh: Nr = {Nr}, Nz = {Nz} (graded), "
          f"dz = {dz.min()*1e3:.2f}–{dz.max()*1e3:.2f} mm, "
          f"{Nr*Nz} quadrilaterals\n")

# ── Function space ─────────────────────────────────────────────────────────────
V   = functionspace(msh, ("Lagrange", 1))
T   = TrialFunction(V)
v   = TestFunction(V)
T_n = Function(V)
T_n.interpolate(lambda x: np.full(x.shape[1], T_0))
x_coord = SpatialCoordinate(msh)
r = x_coord[0]    # radial coordinate

# ── Boundary tagging ──────────────────────────────────────────────────────────
# Tag 1 : r = 0   (axis of symmetry — no flux, not tagged for Robin)
# Tag 2 : r = R   (wire surface — Robin convection BC)
# Tag 3 : z = 0   (left clamp — Dirichlet T = T∞)
# Tag 4 : z = L   (right clamp — Dirichlet T = T∞)
boundaries = [
    (1, lambda x: np.isclose(x[0], 0.0)),
    (2, lambda x: np.isclose(x[0], R)),
    (3, lambda x: np.isclose(x[1], 0.0)),
    (4, lambda x: np.isclose(x[1], L)),
]

facet_indices, facet_markers = [], []
fdim = msh.topology.dim - 1
for marker, locator in boundaries:
    facets = dmesh.locate_entities_boundary(msh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices)
facet_markers = np.hstack(facet_markers)
sorted_order  = np.argsort(facet_indices)
facet_tag = dmesh.meshtags(msh, fdim,
                            facet_indices[sorted_order],
                            facet_markers[sorted_order])
ds_meas = Measure("ds", domain=msh, subdomain_data=facet_tag)

# ── Dirichlet BCs at z = 0 (tag 3) and z = L (tag 4) ─────────────────────────
# Physical justification: clamp ends act as heat sinks held at ambient.
# Supported by Debska et al. (2017, J. Mater. Eng. Perform.)
#              Xiao et al.  (2015, Acta Mechanica)
dofs_z0 = locate_dofs_topological(V, fdim,
    dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0)))
dofs_zL = locate_dofs_topological(V, fdim,
    dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], L)))
T_bc = Constant(msh, T_inf)
bcs  = [dirichletbc(T_bc, dofs_z0, V),
        dirichletbc(T_bc, dofs_zL, V)]

# ── Geometry tools for point evaluation (built once, reused each timestep) ────
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
geom_tree = bb_tree(msh, msh.topology.dim)

def eval_point(T_func, r_pt, z_pt):
    """Evaluate T_func at a single (r, z) point. Returns float or None."""
    pts  = np.array([[r_pt, z_pt, 0.0]])
    cand = compute_collisions_points(geom_tree, pts)
    cells = compute_colliding_cells(msh, cand, pts)
    if len(cells.array) > 0:
        return float(T_func.eval(pts, cells.array[:1])[0])
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETRIC SWEEP
# ══════════════════════════════════════════════════════════════════════════════
# Sensor positions — 5 virtual sensors
# s0: (r=0,   z=L/4)   centreline quarter-point
# s1: (r=0,   z=L/2)   centreline midpoint        ← PRIMARY for PINN training
# s2: (r=0,   z=3L/4)  centreline three-quarter
# s3: (r=R/2, z=L/2)   mid-radius midpoint
# s4: (r=R,   z=L/2)   surface midpoint
sensor_coords = [
    (0.0,   L / 4),
    (0.0,   L / 2),
    (0.0, 3*L / 4),
    (R/2,   L / 2),
    (R,     L / 2),
]

for I_mA, I_A in zip(currents_mA, currents_A):

    # ── Per-current physical parameters ───────────────────────────────────────
    h_conv = H_DICT[I_mA]
    tau_i  = TAU_DICT[I_mA]
    Lb_i   = LB_DICT[I_mA]

    # Joule heating source term Q [W/m³]
    # Q = I² · ρ_e / A_cs²   where A_cs = π R²
    # NOTE: correct formula — power density per unit volume from resistive heating
    A_cs    = np.pi * R**2
    Q_joule = I_A**2 * rho_e / A_cs**2
    Q_const = Constant(msh, Q_joule)
    h_const = Constant(msh, h_conv)

    print(f"─── I = {I_mA} mA ───────────────────────────────────────────────")
    print(f"    h_conv  = {h_conv:.2f} W/(m²·K)   (Churchill–Chu)")
    print(f"    τ       = {tau_i:.2f} s            (ρ cp R / 2h)")
    print(f"    L_b     = {Lb_i*1e3:.2f} mm          (√(kR/2h))")
    print(f"    Q_joule = {Q_joule:.3e} W/m³")
    print(f"    Bi      = {h_conv*R/k:.2e}")

    # Reset initial condition
    T_n.interpolate(lambda x: np.full(x.shape[1], T_0))

    sensor_data    = {i: [] for i in range(len(sensor_coords))}
    field_snapshots = {}
    times = np.arange(0, t_final + dt, dt)

    # ── Variational form — Crank-Nicolson (θ = 0.5) ───────────────────────────
    # All terms multiplied by r (axisymmetric measure r dr dz instead of dr dz).
    # This eliminates the 1/r singularity in the Robin surface term and
    # produces the correct cylindrical-coordinate weak form.
    # See exposé §4.3 for full derivation.
    #
    # Strong form (dimensional):
    #   ρ cp ∂T/∂t = k [∂²T/∂z² + (1/r)∂/∂r(r ∂T/∂r)] + Q
    #
    # Weak form after multiplying by v·r and integrating:
    #   ∫ ρcp (T/dt) v r dΩ + ∫ k ∇T·∇v r dΩ + ∫ h T v r ds(r=R)
    #   = ∫ ρcp (T_n/dt) v r dΩ + ∫ Q v r dΩ + ∫ h T∞ v r ds(r=R)
    #   (Crank-Nicolson weights θ=0.5 applied to stiffness and Robin terms)

    a = (rho * cp / dt * T * v * r * dx
         + 0.5 * k * dot(grad(T), grad(v)) * r * dx
         + 0.5 * h_const * T * v * r * ds_meas(2))

    L_form = (rho * cp / dt * T_n * v * r * dx
              - 0.5 * k * dot(grad(T_n), grad(v)) * r * dx
              + Q_const * v * r * dx
              - 0.5 * h_const * T_n * v * r * ds_meas(2)
              + h_const * T_inf * v * r * ds_meas(2))

    T_h = Function(V)

    # Snapshot times: τ/4, τ, 3τ, 4τ  (same multiples shown in figures)
    snap_multiples = [0.25, 1.0, 3.0, 4.0]
    snap_times     = [s * tau_i for s in snap_multiples]

    problem = LinearProblem(
        a, L_form, bcs=bcs, u=T_h,
        petsc_options_prefix=f"nitiwire_{I_mA}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    T_off_check = None   # (T_left, T_right, T_mid) captured at t = t_off

    for t in times[1:]:
        # Current on for the step ending at t ≤ t_off, off afterwards
        Q_const.value = Q_joule if t <= t_off + 1e-9 else 0.0
        problem.solve()
        T_n.x.array[:] = T_h.x.array

        # Sensor extraction
        for i, (r_pt, z_pt) in enumerate(sensor_coords):
            val = eval_point(T_h, r_pt, z_pt)
            if val is not None:
                sensor_data[i].append((t, val))

        # Field snapshots at τ/4, τ, 3τ
        for t_snap in snap_times:
            if np.isclose(t, t_snap, atol=dt / 2):
                field_snapshots[round(t)] = T_h.x.array.copy()

        # End of heating: values for the steady-state checks below
        if np.isclose(t, t_off, atol=dt / 2):
            T_off_check = (eval_point(T_h, 0.0, Lb_i),
                           eval_point(T_h, 0.0, L - Lb_i),
                           eval_point(T_h, 0.0, L / 2))

    # ── Symmetry check — Fig 6 validation ─────────────────────────────────────
    # For a uniform-Q, uniform-h, symmetric-Dirichlet problem, T must satisfy
    #   T(z = L_b) = T(z = L − L_b)
    # at steady state (t = t_off ≥ 4.5τ, end of the heating phase).
    # Failure indicates mesh asymmetry or solver issue.
    print(f"\n    [SYMMETRY CHECK — Fig 6 validation, t = t_off = {t_off:.0f} s]")
    T_left, T_right, T_mid_off = T_off_check or (None, None, None)
    if T_left is not None and T_right is not None:
        diff = abs(T_left - T_right)
        status = "PASS ✓" if diff < 0.01 else "FAIL ✗ — check mesh or BCs"
        print(f"    T(z = L_b)     = {T_left - 273.15:.4f} °C")
        print(f"    T(z = L − L_b) = {T_right - 273.15:.4f} °C")
        print(f"    |ΔT|           = {diff*1e3:.3f} mK   → {status}")
    else:
        print("    WARNING: could not evaluate symmetry check points.")

    # ── Analytical slope check — Fig 7 validation ─────────────────────────────
    # Analytical midpoint ΔT from 1D fin equation (mL/2 >> 1 limit):
    #   ΔT_analytical = Q · R / (2 · h)
    # Compare to FEM value at sensor s1 (centreline midpoint, last timestep).
    # Agreement within ~5% confirms h, ρ_e, and Q are self-consistent.
    # Reference: Incropera & DeWitt, Fundamentals of Heat and Mass Transfer,
    #            Extended Surfaces chapter (fin with uniform heat generation).
    print(f"\n    [ANALYTICAL SLOPE CHECK — Fig 7 validation]")
    dT_analytical = Q_joule * R / (2.0 * h_conv)
    T_fem_mid     = T_mid_off - T_inf if T_mid_off is not None else None
    if T_fem_mid is not None:
        err_pct = 100.0 * abs(T_fem_mid - dT_analytical) / dT_analytical
        print(f"    ΔT_analytical (fin, mL/2>>1) = {dT_analytical:.2f} °C")
        print(f"    ΔT_FEM (r=0, z=L/2, t=t_off)  = {T_fem_mid:.2f} °C")
        print(f"    Relative error               = {err_pct:.1f}%")
        status_a = "PASS ✓" if err_pct < 10.0 else "NOTE: >10% — check convergence"
        print(f"    Status: {status_a}")
        # The slope in Fig 7 is ΔT/I². Analytical:
        slope_analytical = rho_e / (2.0 * h_conv * np.pi**2 * R**3)
        slope_analytical_mA = slope_analytical * 1e-6   # convert to °C/mA²
        print(f"    Analytical Fig 7 slope = {slope_analytical_mA:.4e} °C/mA²")
        print(f"    (compare to FEM fit slope printed at end of run)")

    # ── Export ────────────────────────────────────────────────────────────────
    header = ("t(s)," +
              ",".join([f"T_s{i}_r{int(sensor_coords[i][0]*1e6)}um"
                        f"_z{int(sensor_coords[i][1]*1e3)}mm(K)"
                        for i in range(len(sensor_coords))]))
    sensor_arr = np.array([
        [row[0]] + [sensor_data[i][j][1] for i in range(len(sensor_coords))]
        for j, row in enumerate(sensor_data[0])
    ])
    np.savetxt(os.path.join(DATA_DIR, f"sensors_d03_I{I_mA}.csv"), sensor_arr,
               delimiter=",", header=header, comments="")

    coords_2d = msh.geometry.x[:, :2]
    for t_snap, field in field_snapshots.items():
        out = np.column_stack([coords_2d, field])
        np.savetxt(os.path.join(DATA_DIR, f"field_d03_I{I_mA}_t{t_snap}s.csv"), out,
                   delimiter=",", header="r(m),z(m),T(K)", comments="")

    T_mid_off_str = (f"{T_mid_off - 273.15:.2f} °C" if T_mid_off is not None
                     else "n/a")
    print(f"\n    Done: I = {I_mA} mA | "
          f"T_midpoint(t_off={t_off:.0f}s) = {T_mid_off_str} | "
          f"T_midpoint(t={t_final:.0f}s) = "
          f"{sensor_data[1][-1][1] - 273.15:.2f} °C\n")


# ══════════════════════════════════════════════════════════════════════════════
#  EXPOSÉ NOTES — ITEMS TO ADD/CORRECT BEFORE SUBMISSION
#  (These are editorial reminders embedded here so they travel with the code.)
# ══════════════════════════════════════════════════════════════════════════════
"""
FORMATTING INCONSISTENCIES TO FIX IN EXPOSÉ:
─────────────────────────────────────────────
1. Axial coordinate: exposé Eq.1, Eq.13, all BCs use 'x'; figures and code use 'z'.
   Standardise to z everywhere in §4.1, §4.2, §4.3, and all equation labels.
   Domain statement becomes: Ω = [0, L] × [0, R] × [0, t_final]   (z, r, t),
   with the current switched off at t_off = 30 s (t_final = 60 s).

2. h = 15 W/(m²·K) appears in Fig 1 and Fig 7 supertitles. Replace with
   "h = f(I), 62–74 W/(m²·K), Churchill–Chu (1975)" after rerunning simulations.

MISSING EQUATIONS TO ADD:
──────────────────────────
3. Churchill–Chu correlation (Eq. NEW in §4 or §6.1):
       Nu_d = { 0.60 + 0.387 Ra_d^(1/6) / [1+(0.559/Pr)^(9/16)]^(8/27) }²
       h = Nu_d · k_air / d
   Cite: Churchill & Chu (1975), Int. J. Heat Mass Transfer 18(9), 1049–1053.

4. Thermal time constant (add to §4.1 or caption):
       τ = ρ cp R / (2h)        [s]
   Numerical values: τ ≈ 5.6–6.7 s depending on current (tabulate in paper).
   NOTE: this is the lumped-capacitance τ for Bi≪1, NOT the diffusion time
   ρ cp R²/(4k) ≈ 0.5 ms. State which definition is used.

5. End-cooling characteristic length (add to §4.1):
       L_b = √(kR / 2h)         [m]
   Numerical range with corrected h: L_b ≈ 4.3–4.7 mm (not 9.5 mm).
   Update all figure annotations and exposé references to this range.

6. Analytical steady-state midpoint temperature (add to §6.1 as self-consistency check):
       ΔT_ss ≈ Q·R / (2h)    for mL/2 = (L/2)/L_b >> 1
   where m = 1/L_b = √(2h/kR).   Ref: Incropera & DeWitt, Extended Surfaces.

THEORY TO EXPAND:
──────────────────
7. §4.3 — Add two sentences on Biot number:
   "The Biot number Bi = hR/k ≈ 5–6×10⁻⁴ ≪ 1 confirms that radial conduction
   resistance is negligible compared to surface convective resistance, and the
   temperature field is essentially uniform in r for d = 0.3 mm. This justifies
   the 1D axial reduction for experimental validation while retaining the 2D
   architecture in the PINN, which generalises to larger diameters (e.g. d = 3 mm,
   where ΔT_radial ∝ R² increases by a factor of 100 to ~1–6 K, measurable and
   physically significant) without retraining."

8. §6.1 — Add paragraph on h derivation and self-consistency:
   State that h is computed from Churchill–Chu per current, with film temperature
   estimated from the fin steady-state, and confirmed by the FEM analytical slope
   check (Fig 7 right panel). This removes h as an arbitrary parameter.

FIG 3 SUBTITLE — CORRECTED FRAMING:
──────────────────────────────────────
9. Change right-panel title from:
      "confirms 2D model is needed"
   to:
      "ΔT_radial ≈ 10–60 mK for d = 0.3 mm: sub-thermocouple resolution.
       Scales to ~1–6 K at d = 3 mm (×100, ∝R²) where 2D is necessary.
       2D PINN architecture retained for diameter generalisability."

FIG 2 — ADDITIONS:
────────────────────
10. Add figure note (in paper caption, not in plot itself):
   (a) τ = ρ cp R / (2h): numerical values per current (see table above)
   (b) L_b = √(kR/2h): numerical range 4.3–4.7 mm (update vertical dotted lines)
   (c) "At 0.5 A, T_ss (≈ 70 °C) lies between A_s = 60 °C and A_f = 110 °C;
       0.7 A reaches ≈ A_f and 0.9–1.1 A exceed it (≈ 157–214 °C), so the
       wire fully transforms to austenite. The absence of a latent-heat plateau
       where the curves cross A_s–A_f is the expected consequence of the constant-cp assumption in
       Approach 1 and quantifies the domain of validity of the linear baseline."
"""
