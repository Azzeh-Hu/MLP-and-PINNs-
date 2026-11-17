# FEM reference for Joule-heated NiTi wire (FEniCSx)

Ground truth for the PINN study. Three model levels share **one** parameter file, mesh,
boundary conditions and solver, so any difference between levels is caused by the physics
each level keeps, not by the numerics.

| Level | Script | Physics | Role |
|---|---|---|---|
| L0 | `L0_linear.py` | constant properties, no transformation | baseline, same PDE as the PINN |
| L1 | `L1_apparent_heat_capacity.py` | latent heat via apparent heat capacity, ξ = ξ_eq(T), ρ_e(ξ), no hysteresis | phase-aware reference (heating) |
| L2 | `L2_coupled_kinetics.py` | T and ξ solved together, relaxation kinetics, separate heating/cooling curves | main reference (heating + cooling) |

Each script starts with the governing equation it solves and what it simplifies,
introduces or removes relative to the others.

## Governing equations (L2, full form)

```
$$rho [c_p dT/dt + dH dxi/dt] = (1/r) d/dr(r k dT/dr) + d/dz(k dT/dz) + I^2 rho_e(xi) / (pi^2 R^4)$$
tau_xi dxi/dt              = max(xi_A(T) - xi, 0) - max(xi - xi_M(T), 0)

T = T_inf at z = 0, L   |   -k dT/dr = h (T - T_inf) at r = R   |   dT/dr = 0 at r = 0
T = T_inf, xi = 0 at t = 0;   current on for 0 < t <= t_off, off afterwards
```

```
rho [c_p dT/dt + dH dxi/dt] = (1/r) d/dr(r k dT/dr) + d/dz(k dT/dz) + I^2 rho_e(xi) / (pi^2 R^4)
tau_xi dxi/dt              = max(xi_A(T) - xi, 0) - max(xi - xi_M(T), 0)

T = T_inf at z = 0, L   |   -k dT/dr = h (T - T_inf) at r = R   |   dT/dr = 0 at r = 0
T = T_inf, xi = 0 at t = 0;   current on for 0 < t <= t_off, off afterwards
```
- `xi_A`, `xi_M`: logistic curves, 1 % / 99 % transformed at A_s / A_f (heating) and M_f / M_s (cooling)
- `h = h_conv + h_rad`: Churchill–Chu (1975) plus linearised radiation, constant per case
- L1 uses `xi = xi_A(T)`, written in enthalpy form `rho [H(T) - H(T_n)] / dt`, `H = c_p T + dH xi_A(T)`
- L0 drops `xi` (martensite properties throughout)

## Test matrix

2 diameters × 5 current levels. Each current is chosen so the **phase-aware** steady mid-span
temperature hits a target (steady energy balance `I = pi R sqrt(2 R h dT / rho_e)`):

| Level | Target | Regime |
|---|---|---|
| C1 | A_s − 10 K | fully martensitic |
| C2 | A_s | onset |
| C3 | (A_s + A_f)/2 | mid-transformation |
| C4 | A_f | completing |
| C5 | A_f + 30 K | front travels toward the clamps |

Cycle: current on 0–90 s, off 90–180 s (90 s ≥ 5 τ_th for the 0.508 mm wire).
Case names encode diameter, level and current, e.g. `D0305_C3_I0603mA`.

## Outputs — `results/fem/<level>/<case>/`

| Path | Content |
|---|---|
| `data/field.csv.gz` | `t_s, z_m, T_axis_K, T_surf_K[, xi_surf]`, every 1 s on a 1 mm axial grid |
| `sensors/sensors_N{3..8}.csv` | surface temperature at `N` uniformly spaced sensors `z_i = i L/(N+1)`, every 0.1 s, **noise-free** (add noise at training time) |
| `figures/` | sensor transients, axial profiles (+ clamp zoom), space–time maps of T and ξ |
| `summary.json` | inputs, h, Bi, L_b, τ_th, peak values, solver statistics, level-specific checks |

`results/fem/comparison/` (from `compare_levels.py`): per-case overlays, the phase-gap map
`T_L0 − T_L2`, and `model_errors.csv` (RMS / max error of L0 and L1 against L2, whole domain
and inside / outside the transformation region).

`results/fem/verification/report.json` (from `verify.py`): analytic and target checks, mesh and
time-step convergence, and the L2 → L1 limit.

## Run

```bash
./run_all.sh                 # L0, L1, L2 for all cases + comparison  (Windows: run_all.bat)
./run_all.sh --verify        # + verification
python L2_coupled_kinetics.py --only D0305_C4   # one case
```
Requires a FEniCSx (DOLFINx ≥ 0.10) conda environment with matplotlib.

## Code layout

```
fem/
  L0_linear.py, L1_apparent_heat_capacity.py, L2_coupled_kinetics.py   physics (one level each)
  compare_levels.py, verify.py, run_all.sh
  core/params.py       ALL inputs: geometry, material, transformation temperatures, test matrix, numerics
  core/convection.py   Churchill–Chu + radiation
  core/mesh.py         axisymmetric graded mesh, point probes
  core/solver.py       Newton settings
  core/runner.py       time loop + export (identical for all levels)
  core/plots.py        per-case figures
```

## Open inputs (marked [TBD] in `core/params.py`)

A_s, A_f, M_s, M_f and ΔH from DSC of the purchased wire batch; emissivity; the phase
dependence of k (literature disagrees on its sign). Change them in `core/params.py` only.
