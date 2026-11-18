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
| `data/energy.csv` | global energy balance every 0.1 s: stored energy change, Joule input, surface and clamp losses, imbalance |
| `sensors/<layout>_N{3..8}.csv` | surface temperature every 0.1 s, **noise-free** (add noise at training time), for three layouts (below) |
| `figures/` | sensor transients, axial profiles (+ clamp zoom), space–time maps of T and ξ |
| `summary.json` | inputs, h, Bi, L_b, τ_th, peak values, solver statistics, level-specific checks |

Sensor layouts (`core/params.py: sensor_positions`):

| Layout | Positions | Purpose |
|---|---|---|
| `P1_uniform` | `z_i = i L/(N+1)` | naive baseline; all sensors sit in the flat mid-section |
| `P2_clamp` | 1 at mid-span, the rest alternating between clamps at 2–25 mm (≈ 5 L_b) | covers the end-cooling layer, where the gradients are |
| `P3_half` | `z_i = i (L/2)/N` on one half | tests whether the PINN infers symmetry |

`results/fem/comparison/` (from `compare_levels.py`): per-case overlays, the phase-gap map
`T_L0 − T_L2`, and `model_errors.csv` (RMS / max error of L0 and L1 against L2, whole domain
and inside / outside the transformation region).

`results/fem/verification/report.json` (from `verify.py`): analytic, target and energy checks,
mesh (2× in r and z, and 2× in r alone) and time-step convergence, symmetry, and the L2 → L1 limit.

`results/fem/verification/mms.json` (from `mms.py`): method of manufactured solutions for L0, L1
and L2, using the model classes unchanged. Observed order in the L2 norm ≈ 1.95 for T (all
levels) and ξ (L2); expected 2 for P1 elements.

`results/fem/sensitivity/<case>.csv|png` (from `sensitivity.py`): one-at-a-time sensitivity of
L1 (heating) to k, c_p, ΔH, h, ε, A_s/A_f, ρ_e, plus two model-form variants (local h(T),
finite clamp conductance). The current is held fixed.

## Run

```bash
./run_all.sh                 # L0, L1, L2 for all cases + comparison  (Windows: run_all.bat)
./run_all.sh --verify        # + MMS and convergence verification
./run_all.sh --verify --sens # + sensitivity study
python L2_coupled_kinetics.py --only D0305_C4   # one case
```
Environment: `conda env create -f environment.yml` (DOLFINx 0.11, PETSc 3.25 with MUMPS).
`--replot` on any level script redraws its figures from the saved files without re-solving.

## Code layout

```
fem/
  L0_linear.py, L1_apparent_heat_capacity.py, L2_coupled_kinetics.py   physics (one level each)
  compare_levels.py, verify.py, mms.py, sensitivity.py, run_all.sh
  core/params.py       ALL inputs: geometry, material, transformation temperatures, test matrix, numerics
  core/convection.py   Churchill–Chu + radiation
  core/mesh.py         axisymmetric graded mesh, point probes
  core/solver.py       Newton settings (MUMPS LU: the mixed T-xi system needs pivoting)
  core/energy.py       global energy balance per time step
  core/runner.py       time loop + export (identical for all levels)
  core/plots.py        per-case figures
```

## Status

All 30 runs (2 diameters × 5 currents × 3 levels), comparison, verification and sensitivity
completed on 18 Nov 2026 with the inputs currently in `core/params.py` (several still [TBD]).

**Verification** (`verification/report.json`, `mms.json`)

| Check | Result |
|---|---|
| L0 vs fin solution | ≤ 0.02 K (0.305 mm); ≤ 0.16 K (0.508 mm, not fully steady at 90 s) |
| Mesh 2× in r and z / 2× in r only | 0.0007 K / 0.00002 K |
| Time step halved | 0.13 K (backward Euler; ≈ 0.27 K error at Δt = 0.1 s) |
| Symmetry T(z) vs T(L−z) | 5·10⁻¹¹ K |
| L2 → L1 on heating as τ_ξ → 0 | 0.083 K (τ = 0.1 s), 0.008 K (τ = 0.01 s) |
| Energy balance | ≤ 2·10⁻⁴ of the Joule input, every case and level |
| MMS (L0, L1, L2; T and ξ) | observed order ≈ 1.95 (expected 2) |

The largest numerical error (≈ 0.3 K, time step) is well below the phase gap.

**Phase gap** (`comparison/model_errors.csv`, max |T_L0 − T_L2|): C1 0.03 K, C2 0.2 K,
C3 7–9 K, C4 15 K, C5 16 K. L1 vs L2 differs by up to 15 K, all of it during cooling
(L1 releases latent heat between A_s and A_f, L2 between M_s and M_f).

**Sensitivity** (`sensitivity/D0305_C3_*`, L1 heating, current fixed; change in mid-span T):
h ±20 % → −12 / +19 K; ρ_e ±5 % → ±4 K; ε 0.1–0.6 → +2 / −3 K; A_s, A_f ±5 K → ∓1 K;
k, c_p, ΔH → < 0.2 K at steady state (c_p up to 11 K during the transient);
local h(T) → 0.1 K. Clamp conductance 10³ / 10⁴ W/m²K instead of T = T_inf changes the
temperature within 25 mm of the clamps by up to 49 / 17 K and leaves mid-span unchanged.

## Open tasks

| # | Task | Type |
|---|---|---|
| 1 | DSC of the purchased wire: A_s, A_f, M_s, M_f, ΔH; replace the [TBD] values and rerun | inputs |
| 2 | Pin down h: it dominates the uncertainty. Measure it (cooling-curve fit on the real wire) or treat h as an unknown the PINN identifies | decision |
| 3 | Clamp condition: the clamp region (where P2 sensors sit) is controlled by the unknown clamp contact. Measure clamp temperature, or let the PINN identify a clamp conductance | decision |
| 4 | Confirm c_p, ρ_e, ε against the current Flexinol datasheet and the [verify] ranges in `sensitivity.py` | inputs |
| 5 | t_off: 0.508 mm C3/C4 end heating 2.3 / 1.0 K below target. Lengthen to 150 s or report as reached peak | decision |
| 6 | Validation against published data (Debska 2017 IR profiles, Zanotti 2009): needs the figures digitised | data |
| 7 | State in the paper: stress-free wire, no R-phase, phase-independent k and c_p | writing |
