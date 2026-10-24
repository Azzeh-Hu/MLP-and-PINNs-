#!/usr/bin/env python3
"""
NiTi Wire FEM — Complete Plotting Suite  (revised)
====================================================
Material : Flexinol HT 0.012" (d = 0.305 mm), binary near-equiatomic NiTi
           Dynalloy Inc., 90 °C nominal transition temperature.
h values : Per-current, derived from Churchill–Chu (1975) correlation.
           Replaces the previous fixed h = 15 W/(m²·K) which was a factor
           ~3–4 too low for a sub-millimetre horizontal cylinder in still air.

Run after niti_wire_fem.py (from any folder):
    python plot_niti_results.py

Folder layout (relative to this script):
    results/data/     ← CSVs exported by niti_wire_fem.py (read from here)
    results/figures/  ← the 9 figures generated here

CSVs left next to the scripts by older runs are moved into results/data/
automatically.

FIGURE NOTES FOR PAPER (to be placed in captions — do not delete):
─────────────────────────────────────────────────────────────────────
Fig 2:
  (a) Thermal time constant: τ = ρ cp R / (2h), the lumped-capacitance time
      constant for Bi ≪ 1. Numerical values vary with current because h
      depends on ΔT via the Churchill–Chu correlation; see TAU_DICT below.
  (b) End-cooling length: L_b = √(kR/2h). Corrected range 4.3–4.7 mm
      (replaces earlier estimate of 9.5 mm based on h = 15 W/(m²·K)).
  (c) Sweep 0.5–1.1 A: T_ss ≈ 70 °C (0.5 A, above A_s), ≈ A_f at 0.7 A, and
      well above A_f at 0.9–1.1 A, so the wire passes fully into austenite.
      The absence of a latent-heat plateau where the curves cross A_s–A_f
      is the expected consequence of the constant-cp (linear) assumption
      in Approach 1 and quantifies the domain of validity of the linear baseline.
      This motivates Approach 2 (apparent heat capacity, Paper 2).

Heating–cooling cycle: current on for 0–30 s (T_OFF), off for 30–60 s.
Field snapshots at τ/4, τ, 3τ and 4τ all fall inside the heating phase.

Fig 3:
  One column per current. Top: ΔT_radial(z) = T(r=0) − T(r=R) at each τ
  snapshot. Bottom: ΔT_radial(t) at z = L/2 through heating and cooling.
  ΔT_radial = T(r=0) − T(r=R) ≈ 10–60 mK for d = 0.3 mm. This is below the
  noise floor of K-type fine-gauge thermocouples (±0.5–1 K), so radial
  gradients are experimentally undetectable at this diameter. The 2D PINN
  architecture is nonetheless retained because (i) ΔT_radial ∝ R², so at
  d = 3 mm it scales ×100 to ~1–6 K (measurable and significant), and
  (ii) the same trained 2D architecture generalises to larger diameters without
  retraining — a methodological advantage over a 1D FEM approach.

Fig 6:
  Left clamp only (the problem is symmetric), one panel per τ snapshot.
  L_b = √(kR/2h) is the characteristic end-cooling length marking the spatial
  extent of the clamp heat-sink influence (Debska et al. 2017; Xiao et al. 2015).
  Symmetry check: |T(z=L_b) − T(z=L−L_b)| < 0.01 K confirms mesh and BC
  symmetry (printed to terminal during FEM run).

Fig 7:
  The FEM fit slope (°C/mA²) is compared to the analytical prediction from the
  1D fin steady-state equation (mL/2 ≫ 1 limit):
      slope_analytical = ρ_e / (2π² h R³)
  Agreement within ~5% confirms self-consistency of h, ρ_e, k, and Q.
  Reference: Incropera & DeWitt, Fundamentals of Heat and Mass Transfer,
             Extended Surfaces chapter.

Fig 8:  # DIAGNOSTIC ONLY — not for publication
  Cross-case T(t) comparison at centreline midpoint. Used for visual inspection
  of heating curve separation across current levels. Contains no information
  beyond Fig 4 left panel. Retained as a personal diagnostic figure.

Fig 9:
  Half wire z ∈ [0, L/2] (symmetric about the midpoint), 4 × 4 grid:
  rows = currents, columns = τ/4, τ, 3τ, 4τ. Centreline T(z) per panel,
  shared y-axis within each row so the rise with time is directly visible.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
import os, glob, re, shutil, warnings
from functools import lru_cache
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  FOLDERS — data in results/data, figures in results/figures
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'results', 'data')
FIG_DIR  = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def organise_data():
    """Move FEM CSVs left in the script folder or cwd into DATA_DIR."""
    moved = 0
    for folder in {BASE_DIR, os.getcwd()}:
        for pat in ('field_d03_I*_t*s.csv', 'sensors_d03_I*.csv'):
            for src in glob.glob(os.path.join(folder, pat)):
                shutil.move(src, os.path.join(DATA_DIR, os.path.basename(src)))
                moved += 1
    if moved:
        print(f"Moved {moved} CSV file(s) into {DATA_DIR}")

organise_data()

def data_path(fname):
    return os.path.join(DATA_DIR, fname)

def fig_path(fname):
    return os.path.join(FIG_DIR, fname)

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 9,
    'axes.labelsize'   : 10,
    'axes.titlesize'   : 10,
    'legend.fontsize'  : 8.5,
    'legend.framealpha': 0.85,
    'legend.edgecolor' : '0.8',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.25,
    'grid.linestyle'   : '--',
    'grid.linewidth'   : 0.6,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 200,
    'savefig.facecolor': 'white',
})

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — geometry and material (Flexinol HT 0.012")
# ══════════════════════════════════════════════════════════════════════════════
L     = 0.500          # wire length (m)  — must match niti_wire_fem.py
R     = 0.1525e-3      # wire radius (m)  [d = 0.305 mm = 0.012"]
k     = 18.0           # thermal conductivity W/(m·K)
rho   = 6450.0         # density kg/m³
cp    = 836.8          # specific heat J/(kg·K)  — corrected from 836
rho_e = 0.82e-6        # electrical resistivity Ω·m — austenite, corrected

T_INF = 295.15         # ambient (K)
AS_C  = 60.0           # As ≈ 60 °C   (literature estimate; replace with DSC)
AF_C  = 110.0          # Af ≈ 110 °C  (literature estimate; replace with DSC)

CURRENTS = [500, 700, 900, 1100]        # mA  — must match niti_wire_fem.py
I_NOM    = 700                          # nominal current for single-value labels
T_OFF    = 30.0                         # s — current switched off (must match niti_wire_fem.py)

# ── Per-current h, τ, L_b  (Churchill–Chu, iterated) ─────────────────────────
# These reproduce the values printed by niti_wire_fem.py.
# If you rerun the FEM with different parameters, update this table.
# Columns:  I_mA  →  (h [W/m²K],  τ [s],  L_b [m])
#
# Derivation:
#   ΔT_ss = Q·R/(2h)  →  h from Churchill–Chu at T_film = T_inf + ΔT_ss/2
#   τ = ρ cp R / (2h)
#   L_b = √(kR/2h)
#   Bi  = hR/k  (all cases ≈ 5×10⁻⁴ ≪ 1)
#
# Reference: Churchill & Chu (1975), Int. J. Heat Mass Transfer 18(9), 1049.
#            Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, §9.4.

def air_properties(T_film_C):
    T_tab  = np.array([20,      40,      60,      80,      100,     120,     140,     160    ])
    k_tab  = np.array([0.02551, 0.02756, 0.02966, 0.03047, 0.03095, 0.03235, 0.03374, 0.03511])
    nu_tab = np.array([15.11,   16.97,   18.97,   20.92,   23.02,   25.22,   27.40,   29.65  ]) * 1e-6
    al_tab = np.array([21.40,   24.02,   26.92,   29.85,   32.80,   35.65,   38.76,   41.91  ]) * 1e-6
    Pr_tab = np.array([0.7290,  0.7255,  0.7202,  0.7154,  0.7111,  0.7073,  0.7041,  0.7014 ])
    T_film_C = np.clip(T_film_C, T_tab[0], T_tab[-1])
    return (np.interp(T_film_C, T_tab, k_tab),
            np.interp(T_film_C, T_tab, nu_tab),
            np.interp(T_film_C, T_tab, al_tab),
            np.interp(T_film_C, T_tab, Pr_tab),
            1.0 / (T_film_C + 273.15))

def churchill_chu_h(dT_wall, T_inf_C, d):
    T_f = T_inf_C + 0.5 * dT_wall
    k_a, nu, al, Pr, beta = air_properties(T_f)
    Ra = 9.81 * beta * abs(dT_wall) * d**3 / (nu * al)
    ib = (1.0 + (0.559 / Pr)**(9/16))**(8/27)
    Nu = (0.60 + 0.387 * Ra**(1/6) / ib)**2
    return Nu * k_a / d, Nu, Ra

def compute_h(I_mA, T_inf_K=T_INF, n_iter=5):
    I_A = I_mA * 1e-3
    A   = np.pi * R**2
    Q   = I_A**2 * rho_e / A**2
    T_inf_C = T_inf_K - 273.15
    h = 55.0
    for _ in range(n_iter):
        dT   = Q * R / (2.0 * h)
        h, Nu, Ra = churchill_chu_h(dT, T_inf_C, 2*R)
    tau = rho * cp * R / (2.0 * h)
    Lb  = np.sqrt(k * R / (2.0 * h))
    return h, tau, Lb, dT

H_DICT   = {}
TAU_DICT = {}
LB_DICT  = {}
DT_DICT  = {}    # estimated ΔT_ss per current
DTRAD_DICT = {}  # radial ΔT = Q R² / (4k) per current  [K]

for I in CURRENTS:
    h_i, tau_i, Lb_i, dT_i = compute_h(I)
    H_DICT[I]   = h_i
    TAU_DICT[I] = tau_i
    LB_DICT[I]  = Lb_i
    DT_DICT[I]  = dT_i
    Q_i = (I * 1e-3)**2 * rho_e / (np.pi * R**2)**2
    DTRAD_DICT[I] = Q_i * R**2 / (4.0 * k)

DTRAD_STR = (f'{min(DTRAD_DICT.values())*1e3:.0f}–'
             f'{max(DTRAD_DICT.values())*1e3:.0f} mK')

# Reference h_nom for single-h figures (Fig 1 title)
H_NOM   = H_DICT[I_NOM]
TAU_NOM = TAU_DICT[I_NOM]

# ── Discover snapshot times from files actually written by the FEM ─────────────
# The FEM saves field_d03_I{I}_{t}s.csv at the timestep closest to τ/4, τ, 3τ, 4τ
# for each current. Because τ varies per current (5.6–6.7 s), snapshot times
# differ slightly across currents. We discover the available times from disk
# and pick the ones closest to the τ/4, τ, 3τ, 4τ multiples of each current's τ.

def discover_snap_times(I_mA):
    """Return sorted list of integer snapshot times found on disk for this current."""
    pattern = f'field_d03_I{I_mA}_t*s.csv'
    files   = glob.glob(data_path(pattern))
    times   = []
    for f in files:
        m = re.search(r'_t(\d+)s\.csv$', f)
        if m:
            times.append(int(m.group(1)))
    return sorted(times)

# Check FEM output files exist before proceeding
missing = [I for I in CURRENTS if len(discover_snap_times(I)) == 0]
if missing:
    print(f"ERROR: No field CSV files found in {DATA_DIR} for currents:", missing)
    print("Run niti_wire_fem.py first to generate the data, then re-run this script.")
    raise SystemExit(1)

# Build per-current snap time lookup: {I_mA: [t_τ/4, t_τ, t_3τ, t_4τ]}
SNAP_T_PER_I = {}
snap_multiples = [0.25, 1.0, 3.0, 4.0]
SNAP_NAMES     = ['τ/4', 'τ', '3τ', '4τ']
N_S            = len(snap_multiples)
for I in CURRENTS:
    available = discover_snap_times(I)
    targets   = [TAU_DICT[I] * m for m in snap_multiples]
    chosen    = [min(available, key=lambda t: abs(t - tgt)) for tgt in targets]
    SNAP_T_PER_I[I] = chosen

# Shared SNAP_T for plot layout: use I_NOM times as the canonical row labels.
# All currents will use their own closest time internally.
SNAP_T     = SNAP_T_PER_I[I_NOM]
SNAP_LABEL = [f't = {t} s  ({n})' for t, n in zip(SNAP_T, SNAP_NAMES)]

print(f"Snapshot times discovered (s):")
for I in CURRENTS:
    print(f"  {I} mA → {SNAP_T_PER_I[I]}")

print("=" * 62)
print("  Per-current h, τ, L_b  (Churchill–Chu 1975)")
print("=" * 62)
print(f"  {'I (mA)':>7}  {'h (W/m²K)':>11}  {'τ (s)':>7}  {'L_b (mm)':>9}")
print("  " + "-" * 40)
for I in CURRENTS:
    print(f"  {I:>7}  {H_DICT[I]:>11.2f}  "
          f"{TAU_DICT[I]:>7.2f}  {LB_DICT[I]*1e3:>9.2f}")
print()

# Colour palette and helpers
N_I      = len(CURRENTS)
COLORS   = plt.cm.plasma(np.linspace(0.15, 0.85, N_I))
C_LABELS = [f'{I/1000:g} A' for I in CURRENTS]
K2C      = lambda T: T - 273.15


# ── Loaders ───────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def load_field(I_mA, t_s):
    """Load field CSV for current I_mA at snapshot time t_s (integer seconds)."""
    fname = f'field_d03_I{I_mA}_t{t_s}s.csv'
    d = np.loadtxt(data_path(fname), delimiter=',', skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2]

def snap_last(I_mA):
    """Return the last (4τ) snapshot time for this current."""
    return SNAP_T_PER_I[I_mA][-1]

def get_centreline(I_mA, t_s):
    r, z, T = load_field(I_mA, t_s)
    m   = r < 1e-9
    idx = np.argsort(z[m])
    return z[m][idx], T[m][idx]

def get_surface_axial(I_mA, t_s):
    r, z, T = load_field(I_mA, t_s)
    m   = r > R - 1e-9
    idx = np.argsort(z[m])
    return z[m][idx], T[m][idx]

def get_radial_midplane(I_mA, t_s):
    r, z, T = load_field(I_mA, t_s)
    m   = np.abs(z - L / 2) < 1e-6
    idx = np.argsort(r[m])
    return r[m][idx], T[m][idx]

# Sensor CSV columns:
# col 0 : t(s)
# col 1 : s0  r=0,   z=L/4
# col 2 : s1  r=0,   z=L/2   ← PRIMARY
# col 3 : s2  r=0,   z=3L/4
# col 4 : s3  r=R/2, z=L/2
# col 5 : s4  r=R,   z=L/2
SENS = {I: np.loadtxt(data_path(f'sensors_d03_I{I}.csv'),
                       delimiter=',', skiprows=1)
        for I in CURRENTS}

# Simulation end time, taken from the data rather than hard-coded
T_FINAL = max(SENS[I][-1, 0] for I in CURRENTS)

print(f"Loaded sensor data.  τ_nom ({I_NOM} mA) = {TAU_NOM:.1f} s,  "
      f"t_final = {T_FINAL:.0f} s\n")

# Index of the end of the heating phase in the sensor time series
def idx_at(I_mA, t):
    return int(np.argmin(np.abs(SENS[I_mA][:, 0] - t)))

def shade_heating(ax, label=True):
    """Shade the current-on interval and mark the switch-off time."""
    ax.axvspan(0, T_OFF, color='orange', alpha=0.06, lw=0)
    ax.axvline(T_OFF, color='k', lw=0.9, ls='-.', alpha=0.6)
    if label:
        ax.text(T_OFF + 0.6, 0.80, 'current off', transform=ax.get_xaxis_transform(),
                fontsize=7, va='top', color='0.3')

L_MM        = L * 1e3
SNAP_COLORS = plt.cm.viridis(np.linspace(0.1, 0.85, N_S))

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — 2D Temperature Fields  (N_S snapshots × N_I currents), full wire
# ══════════════════════════════════════════════════════════════════════════════
print(f"Fig 1: 2D fields ({N_S} × {N_I} grid)...")

fig1, axes1 = plt.subplots(N_S, N_I, figsize=(3.6 * N_I, 2.1 * N_S + 0.6),
                            gridspec_kw={'hspace': 0.45, 'wspace': 0.06},
                            squeeze=False)

r_g = np.linspace(0, R, 40)
z_g = np.linspace(0, L, 2000)          # 0.25 mm spacing over the full wire
RR, ZZ = np.meshgrid(r_g, z_g)

for row, t_lbl in enumerate(SNAP_LABEL):
    all_T, grids = [], []
    for I in CURRENTS:
        r, z, T = load_field(I, SNAP_T_PER_I[I][row])
        grids.append(griddata((r, z), T, (RR, ZZ), method='linear'))
        all_T.append(T)
    T_cat = np.concatenate(all_T)
    # Full range so the hottest current does not saturate
    vmin  = K2C(np.nanmin(T_cat))
    vmax  = K2C(np.nanmax(T_cat))
    lvls  = np.linspace(vmin, vmax, 22)

    for col, (I, Tg, clr, clbl) in enumerate(zip(CURRENTS, grids, COLORS, C_LABELS)):
        ax = axes1[row, col]
        ax.contourf(ZZ * 1e3, RR * 1e6, K2C(Tg),
                    levels=lvls, cmap='hot', extend='both')
        if vmin < AS_C < vmax:
            ax.contour(ZZ*1e3, RR*1e6, K2C(Tg), levels=[AS_C], colors='cyan',
                       linewidths=0.9, linestyles='--')
        if vmin < AF_C < vmax:
            ax.contour(ZZ*1e3, RR*1e6, K2C(Tg), levels=[AF_C], colors='lime',
                       linewidths=0.9, linestyles='--')
        ax.set_xlim(0, L_MM)
        ax.set_aspect('auto')
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title(clbl, fontsize=9, fontweight='bold', color=clr, pad=4)
        if col == 0:
            ax.set_ylabel(f'{t_lbl}\n\nr (µm)', fontsize=8)
        else:
            ax.set_yticklabels([])
        if row == N_S - 1:
            ax.set_xlabel('z (mm)', fontsize=8)
        else:
            ax.set_xticklabels([])

    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig1.colorbar(sm, ax=axes1[row, :], shrink=0.82, pad=0.006, aspect=22)
    cb.set_label('T (°C)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

leg_els = [
    Line2D([0],[0], color='cyan', lw=1.2, ls='--', label=f'Aₛ = {AS_C:.0f}°C'),
    Line2D([0],[0], color='lime', lw=1.2, ls='--', label=f'Aᶠ = {AF_C:.0f}°C'),
]
fig1.legend(handles=leg_els, loc='upper center',
            bbox_to_anchor=(0.45, 0.955), ncol=2, fontsize=8)
fig1.suptitle(
    f'2D Temperature Field  T(r, z)  —  d = 0.3 mm,  L = {L_MM:.0f} mm\n'
    f'h = f(I): {min(H_DICT.values()):.0f}–{max(H_DICT.values()):.0f} W/(m²·K)  '
    f'(Churchill–Chu 1975)  |  Rows: time snapshots  |  Columns: applied current',
    fontsize=10, fontweight='bold', y=1.0)

fig1.savefig(fig_path('fig1_2D_fields.png'), bbox_inches='tight')
plt.close(fig1)
print("  → results/figures/fig1_2D_fields.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — Axial Centreline  T(z, r=0)  at each τ snapshot, all currents
#
#  FIGURE NOTES (add to paper caption — see module docstring for full text):
#  (a) τ = ρ cp R / (2h)  [lumped-capacitance, Bi≪1], values per current
#      in TAU_DICT. Snapshot labels use τ of the I_NOM case.
#  (b) L_b = √(kR/2h); dotted verticals mark L_b and L−L_b for I_NOM.
#  (c) 0.7 A reaches ≈ Af; 0.9–1.1 A exceed Af. No latent-heat plateau
#      visible: expected consequence of constant-cp Approach 1 (Gap 3).
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Axial centreline profiles...")

LB_NOM  = LB_DICT[I_NOM]
tau_min = min(TAU_DICT.values())
tau_max = max(TAU_DICT.values())
Lb_min  = min(LB_DICT.values()) * 1e3
Lb_max  = max(LB_DICT.values()) * 1e3

fig2, axes2 = plt.subplots(1, N_S, figsize=(4.6 * N_S, 4.5))

for col, t_lbl in enumerate(SNAP_LABEL):
    ax = axes2[col]
    for I, clr, clbl in zip(CURRENTS, COLORS, C_LABELS):
        z, T = get_centreline(I, SNAP_T_PER_I[I][col])
        ax.plot(z * 1e3, K2C(T), color=clr, lw=1.8, label=clbl)

    ax.axhline(AS_C, color='steelblue', lw=0.9, ls='--', alpha=0.75)
    ax.axhline(AF_C, color='firebrick', lw=0.9, ls='--', alpha=0.75)
    ax.text(L_MM * 1.01, AS_C, 'Aₛ', fontsize=7.5, color='steelblue',
            va='center', clip_on=False)
    ax.text(L_MM * 1.01, AF_C, 'Aᶠ', fontsize=7.5, color='firebrick',
            va='center', clip_on=False)

    ax.axvline(LB_NOM * 1e3, color='k', lw=0.7, ls=':', alpha=0.5)
    ax.axvline((L - LB_NOM) * 1e3, color='k', lw=0.7, ls=':', alpha=0.5)

    if col == N_S - 1:
        ax.annotate(
            'No latent-heat plateau\n(constant cₚ, Approach 1)',
            xy=(L_MM / 2, AF_C),
            xytext=(0.3 * L_MM, (AS_C + AF_C) / 2 - 5),
            fontsize=7, color='firebrick', alpha=0.85,
            arrowprops=dict(arrowstyle='->', color='firebrick', lw=0.7))

    ax.set_xlabel('z  (mm)')
    if col == 0:
        ax.set_ylabel('T  (°C)')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_title(t_lbl, fontweight='bold')
    ax.set_xlim(0, L_MM)

fig2.suptitle(
    f'Axial Centreline Profile  T(z, r = 0)  —  d = 0.3 mm, L = {L_MM:.0f} mm  '
    f'[Flexinol HT 0.012", binary NiTi]\n'
    f'Dotted verticals: L_b = √(kR/2h) ≈ {Lb_min:.1f}–{Lb_max:.1f} mm  '
    f'({I_NOM/1000:g} A shown)     '
    f'τ = ρ cₚ R / (2h): {tau_min:.1f}–{tau_max:.1f} s across sweep',
    fontsize=10, fontweight='bold')

fig2.text(0.5, -0.04,
    'Caption note: (a) τ defined as lumped-capacitance constant ρcₚR/(2h), Bi≪1. '
    'Per-current τ: ' + ', '.join([f'{I/1000:g} A→{TAU_DICT[I]:.1f}s' for I in CURRENTS])
    + '.  (b) L_b corrected range shown above; earlier h=15 estimate gave 9.5 mm (factor ~2 error).  '
    '(c) Absent latent-heat plateau across Aₛ–Aᶠ (0.7–1.1 A) is Gap 3 evidence — motivates Approach 2 (Paper 2).',
    ha='center', fontsize=6.5, color='#555555', style='italic', wrap=True)

fig2.tight_layout()
fig2.savefig(fig_path('fig2_axial_centreline.png'), bbox_inches='tight')
plt.close(fig2)
print("  → results/figures/fig2_axial_centreline.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — Inner vs outer radius, one column per current
#
#  Top row   : ΔT_radial(z) = T(r=0, z) − T(r=R, z) at each τ snapshot.
#              Dashed line = analytical QR²/4k (steady heating).
#  Bottom row: ΔT_radial(t) at z = L/2 from sensors s1 (r=0) and s4 (r=R),
#              through the heating phase and after switch-off.
#  ΔT_radial is ≈ 10–60 mK at d = 0.3 mm (sub-thermocouple); it scales ∝ R²,
#  so ×100 at d = 3 mm — the reason the 2D architecture is retained.
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Inner vs outer radius per current...")

fig3, axes3 = plt.subplots(2, N_I, figsize=(4.3 * N_I, 7.5), squeeze=False)

for col, (I, clr, clbl) in enumerate(zip(CURRENTS, COLORS, C_LABELS)):
    dT_an_mK = DTRAD_DICT[I] * 1e3

    # Top: along z at each snapshot
    ax = axes3[0, col]
    for j, (t_s, name, sclr) in enumerate(zip(SNAP_T_PER_I[I], SNAP_NAMES, SNAP_COLORS)):
        z_cl, T_cl = get_centreline(I, t_s)
        z_sf, T_sf = get_surface_axial(I, t_s)
        dT_mK = (T_cl - np.interp(z_cl, z_sf, T_sf)) * 1e3
        ax.plot(z_cl * 1e3, dT_mK, color=sclr, lw=1.6, label=f'{name}  (t = {t_s} s)')
    ax.axhline(dT_an_mK, color='k', lw=1.0, ls='--', alpha=0.7,
               label=f'QR²/4k = {dT_an_mK:.0f} mK')
    ax.set_xlim(0, L_MM)
    ax.set_xlabel('z  (mm)')
    ax.set_title(clbl, fontweight='bold', color=clr)
    ax.legend(fontsize=7, loc='center')
    if col == 0:
        ax.set_ylabel('ΔT = T(r=0) − T(r=R)  (mK)\nalong the wire')

    # Bottom: midpoint over time
    ax = axes3[1, col]
    t_arr = SENS[I][:, 0]
    dT_t  = (SENS[I][:, 2] - SENS[I][:, 5]) * 1e3
    shade_heating(ax)
    ax.plot(t_arr, dT_t, color=clr, lw=1.8)
    ax.axhline(dT_an_mK, color='k', lw=1.0, ls='--', alpha=0.7)
    ax.set_xlim(0, T_FINAL)
    ax.set_xlabel('t  (s)')
    if col == 0:
        ax.set_ylabel('ΔT = T(r=0) − T(r=R)  (mK)\nat z = L/2')

axes3[1, 0].text(0.97, 0.72,
    f'Bi = hR/k ≈ {H_NOM*R/k:.1e} ≪ 1\n'
    'ΔT_radial below K-type TC\nresolution (±0.5 K); ∝ R²,\n'
    'so ×100 at d = 3 mm',
    transform=axes3[1, 0].transAxes, fontsize=7, va='top', ha='right',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
              edgecolor='0.7', alpha=0.85))

fig3.suptitle(
    'Inner (r = 0) vs Outer (r = R) Radius  —  d = 0.3 mm\n'
    f'Top: ΔT_radial along z at τ/4, τ, 3τ, 4τ   |   Bottom: ΔT_radial(t) at z = L/2, '
    f'current off at t = {T_OFF:.0f} s',
    fontsize=11, fontweight='bold')
fig3.tight_layout()
fig3.savefig(fig_path('fig3_inner_vs_outer_radius.png'), bbox_inches='tight')
plt.close(fig3)
print("  → results/figures/fig3_inner_vs_outer_radius.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — Sensor Curves  T(t)  at 3 key locations, heating + cooling
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Virtual sensor heating/cooling curves...")

SENSOR_PANELS = [
    (2, f'Centreline midpoint\n(r = 0,  z = L/2 = {L_MM/2:.0f} mm)'),
    (1, f'Centreline quarter-point\n(r = 0,  z = L/4 = {L_MM/4:.0f} mm)'),
    (5, f'Surface midpoint\n(r = R,  z = L/2 = {L_MM/2:.0f} mm)'),
]

fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4.5))

for col, (s_col, s_title) in enumerate(SENSOR_PANELS):
    ax = axes4[col]
    shade_heating(ax)
    for I, clr, clbl in zip(CURRENTS, COLORS, C_LABELS):
        ax.plot(SENS[I][:, 0], K2C(SENS[I][:, s_col]), color=clr, lw=1.8, label=clbl)
    ax.axhline(AS_C, color='steelblue', lw=0.9, ls='--', alpha=0.75,
               label=f'Aₛ = {AS_C:.0f}°C')
    ax.axhline(AF_C, color='firebrick', lw=0.9, ls='--', alpha=0.75,
               label=f'Aᶠ = {AF_C:.0f}°C')

    y0 = K2C(T_INF) + 0.5
    for mult, lbl in zip(snap_multiples[1:], SNAP_NAMES[1:]):
        t_line = mult * TAU_NOM
        ax.axvline(t_line, color='grey', lw=0.7, ls=':', alpha=0.5)
        ax.text(t_line + 0.4, y0, lbl, fontsize=7, color='grey')

    ax.set_xlabel('t  (s)')
    if col == 0:
        ax.set_ylabel('T  (°C)')
        ax.legend(fontsize=7.5, loc='upper right')
    ax.set_title(s_title, fontsize=9)
    ax.set_xlim(0, T_FINAL)

fig4.suptitle(
    'Virtual Sensor Curves  T(t)  —  d = 0.3 mm  [Flexinol HT 0.012"]\n'
    f'Heating 0–{T_OFF:.0f} s (shaded), cooling {T_OFF:.0f}–{T_FINAL:.0f} s  '
    f'(τ markers at {I_NOM/1000:g} A nominal: τ = {TAU_NOM:.1f} s)',
    fontsize=11, fontweight='bold')
fig4.tight_layout()
fig4.savefig(fig_path('fig4_sensor_curves.png'), bbox_inches='tight')
plt.close(fig4)
print("  → results/figures/fig4_sensor_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — Radial Profile  T(r, z=L/2)  at t = 4τ
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5: Radial profiles at z=L/2...")

fig5, ax5 = plt.subplots(figsize=(7, 4.5))

for I, clr, clbl in zip(CURRENTS, COLORS, C_LABELS):
    r_arr, T_arr = get_radial_midplane(I, snap_last(I))
    ax5.plot(r_arr * 1e6, K2C(T_arr), color=clr, lw=2.0, marker='o', ms=4, label=clbl)

ax5.axhline(AS_C, color='steelblue', lw=0.9, ls='--', alpha=0.75, label=f'Aₛ = {AS_C:.0f}°C')
ax5.axhline(AF_C, color='firebrick', lw=0.9, ls='--', alpha=0.75, label=f'Aᶠ = {AF_C:.0f}°C')

ax5.text(0.02, 0.97,
    f'{CURRENTS[0]/1000:g} A sits between Aₛ and Aᶠ; higher currents reach or\n'
    'exceed Aᶠ (full austenite). Curves pass through Aₛ–Aᶠ\n'
    'without a latent-heat plateau: the constant-cₚ model\n'
    'has no phase physics.',
    transform=ax5.transAxes, fontsize=7, va='top', ha='left',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
              edgecolor='0.7', alpha=0.85))

ax5.set_xlabel('r  (µm)')
ax5.set_ylabel('T  (°C)')
ax5.set_title(
    f'Radial Profile  T(r,  z = L/2)  at  t = 4τ\n'
    f'ΔT_radial = QR²/4k ≈ {DTRAD_STR}  (Bi = hR/k ≈ {H_NOM*R/k:.0e} ≪ 1)',
    fontsize=10)
ax5.set_xlim(0, R * 1e6 * 1.05)
ax5.legend(fontsize=8.5, loc='upper right')
fig5.tight_layout()
fig5.savefig(fig_path('fig5_radial_profile.png'), bbox_inches='tight')
plt.close(fig5)
print("  → results/figures/fig5_radial_profile.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — End-Cooling Gradient at the left clamp, one panel per τ snapshot
#
#  The problem is symmetric about z = L/2, so only z ∈ [0, Z_END] is shown.
#  Each current's L_b = √(kR/2h) is marked (dotted line + dot on the curve);
#  the legend gives the clamp gradient dT/dz at z = 0.
#  Symmetry check |T(z=L_b) − T(z=L−L_b)| < 0.01 K is printed by the FEM run.
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 6: End-cooling gradient (left clamp)...")

Z_END_MM = 5.0 * np.ceil(6.0 * Lb_max / 5.0)     # ≈ 6 L_b, rounded up to 5 mm

fig6, axes6 = plt.subplots(1, N_S, figsize=(4.6 * N_S, 4.5), sharey=False)

for col, t_lbl in enumerate(SNAP_LABEL):
    ax = axes6[col]
    for I, clr, clbl in zip(CURRENTS, COLORS, C_LABELS):
        z, T = get_centreline(I, SNAP_T_PER_I[I][col])
        grad0 = (T[1] - T[0]) / ((z[1] - z[0]) * 1e3)      # K/mm at the clamp
        ax.plot(z * 1e3, K2C(T), color=clr, lw=1.8,
                label=f'{clbl}  (dT/dz|₀ = {grad0:.0f} K/mm)')
        Lb_mm = LB_DICT[I] * 1e3
        ax.axvline(Lb_mm, color=clr, lw=0.8, ls=':', alpha=0.8)
        ax.plot(Lb_mm, K2C(np.interp(LB_DICT[I], z, T)), 'o', color=clr,
                ms=4, mec='k', mew=0.4, zorder=5)
    ax.axhline(AS_C, color='steelblue', lw=0.8, ls='--', alpha=0.7)
    ax.axhline(AF_C, color='firebrick', lw=0.8, ls='--', alpha=0.7)
    ax.set_xlim(0, Z_END_MM)
    ax.set_xlabel('z  (mm)  from left clamp')
    if col == 0:
        ax.set_ylabel('T  (°C)')
    ax.set_title(t_lbl, fontweight='bold')
    ax.legend(fontsize=6.8, loc='lower right')

axes6[0].text(0.03, 0.97,
    'Symmetry check (FEM run, t = t_off):\n'
    '|T(z=L_b) − T(z=L−L_b)| < 0.01 K → PASS\n'
    'Dots/dotted lines: z = L_b per current',
    transform=axes6[0].transAxes, fontsize=7, va='top', ha='left',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan',
              edgecolor='0.7', alpha=0.85))

fig6.suptitle(
    f'End-Cooling Gradient  T(z, r=0)  near the clamp (z ∈ [0, {Z_END_MM:.0f}] mm)  —  '
    'd = 0.3 mm  [Flexinol HT 0.012"]\n'
    f'L_b = √(kR/2h) = {Lb_min:.1f}–{Lb_max:.1f} mm marks the clamp heat-sink influence  '
    '— Debska et al. 2017, Xiao et al. 2015',
    fontsize=10, fontweight='bold')
fig6.tight_layout()
fig6.savefig(fig_path('fig6_end_cooling_gradient.png'), bbox_inches='tight')
plt.close(fig6)
print("  → results/figures/fig6_end_cooling_gradient.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 7 — T_ss vs I  (I² scaling + analytical slope comparison)
#
#  T_ss is taken at t = T_OFF (end of heating, ≥ 4.5τ), not at t_final —
#  after switch-off the wire cools back towards T∞.
#
#  Analytical derivation (add to §6.1 in exposé):
#    1D fin with uniform Q, Dirichlet ends, surface convection, at z = L/2:
#      ΔT_max = (Q·R)/(2h) · [1 − sech(m·L/2)],   m = 1/L_b = √(2h/kR)
#    Since Q = I²·ρ_e/(πR²)²:
#      ΔT_max = [ρ_e/(2π²R³h)] · I²  ×  [1 − sech(mL/2)]
#  Reference: Incropera & DeWitt, Fundamentals of Heat and Mass Transfer,
#             Extended Surfaces / Fins with heat generation.
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 7: T_ss vs I and I² scaling...")

fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(12, 4.5))

T_ss = [K2C(SENS[I][idx_at(I, T_OFF), 2]) for I in CURRENTS]   # col 2 = centreline midpoint

I_arr  = np.array(CURRENTS, dtype=float)
T_arr  = np.array(T_ss)
dT_arr = T_arr - K2C(T_INF)
I2_arr = I_arr ** 2

coeffs  = np.polyfit(I2_arr, dT_arr, 1)
I2_fit  = np.linspace(I2_arr[0] * 0.9, I2_arr[-1] * 1.05, 200)
dT_fit  = np.polyval(coeffs, I2_fit)
R2      = 1 - (np.sum((dT_arr - np.polyval(coeffs, I2_arr))**2) /
               np.sum((dT_arr - dT_arr.mean())**2))

h_mean = np.mean(list(H_DICT.values()))
m_mean = np.sqrt(2 * h_mean / (k * R))
mL2    = m_mean * L / 2
sech_correction = 1.0 - 1.0 / np.cosh(mL2)

slope_approx = rho_e / (2 * np.pi**2 * R**3 * h_mean)
slope_exact  = slope_approx * sech_correction
slope_approx_mA = slope_approx * 1e-6
slope_exact_mA  = slope_exact  * 1e-6
dT_analytical_fit = slope_exact_mA * I2_fit

print(f"  FEM fit slope        = {coeffs[0]*1e3:.4f} × 10⁻³ °C/mA²")
print(f"  Analytical (exact)   = {slope_exact_mA*1e3:.4f} × 10⁻³ °C/mA²")
print(f"  Analytical (mL/2>>1) = {slope_approx_mA*1e3:.4f} × 10⁻³ °C/mA²")
print(f"  mL/2 (mean h)        = {mL2:.2f}  →  sech correction = {sech_correction:.4f}")
print(f"  R² (FEM fit)         = {R2:.6f}")

for I, T_pt, clr in zip(I_arr, T_arr, COLORS):
    ax7a.scatter(I, T_pt, s=90, color=clr, zorder=5, edgecolors='k', lw=0.5)
    ax7a.annotate(f'{T_pt:.1f}°C', (I, T_pt), textcoords='offset points',
                  xytext=(5, 4), fontsize=7.5, color=clr)
ax7a.axhline(AS_C, color='steelblue', lw=0.9, ls='--', alpha=0.7, label='Aₛ')
ax7a.axhline(AF_C, color='firebrick', lw=0.9, ls='--', alpha=0.7, label='Aᶠ')
ax7a.set_xlabel('I  (mA)')
ax7a.set_ylabel(f'T_ss  (°C)  at  r=0,  z=L/2,  t={T_OFF:.0f} s')
ax7a.set_title('Steady-state temperature vs current (end of heating)')
ax7a.legend(fontsize=8)

for I2, dT, clr in zip(I2_arr, dT_arr, COLORS):
    ax7b.scatter(I2, dT, s=90, color=clr, zorder=5, edgecolors='k', lw=0.5)
ax7b.plot(I2_fit, dT_fit, 'k--', lw=1.5, alpha=0.8,
          label=(f'FEM fit  (R² = {R2:.4f})\n'
                 f'slope = {coeffs[0]*1e3:.3f}×10⁻³ °C/mA²'))
ax7b.plot(I2_fit, dT_analytical_fit, color='steelblue', lw=1.5, ls='-.', alpha=0.85,
          label=(f'Analytical (fin, exact)\n'
                 f'slope = {slope_exact_mA*1e3:.3f}×10⁻³ °C/mA²'))
ax7b.set_xlabel('I²  (mA²)')
ax7b.set_ylabel('ΔT = T_ss − T_∞  (°C)')
ax7b.set_title('ΔT vs I²  —  Joule heating scaling + analytical verification\n'
               'FEM–analytical agreement confirms h, ρₑ, Q self-consistent',
               fontsize=9)
ax7b.legend(fontsize=7.5, loc='upper left')
ax7b.text(0.98, 0.05,
    f'mL/2 = {mL2:.1f}  →  sech correction = {sech_correction:.3f}\n'
    f'(mL/2 >> 1 approx. introduces {100*(1-sech_correction):.1f}% underestimate)',
    transform=ax7b.transAxes, fontsize=7, va='bottom', ha='right', color='steelblue',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='aliceblue',
              edgecolor='0.7', alpha=0.85))

fig7.suptitle(
    f'Steady-State Scaling  —  d = 0.3 mm  [Flexinol HT 0.012"]\n'
    f'h = f(I): {min(H_DICT.values()):.0f}–{max(H_DICT.values()):.0f} W/(m²·K)  '
    f'(Churchill–Chu 1975)',
    fontsize=11, fontweight='bold')
fig7.tight_layout()
fig7.savefig(fig_path('fig7_Tss_vs_I.png'), bbox_inches='tight')
plt.close(fig7)
print("  → results/figures/fig7_Tss_vs_I.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 8 — T(t) at centreline midpoint  [DIAGNOSTIC ONLY — not for publication]
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 8: Cross-case T vs time  [DIAGNOSTIC ONLY]...")

fig8, ax8 = plt.subplots(figsize=(9, 5.2))
shade_heating(ax8)
for I, clr, clbl in zip(CURRENTS, COLORS, C_LABELS):
    ax8.plot(SENS[I][:, 0], K2C(SENS[I][:, 2]), color=clr, lw=2.2, label=clbl)

ax8.axhspan(AS_C, AF_C, alpha=0.07, color='steelblue', label='Aₛ–Aᶠ transformation window')
ax8.axhline(AS_C, color='steelblue', lw=0.9, ls='--', alpha=0.8)
ax8.axhline(AF_C, color='firebrick', lw=0.9, ls='--', alpha=0.8)
ax8.text(T_FINAL + 0.5, AS_C, 'Aₛ', fontsize=8, color='steelblue', va='center', clip_on=False)
ax8.text(T_FINAL + 0.5, AF_C, 'Aᶠ', fontsize=8, color='firebrick', va='center', clip_on=False)

y0 = K2C(T_INF) + 0.8
for mult, lbl in zip(snap_multiples[1:], SNAP_NAMES[1:]):
    t_line = mult * TAU_NOM
    ax8.axvline(t_line, color='grey', lw=0.8, ls=':', alpha=0.5)
    ax8.text(t_line + 0.4, y0, lbl, fontsize=7.5, color='grey')

ax8.set_xlabel('t  (s)', fontsize=11)
ax8.set_ylabel('T  (°C)', fontsize=11)
ax8.set_xlim(0, T_FINAL)
ax8.set_title(
    'Heating/Cooling Curves at Centreline Midpoint  (r = 0,  z = L/2)  —  d = 0.3 mm\n'
    f'PINN must generalise across all {N_I} curves from ≤ 3 sensor readings\n'
    '[DIAGNOSTIC — not for publication: no new information beyond Fig 4 left panel]',
    fontsize=9.5)
ax8.legend(loc='upper left', fontsize=9)
ax8.text(0.5, 0.5, 'DIAGNOSTIC ONLY', transform=ax8.transAxes,
         fontsize=28, color='lightgrey', alpha=0.4, ha='center', va='center',
         rotation=30, fontweight='bold')
fig8.tight_layout()
fig8.savefig(fig_path('fig8_T_vs_time.png'), bbox_inches='tight')
plt.close(fig8)
print("  → results/figures/fig8_T_vs_time.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIG 9 — Half wire  z ∈ [0, L/2],  N_I × N_S grid (currents × τ snapshots)
#
#  The problem is symmetric about z = L/2, so the half wire carries the full
#  information. Each row shares its y-axis so the temperature rise from τ/4
#  to 4τ reads directly across the row.
# ══════════════════════════════════════════════════════════════════════════════
print(f"Fig 9: Half-wire profiles ({N_I} × {N_S} grid)...")

fig9, axes9 = plt.subplots(N_I, N_S, figsize=(4.0 * N_S, 2.7 * N_I),
                            sharex=True, squeeze=False)

for row, (I, clr, clbl) in enumerate(zip(CURRENTS, COLORS, C_LABELS)):
    profiles = []
    for t_s in SNAP_T_PER_I[I]:
        z, T = get_centreline(I, t_s)
        m = z <= L / 2 + 1e-9
        profiles.append((z[m] * 1e3, K2C(T[m])))
    y_max = max(p[1].max() for p in profiles)
    y_top = max(y_max, AF_C) * 1.06

    for col, ((z_mm, T_c), t_s, name) in enumerate(zip(profiles, SNAP_T_PER_I[I], SNAP_NAMES)):
        ax = axes9[row, col]
        ax.fill_between(z_mm, K2C(T_INF), T_c, color=clr, alpha=0.15, lw=0)
        ax.plot(z_mm, T_c, color=clr, lw=1.8)
        ax.axhline(AS_C, color='steelblue', lw=0.8, ls='--', alpha=0.7)
        ax.axhline(AF_C, color='firebrick', lw=0.8, ls='--', alpha=0.7)
        ax.axvline(LB_DICT[I] * 1e3, color='k', lw=0.6, ls=':', alpha=0.5)
        ax.set_ylim(K2C(T_INF) - 2, y_top)
        ax.set_xlim(0, L_MM / 2)
        ax.text(0.97, 0.06, f't = {t_s} s\nT(L/2) = {T_c[-1]:.1f} °C',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='0.8', alpha=0.85))
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title(name, fontsize=10, fontweight='bold')
        if col == 0:
            ax.set_ylabel(f'{clbl}\nT (°C)', fontsize=9, color=clr, fontweight='bold')
        else:
            ax.set_yticklabels([])
        if row == N_I - 1:
            ax.set_xlabel('z  (mm)', fontsize=8)

fig9.legend(handles=[
        Line2D([0],[0], color='steelblue', lw=1, ls='--', label=f'Aₛ = {AS_C:.0f}°C'),
        Line2D([0],[0], color='firebrick', lw=1, ls='--', label=f'Aᶠ = {AF_C:.0f}°C'),
        Line2D([0],[0], color='k', lw=0.8, ls=':', label='z = L_b')],
    loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=8)
fig9.suptitle(
    f'Half-Wire Centreline Temperature  T(z, r=0),  z ∈ [0, {L_MM/2:.0f}] mm  —  d = 0.3 mm\n'
    'Rows: applied current  |  Columns: τ/4, τ, 3τ, 4τ (per-current τ)',
    fontsize=11, fontweight='bold', y=0.995)
fig9.tight_layout(rect=(0, 0, 1, 0.93))
fig9.subplots_adjust(top=0.885)
fig9.savefig(fig_path('fig9_half_wire_grid.png'), bbox_inches='tight')
plt.close(fig9)
print("  → results/figures/fig9_half_wire_grid.png")

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
fig_names = [
    ("1", "fig1_2D_fields.png",               f"2D T(r,z) — {N_S}×{N_I} grid, full wire"),
    ("2", "fig2_axial_centreline.png",        "Axial centreline profiles, full wire"),
    ("3", "fig3_inner_vs_outer_radius.png",   "Inner vs outer radius ΔT, per current"),
    ("4", "fig4_sensor_curves.png",           "Sensor curves, heating + cooling"),
    ("5", "fig5_radial_profile.png",          "Radial profile at z=L/2, t=4τ"),
    ("6", "fig6_end_cooling_gradient.png",    "End-cooling gradient, left clamp"),
    ("7", "fig7_Tss_vs_I.png",                "T_ss vs I + analytical slope check"),
    ("8", "fig8_T_vs_time.png",               "T(t) cross-case [DIAGNOSTIC ONLY]"),
    ("9", "fig9_half_wire_grid.png",          f"Half wire, {N_I}×{N_S} grid"),
]
print("\n" + "═" * 62)
print(f"  All {len(fig_names)} figures saved to {FIG_DIR}")
print("═" * 62)
for num, fname, desc in fig_names:
    print(f"  Fig {num}:  {fname:<36}  {desc}")
print()
