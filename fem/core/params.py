"""Single source of truth for geometry, material, transformation, test matrix and numerics.

Every level script (L0/L1/L2) reads from here, so the three levels differ only in
the physics they keep, never in their inputs. Values marked [TBD] are placeholders
until the DSC / datasheet of the purchased wire batch is available.
"""
import numpy as np

from .convection import h_total

# ── Geometry ─────────────────────────────────────────────────────────────────
L = 0.500                                   # wire length between clamps (m)
DIAMETERS = {                               # Flexinol HT (Dynalloy)
    "D0305": 0.305e-3,                      # 0.012"
    "D0508": 0.508e-3,                      # 0.020"
}

# ── Material (Dynalloy Flexinol technical data; verify against current sheet) ─
RHO = 6450.0                                # density (kg/m3), both phases
CP = 836.8                                  # specific heat (J/kg/K) = 0.2 cal/(g K)
K = 18.0                                    # conductivity (W/m/K) = 0.18 W/(cm K)
# Phase dependence of k is reported in opposite directions (Faulkner 2000: k_A > k_M;
# Jury 2023: k_SIM > k_A), so k is kept phase-independent and varied in sensitivity runs.
RHO_E = {"M": 76e-8, "A": 82e-8}            # electrical resistivity (Ohm m)
DH = 24.2e3                                 # transformation enthalpy (J/kg) = 5.78 cal/g
EMISSIVITY = 0.3                            # [TBD] surface emissivity

# ── Transformation temperatures (K) [TBD: replace with DSC of the wire batch] ─
AS, AF = 333.15, 383.15                     # austenite start / finish (60 / 110 C)
MS, MF = 343.15, 313.15                     # martensite start / finish (70 / 40 C)

# ── Environment and loading cycle ────────────────────────────────────────────
T_INF = 295.15                              # ambient = clamp = initial temperature (K)
T_OFF = 90.0                                # current off (s): >= 5 tau_th for D0508
T_END = 180.0                               # end of cooling (s)

# ── Numerics (shared by all levels; justified in verification/) ──────────────
DT = 0.1                                    # time step (s) = 10 Hz sensor rate
NR = 6                                      # radial elements (Bi << 1)
MESH = dict(dz_fine=0.1e-3, z_fine=15e-3,   # 0.1 mm over 15 mm from each clamp
            dz_core=0.5e-3, growth=1.1)     # graded to 0.5 mm in the core
TAU_XI = 0.1                                # L2 transformation relaxation time (s)

# ── Outputs ──────────────────────────────────────────────────────────────────
OUT_DT = 1.0                                # field export interval (s)
OUT_DZ = 1.0e-3                             # field export axial spacing (m)
SENSOR_COUNTS = range(3, 9)                 # surface sensors per layout, N = 3..8
SENSOR_LAYOUTS = ("P1_uniform", "P2_clamp", "P3_half")

# ── Test matrix: current levels defined by target steady-state peak temperature ─
TARGETS = {                                 # level: (target peak T (K), regime)
    "C1": (AS - 10.0, "below As (fully martensitic)"),
    "C2": (AS,        "at As (onset)"),
    "C3": (0.5 * (AS + AF), "mid-transformation"),
    "C4": (AF,        "at Af (completing)"),
    "C5": (AF + 30.0, "above Af (front travels to clamps)"),
}


def xi_heating(T, exp=np.exp):
    """Equilibrium austenite fraction on heating; 1 % at As, 99 % at Af."""
    return 1.0 / (1.0 + exp(-2.0 * np.log(99.0) * (T - 0.5 * (AS + AF)) / (AF - AS)))


def xi_cooling(T, exp=np.exp):
    """Equilibrium austenite fraction on cooling; 99 % at Ms, 1 % at Mf."""
    return 1.0 / (1.0 + exp(-2.0 * np.log(99.0) * (T - 0.5 * (MS + MF)) / (MS - MF)))


def sensor_positions(layout, n):
    """Axial positions (m) of n surface sensors.

    P1_uniform : z_i = i L/(n+1)                    -> all sit in the flat mid-section
    P2_clamp   : 1 at mid-span (plateau level); the other n-1 alternate between the two
                 clamps at distances geomspace(2, 25 mm) ~ 5 L_b (5 mm if only one per side),
                 i.e. inside the end-cooling layer where the gradients are
    P3_half    : z_i = i (L/2)/n on one half only   -> symmetry must be inferred
    """
    if layout == "P1_uniform":
        return L * np.arange(1, n + 1) / (n + 1)
    if layout == "P3_half":
        return 0.5 * L * np.arange(1, n + 1) / n
    if layout == "P2_clamp":
        k_left = n // 2
        side = lambda k: np.array([5e-3]) if k == 1 else np.geomspace(2e-3, 25e-3, k)
        left = side(k_left)
        right = L - side(n - 1 - k_left) if n - 1 - k_left > 0 else np.array([])
        return np.sort(np.concatenate([left, [0.5 * L], right]))
    raise ValueError(layout)


def mix(prop, xi):
    """Rule-of-mixtures property between martensite (xi=0) and austenite (xi=1)."""
    return prop["M"] + xi * (prop["A"] - prop["M"])


def current_for_target(R, T_target):
    """Current (A) whose steady mid-span temperature is T_target.

    Far from the clamps the steady energy balance per unit length is
        I^2 rho_e / (pi R^2) = 2 pi R h (T - T_inf)
    =>  I = pi R sqrt(2 R h dT / rho_e),
    with rho_e at the equilibrium phase state of T_target (phase-aware truth).
    """
    dT = T_target - T_INF
    h = h_total(dT, 2.0 * R, T_INF, EMISSIVITY)
    rho_e = mix(RHO_E, xi_heating(T_target))
    return np.pi * R * np.sqrt(2.0 * R * h * dT / rho_e), h


def cases():
    """All (diameter x current level) cases with their derived parameters."""
    out = []
    for dkey, D in DIAMETERS.items():
        R = 0.5 * D
        for ckey, (T_target, regime) in TARGETS.items():
            I, h = current_for_target(R, T_target)
            I = round(I, 3)                                  # 1 mA resolution
            out.append(dict(
                name=f"{dkey}_{ckey}_I{int(round(I * 1e3)):04d}mA",
                D=D, R=R, I=I, h=h, level=ckey, regime=regime,
                T_target=T_target,
                Bi=h * R / K,                                # radial Biot number
                L_b=np.sqrt(K * R / (2.0 * h)),              # end-cooling length (m)
                tau_th=RHO * CP * R / (2.0 * h),             # lumped time constant (s)
            ))
    return out
