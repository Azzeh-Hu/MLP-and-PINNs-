"""Surface heat transfer coefficient: natural convection + linearised radiation.

h_conv : Churchill & Chu (1975), horizontal cylinder, all Ra
             Nu = {0.60 + 0.387 Ra^(1/6) / [1 + (0.559/Pr)^(9/16)]^(8/27)}^2
h_rad  : secant linearisation of eps*sigma*(Tw^4 - Tinf^4) = h_rad*(Tw - Tinf)
             h_rad = eps*sigma*(Tw^2 + Tinf^2)(Tw + Tinf)
Air properties: Incropera & DeWitt, Table A.4 (1 atm), evaluated at the film temperature.
"""
import numpy as np

SIGMA = 5.670374419e-8                       # Stefan-Boltzmann (W/m2/K4)

_T = np.array([250.0, 300.0, 350.0, 400.0, 450.0])                 # K
_K = np.array([22.3, 26.3, 30.0, 33.8, 37.3]) * 1e-3              # W/m/K
_NU = np.array([11.44, 15.89, 20.92, 26.41, 32.39]) * 1e-6        # m2/s
_AL = np.array([15.9, 22.5, 29.9, 38.3, 47.2]) * 1e-6             # m2/s
_PR = np.array([0.720, 0.707, 0.700, 0.690, 0.686])


def h_conv(dT, d, T_inf):
    """Churchill-Chu natural-convection coefficient (W/m2/K) for wall excess dT."""
    Tf = T_inf + 0.5 * dT
    k, nu, al, pr = (np.interp(Tf, _T, a) for a in (_K, _NU, _AL, _PR))
    Ra = 9.81 * (1.0 / Tf) * abs(dT) * d**3 / (nu * al)
    Nu = (0.60 + 0.387 * Ra ** (1 / 6) / (1 + (0.559 / pr) ** (9 / 16)) ** (8 / 27)) ** 2
    return Nu * k / d


def h_rad(dT, T_inf, eps):
    """Secant radiation coefficient (W/m2/K) for wall excess dT."""
    Tw = T_inf + dT
    return eps * SIGMA * (Tw**2 + T_inf**2) * (Tw + T_inf)


def h_total(dT, d, T_inf, eps):
    return h_conv(dT, d, T_inf) + h_rad(dT, T_inf, eps)
