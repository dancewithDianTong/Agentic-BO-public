"""A toy continuous-flow Suzuki-Miyaura coupling: the natural-language-prior demo problem.

Six process conditions -> isolated yield (%) and a relative cost index. The yield
model is a smooth, deliberately simple kinetic caricature (not a fitted model of any
real reaction), built from chemically motivated pieces:

* **coupling rate** ``k1``: Arrhenius in temperature, saturating in Pd loading
  (Michaelis-Menten-like), base activation of the boronic acid (sigmoid in pH around
  the boronate pKa), and a biphasic mass-transfer factor in the water fraction that
  peaks for a minority aqueous phase (plus a weaker "on-water" bump for water-rich
  media -- a secondary mode);
* **catalyst deactivation** ``kdeact``: slow ligand oxidation (weakly activated), fast
  ligand thermolysis above ~110 C (strongly activated), Pd aggregation at high loading,
  and slightly faster oxidation at higher pressure (more dissolved oxygen); the
  conversion of a deactivating catalyst is ``1 - exp(-k1/kdeact (1 - exp(-kdeact tau)))``;
* a **ligand-free Pd-nanoparticle pathway** that is only active at very low loading and
  very high temperature ("homeopathic" Pd) -- a second, hot / short-residence local mode;
* **side reactions**: protodeboronation competing with the coupling (faster at high
  pH, high water content and temperature), thermal decomposition of the product during
  its residence time (high temperature x long time), homocoupling at high loading;
* **pressure**: a weak effect near the optimum, but heating far above the solvent's
  boiling point at that pressure (Clausius-Clapeyron, 88 C at 1 bar) loses yield.

The best achievable yield is ~94 % (95 % is the isolated-yield ceiling); ``f_star`` and
``x_star`` are found by a dense multi-start search on first use and cached.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from ..space import RealParam, SearchSpace
from .base import Problem, VectorizedEvaluator

__all__ = ["reaction_yield_metrics", "reaction_yield_optimum", "make_reaction_yield", "REACTION_PARAMS"]

REACTION_PARAMS: tuple[RealParam, ...] = (
    RealParam("temperature", 0.0, 150.0),                       # C
    RealParam("pressure", 1.0, 10.0),                           # bar
    RealParam("catalyst_loading", 0.1, 10.0, log_scale=True),   # mol% Pd
    RealParam("residence_time", 1.0, 180.0, log_scale=True),    # min
    RealParam("pH", 6.0, 13.0),                                 # aqueous phase
    RealParam("water_fraction", 0.0, 1.0),                      # volume fraction of water
)

YIELD_CEILING = 95.0           # % isolated yield at complete, perfectly selective conversion

_R = 8.314462618e-3            # kJ / (mol K)
_T_REF = 353.15                # K (80 C): rate constants below are quoted at this temperature

# coupling (ligated Pd), per minute at 80 C
_K_COUPLING, _EA_COUPLING = 0.40, 70.0
_K_SAT = 0.6                   # mol% loading at half-saturated activity
_PKA, _PKA_SLOPE, _BASE_FLOOR = 9.4, 0.8, 0.02
_W_OPT, _W_WIDTH, _W_FLOOR = 0.32, 0.12, 0.08        # biphasic mass-transfer optimum
_W_ONWATER, _W_ONWATER_WIDTH, _ONWATER_GAIN = 0.85, 0.06, 0.30
# catalyst deactivation
_K_DEACT, _E_DEACT = 0.04, 30.0                      # ligand oxidation
_THERMOLYSIS, _E_THERMOLYSIS = 0.02, 200.0           # ligand thermolysis (relative weight at 80 C)
_L_AGG = 4.0                                         # mol% scale of Pd aggregation
_P_OXYGEN = 0.02                                     # relative deactivation increase per bar
# ligand-free nanoparticle pathway
_K_NP, _EA_NP, _L_NP = 0.002, 155.0, 0.15
_K_NP_DEACT, _E_NP_DEACT = 0.30, 40.0
# side reactions
_K_PDB, _E_PDB, _PH_PDB, _W_PDB = 0.0015, 95.0, 11.0, 0.15   # protodeboronation
_K_DECOMP, _E_DECOMP = 3e-5, 150.0                           # product decomposition
_K_HOMO, _E_HOMO = 0.0015, 40.0                              # homocoupling, per mol%
# solvent boiling point vs pressure
_T_BOIL_1BAR, _DH_VAP, _BOIL_LOSS = 88.0, 38.0, 0.5


def _arrhenius(TK: np.ndarray, ea: float) -> np.ndarray:
    return np.exp(-ea / _R * (1.0 / TK - 1.0 / _T_REF))


def _mean_exposure_fraction(z: np.ndarray) -> np.ndarray:
    """Mean post-formation residence time / tau for first-order formation with ``z = k tau``.

    ``1 / (1 - exp(-z)) - 1 / z``: tends to 1/2 for slow formation and to 1 for fast
    formation; the series branch avoids cancellation for small ``z``.
    """
    z = np.asarray(z, dtype=float)
    small = z < 1e-4
    zz = np.where(small, 1.0, z)
    return np.where(small, 0.5 + z / 12.0, 1.0 / (-np.expm1(-zz)) - 1.0 / zz)


def reaction_yield_metrics(X: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized model: raw ``(n, 6)`` conditions -> ``{"yield": %, "cost": index}``.

    Columns follow ``REACTION_PARAMS``: temperature (C), pressure (bar), catalyst
    loading (mol%), residence time (min), pH, water fraction.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    T, P, L, tau, pH, w = X.T
    TK = T + 273.15

    base = _BASE_FLOOR + (1.0 - _BASE_FLOOR) / (1.0 + 10.0 ** (_PKA_SLOPE * (_PKA - pH)))
    phase = (_W_FLOOR + (1.0 - _W_FLOOR) * np.exp(-0.5 * ((w - _W_OPT) / _W_WIDTH) ** 2)
             + _ONWATER_GAIN * np.exp(-0.5 * ((w - _W_ONWATER) / _W_ONWATER_WIDTH) ** 2))
    oxygen = 1.0 + _P_OXYGEN * (P - 1.0)

    k1 = _K_COUPLING * _arrhenius(TK, _EA_COUPLING) * L / (L + _K_SAT) * base * phase
    kdeact1 = (_K_DEACT * (_arrhenius(TK, _E_DEACT) + _THERMOLYSIS * _arrhenius(TK, _E_THERMOLYSIS))
               * (1.0 + (L / _L_AGG) ** 2) * oxygen)
    np_activity = (L / _L_NP) * np.exp(1.0 - L / _L_NP)
    k2 = _K_NP * _arrhenius(TK, _EA_NP) * np_activity * base * phase
    kdeact2 = _K_NP_DEACT * _arrhenius(TK, _E_NP_DEACT) * oxygen

    # integrated rate of two independently deactivating catalysts; conversion = 1 - exp(-z)
    z = k1 / kdeact1 * -np.expm1(-kdeact1 * tau) + k2 / kdeact2 * -np.expm1(-kdeact2 * tau)
    conversion = -np.expm1(-z)

    k_pdb = (_K_PDB * _arrhenius(TK, _E_PDB) * 10.0 ** (pH - _PH_PDB) * (_W_PDB + w) ** 2)
    s_pdb = (k1 + k2) / (k1 + k2 + k_pdb)
    s_decomp = np.exp(-_K_DECOMP * _arrhenius(TK, _E_DECOMP) * tau * _mean_exposure_fraction(z))
    s_homo = 1.0 / (1.0 + _K_HOMO * L * _arrhenius(TK, _E_HOMO))
    t_boil = 1.0 / (1.0 / (273.15 + _T_BOIL_1BAR) - _R * np.log(P) / _DH_VAP) - 273.15
    s_boil = 1.0 - _BOIL_LOSS / (1.0 + np.exp(-(T - t_boil - 5.0) / 4.0))

    yld = YIELD_CEILING * conversion * s_pdb * s_decomp * s_homo * s_boil
    # cost index: Pd (2 per mol%), heating (~ (T + 10) x time), pressurization, fixed materials
    cost = 1.0 + 2.0 * L + 6e-4 * (T + 10.0) * tau + 0.25 * (P - 1.0)
    return {"yield": yld, "cost": cost}


def _unit_to_raw(U: np.ndarray) -> np.ndarray:
    U = np.clip(np.atleast_2d(U), 0.0, 1.0)
    lo = np.array([p.lower for p in REACTION_PARAMS])
    hi = np.array([p.upper for p in REACTION_PARAMS])
    log = np.array([p.log_scale for p in REACTION_PARAMS])
    lin = lo + U * (hi - lo)
    lg = np.exp(np.log(np.where(log, lo, 1.0)) + U * (np.log(np.where(log, hi, 1.0)) - np.log(np.where(log, lo, 1.0))))
    return np.where(log, lg, lin)


@functools.lru_cache(maxsize=1)
def reaction_yield_optimum() -> tuple[float, dict[str, float]]:
    """``(f_star, x_star)`` of the yield by a dense multi-start search (cached; ~0.3 s).

    A 4096-point scrambled Sobol design (fixed seed) seeds L-BFGS-B runs from the best
    distinct designs, in the encoded unit cube (log axes for loading and time).
    """
    def neg(u: np.ndarray) -> float:
        return -float(reaction_yield_metrics(_unit_to_raw(u))["yield"][0])

    U = qmc.Sobol(d=len(REACTION_PARAMS), scramble=True, seed=20240816).random(4096)
    y = reaction_yield_metrics(_unit_to_raw(U))["yield"]
    starts: list[np.ndarray] = []
    for i in np.argsort(-y):
        if all(np.max(np.abs(U[i] - s)) > 0.15 for s in starts):
            starts.append(U[i])
        if len(starts) == 8:
            break
    best_v, best_u = -math.inf, starts[0]
    for u0 in starts:
        res = minimize(neg, u0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * len(REACTION_PARAMS),
                       options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 2000})
        if -res.fun > best_v:
            best_v, best_u = -float(res.fun), np.clip(res.x, 0.0, 1.0)
    x = _unit_to_raw(best_u)[0]
    x_star = {p.name: float(v) for p, v in zip(REACTION_PARAMS, x)}
    f_star = float(reaction_yield_metrics(x[None])["yield"][0])
    return f_star, x_star


DESCRIPTION = (
    "Optimize a continuous-flow Suzuki-Miyaura cross-coupling of an aryl bromide with an "
    "arylboronic acid (phosphine-ligated Pd catalyst, aqueous base / organic co-solvent). "
    "Each experiment sets six process conditions -- temperature (°C), back-pressure (bar), Pd "
    "loading (mol%), residence time (min), pH of the aqueous phase, and the water volume fraction "
    "of the solvent -- and reports the isolated yield of the biaryl product (%) and a relative cost "
    "index for the run. "
    "Goal: maximize the yield."
)

PRIOR = (
    "Practitioner experience with aqueous-organic Suzuki-Miyaura couplings of this kind: the ligated "
    "Pd catalyst usually performs best at moderately elevated temperature, roughly 70-100 °C; near room "
    "temperature the reaction is sluggish and the catalyst tends to die before conversion is complete, "
    "while above about 110 °C the phosphine ligand degrades quickly (palladium black) and the biaryl "
    "product starts to decompose during long holds. "
    "The aqueous phase must be clearly basic to activate the boronic acid (pH of roughly 10-12 is "
    "typical), but strongly caustic conditions accelerate protodeboronation. "
    "Biphasic mixtures with a minority water phase (about 20-40 % water) generally outperform both "
    "anhydrous and water-rich media. "
    "Pd loadings of about 0.5-3 mol% are standard: more catalyst brings diminishing returns, and very "
    "high loadings promote aggregation and homocoupling. "
    "At these temperatures residence times of roughly 20-90 minutes are normally sufficient, and a "
    "modest back-pressure of a few bar mainly serves to keep the solvent from boiling."
)

PRIOR_HINT = {"region": {
    "temperature": [70.0, 100.0],
    "pressure": [1.5, 6.0],
    "catalyst_loading": [0.5, 3.0],
    "residence_time": [20.0, 90.0],
    "pH": [10.0, 12.0],
    "water_fraction": [0.2, 0.4],
}}


def make_reaction_yield() -> Problem:
    """The ``reaction_yield`` problem (computes and caches its optimum on first call)."""
    f_star, x_star = reaction_yield_optimum()
    names = [p.name for p in REACTION_PARAMS]
    ev = VectorizedEvaluator(names, reaction_yield_metrics)
    return Problem(
        name="reaction_yield", space=SearchSpace(list(REACTION_PARAMS)), evaluate_fn=ev,
        objective="yield", minimize=False, f_star=f_star, description=DESCRIPTION,
        prior=PRIOR, prior_hint={"region": {k: list(v) for k, v in PRIOR_HINT["region"].items()}},
        default_budget=40, x_star=dict(x_star),
        outputs={"yield": "isolated yield of the biaryl product in %",
                 "cost": "relative cost index of the run (Pd, heating time, pressurization); lower is cheaper"},
        batch_fn=ev.batch, shift_safe=False,
        meta={"family": "reaction_yield", "dim": 6, "randomized": False},
    )
