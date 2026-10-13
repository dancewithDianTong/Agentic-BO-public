# Agentic BO — design specification

Implementation of *agentic Bayesian optimization* following
**"Agentic Bayesian Optimization through Surrogate-Augmented Autoresearch"**
(Brunzema, Tiao, Le, De Angeli, Xuan, Gligorijevic — arXiv:2608.00316).

The paper's system has two halves, and so does this package:

| Paper | Here | Role |
|---|---|---|
| **lenz** (BoTorch backend, JSON CLI) | `agentic_bo.lenz` | Bayesian machinery: GP surrogates, acquisition functions, a trial log, and a command interface the agent can inspect and reconfigure. Pure numpy/scipy (no torch). |
| **Sara** (LLM agent, Claude Opus, "pi" harness) | `agentic_bo.sara` | The metalevel policy: decides which probes / reconfigurations / proposals to run and which point to commit for evaluation. Claude through the Anthropic API, plus a deterministic scripted agent and the classical baselines. |

This document is the contract between modules. **Signatures, names, and return
shapes written here are binding** — implementers must not rename them. Anything
not specified is the implementer's choice.

---

## 1. Problem setting and the formal model (paper §2, §4)

Sequentially optimize a black-box `f : X -> R`. At iteration `t` the agent commits
`x_t`, observes `y_t = f(x_t) + eps`, total budget `T`. Performance:

* simple regret `r_T = f* - max_{t<=T} f(x_t)` (paper & user: `min_t f* - f(x_t)`)
* cumulative regret `R_T = sum_{t<=T} (f* - f(x_t))`

(maximization convention; minimization problems use `f = -g`). Regret is always
computed from **noise-free** values of the evaluated points.

The agentic-BO state is `s_t^(j) = (D_t, c_t^(j), K_t, H_t^(j))`:

| Symbol | Meaning | Code |
|---|---|---|
| `D_t` | trial log (single source of truth) | `Lenz.frame.trials` |
| `c_t = (M, alpha, B, O, C)` | configuration: model class, acquisition, active region, objectives, constraints | `Lenz.frame.shelf` |
| `K_t` | exogenous context, append-only (problem brief, instructions arriving mid-run) | `AgentSession.context` |
| `H_t` | deliberation history (tool calls, results, reasoning) | `AgentSession.log` (+ LLM transcript) |

Action space `A = A_comp ∪ A_eval ∪ {stop}`:

* **probe** (read-only): `status`, `trials`, `incumbent`, `pareto`, `diagnostics`, `predict`, `score`
* **reconfigure** (mutates `c_t`, never `D_t`): `set_bounds`, `set_acqf`, `set_objectives`, `set_constraints`
* **propose**: `suggest` (acquisition optimization; records nothing, costs no budget)
* **evaluate**: commit one config to the oracle — the only action that advances `t` and spends budget
* **stop**: end early (only if the campaign allows it)
* (**note**: `record_note` — lab-notebook memory; free, does not touch lenz)

A *campaign step* is `Rollout(A, s_t^(0))`: the policy takes `k_t >= 0` computational
actions, then exactly one evaluate (or stop). Special cases recovered by the code:
no backend calls = LLM-as-optimizer (`toolset="llm_only"`); fixed config + accept the
top proposal every step = standard BO `Pi_0` (`FixedBOPolicy`).

---

## 2. Package layout and ownership

```
Agentic-BO/
  pyproject.toml
  README.md
  docs/DESIGN.md                    (this file)
  agentic_bo/
    __init__.py                     public API (finalized at integration)
    utils.py                        jsonable(), rng helpers, LenzError        [foundation, done]
    space.py                        SearchSpace + param types                 [foundation, done]
    metrics.py                      regret curves, run summaries
    api.py                          optimize(f, space, budget, ...) convenience entry point
    experiment.py                   multi-seed benchmark runner, plots, stats, CLI
    problems/
      __init__.py                   get_problem(), list_problems()
      base.py                       Problem dataclass, randomization wrapper
      synthetic.py                  classic test functions
      applied.py                    reaction-yield toy problem (natural-language prior demo)
    lenz/
      __init__.py                   exports Lenz, LenzError
      gp.py                         exact GP, Matern-5/2 / RBF ARD, MAP fit, LOO, fantasize
      acquisition.py                acquisition functions (+ analytic gradients)
      pareto.py                     pareto_mask, hypervolume (2-D exact, MC otherwise)
      optimize.py                   acquisition optimizer (Sobol raw samples + L-BFGS-B), Thompson sampling
      frame.py                      Trial / Event / Shelf / Frame data model + (de)serialization
      engine.py                     Lenz: all commands, model cache, envelopes, save/load
      cli.py, __main__.py           `python -m agentic_bo.lenz --study s.json <command> ...`
    sara/
      __init__.py
      schema.py                     minimal JSON-schema validator for tool inputs
      tools.py                      ToolSpec registry (agent-facing tools -> lenz / harness)
      session.py                    AgentSession, CampaignConfig, Campaign, CampaignResult
      policies/
        __init__.py                 get_policy(name, **kw)
        base.py                     Policy ABC
        baselines.py                SobolPolicy, RandomPolicy, FixedBOPolicy (Pi_0, Ax-like)
        heuristic.py                HeuristicSaraPolicy (scripted, offline agentic policy)
        llm.py                      ClaudeSaraPolicy (Claude via Anthropic API)
      prompts/SYSTEM.md             Sara system prompt
      prompts/LENZ_REF.md           lenz reference sheet given to the agent
      prompts/SYSTEM_LLM_ONLY.md    prompt for the LLM-as-optimizer ablation
      prompts/COMPACTION.md         instructions for server-side compaction summaries
  examples/ ...
  tests/test_<module>.py
```

Rules for implementers: only edit the files you own; if another module is wrong or
missing something, report it in your final output (don't patch it). Python >= 3.10,
dependencies **numpy, scipy** only (matplotlib optional for plots, `anthropic` optional
for the LLM policy). Run tests with `/opt/miniconda3/bin/python3 -m pytest tests/<file> -q`.

---

## 3. Global conventions

* **Maximization internally.** Every objective is maximized inside lenz; a metric with
  `minimize=True` is negated before fitting. Everything returned to callers/agents is
  in **raw metric units** with the raw direction.
* **Encoded space.** `SearchSpace.encode(config)` maps into `[0,1]^D`. GPs and
  optimizers live in encoded space. See `space.py` (already implemented).
* **Randomness.** Functions take `rng: np.random.Generator` (or a seed). Lenz derives
  per-call generators from `(seed, counter)` so a saved+loaded study reproduces the
  same suggestions.
* **JSON.** Every command result must pass through `utils.jsonable()` (numpy -> python,
  non-finite floats -> `None`) and be serializable with `json.dumps(sort_keys=True)`.
* **Errors.** Invalid user/agent input raises `utils.LenzError(message)`; `Lenz.call`
  turns it into an error envelope. Bugs (TypeError, etc.) are not caught silently in
  library calls, but `AgentSession.call` converts *every* exception into an error
  envelope so an agent can never crash the campaign.

---

## 4. `agentic_bo.space` (implemented)

Param types (frozen dataclasses): `RealParam(name, lower, upper, log_scale=False)`,
`IntParam(name, lower, upper, log_scale=False)`, `OrdinalParam(name, values)`,
`CategoricalParam(name, choices)`.

```python
space = SearchSpace.from_spec([{"name": "x0", "type": "real", "lower": -5, "upper": 10},
                               {"name": "k", "type": "int", "lower": 1, "upper": 8},
                               {"name": "opt", "type": "categorical", "choices": ["adam", "sgd"]}])
space.d, space.D                  # number of params, encoded dim (one-hot for categoricals)
space.encode(cfg) -> (D,)         space.encode_many(cfgs) -> (n, D)
space.decode(z) -> dict           space.decode_many(Z) -> list[dict]     (clip + round + argmax)
space.project(Z) -> Z'            # encode(decode(Z)) — snap to representable points
space.validate(cfg) -> dict       # normalized copy; raises LenzError on missing/extra/out-of-domain
space.validate_bounds(b) -> dict  # normalized active-region spec (see below)
space.encoded_bounds(b=None) -> (lb, ub)   # region in encoded coords
space.sobol(n, rng, bounds=None, skip=0) -> list[dict]
space.sample_uniform(n, rng, bounds=None) -> list[dict]
space.distance(a, b) -> float     # RMS distance in encoded space, in [0, 1]
space.center(bounds=None) -> dict
space.to_spec() / SearchSpace.from_spec(spec)
space.describe() -> str           # markdown table for agent prompts
```

**Active-region spec** (`bounds`): `{param_name: [lo, hi]}` for real/int (raw units,
inside the domain, `lo <= hi`), `{param_name: [allowed values...]}` for
ordinal/categorical (non-empty subset). Params not mentioned keep their full domain.
`None` means the full domain.

---

## 5. `agentic_bo.lenz.gp`

```python
class GP:
    def __init__(self, kernel: str = "matern52",          # or "rbf"
                 noise: float | str = "learn",            # fixed noise variance (standardized units) or "learn"
                 n_restarts: int = 3,
                 seed: int = 0): ...
    def fit(self, X: np.ndarray, y: np.ndarray, warm_start: dict | None = None) -> "GP"
    def posterior(self, Xs, full_cov=False, return_grad=False, observation_noise=False)
        # -> (mean (m,), var (m,))                         if not full_cov and not return_grad
        # -> (mean (m,), cov (m,m))                        if full_cov
        # -> (mean, var, dmean (m,D), dvar (m,D))          if return_grad (full_cov must be False)
        # All in ORIGINAL y units (un-standardized). var = latent-f variance (+ noise if observation_noise).
    def sample(self, Xs, n_samples: int, rng) -> np.ndarray      # (n_samples, m) joint latent samples
    def loo(self) -> tuple[np.ndarray, np.ndarray]                # closed-form LOO mean/var (orig units, var incl. noise)
    def cv_r2(self) -> float | None                               # 1 - SSE_loo / SST; None if n < 3 or SST == 0
    def log_marginal_likelihood(self) -> float
    def fantasize(self, X_new, y_new) -> "GP"                     # same hyperparameters, extra data, no refit
    def hyperparameters(self) -> dict                             # {"lengthscales": [...], "outputscale": f, "noise": f (standardized), "noise_var": f (orig units)}
    n_train: int; X_train; y_train
```

Model details (BoTorch-style defaults, paper §5.1 "input normalization and output
standardization"):

* Inputs are already in `[0,1]^D`. Outputs standardized: `(y - mean) / std`; `std` falls
  back to 1 when the targets are constant relative to their magnitude
  (`std <= 1e-12 * max|y|`) or `n == 1` (no absolute floor, so tiny-scale metrics keep
  their resolution). Zero constant mean in standardized space. The acquisition
  functions' variance floors and the standardization of constrained `ucb`/`pm`/`std`
  use the same relative rule.
* Kernel: `outputscale * k(r)`, ARD lengthscales; Matern-5/2 default, RBF option.
* MAP fit of `theta = log([lengthscales (D), outputscale, noise])` with priors:
  lengthscale `LogNormal(mu = sqrt(2) + 0.5*ln(D), sigma = sqrt(3))` (dimension-scaled,
  Hvarfner et al. 2024), outputscale `LogNormal(0, 1)`, noise variance
  `LogNormal(-4, 1)` clamped to `[1e-6, 1.0]`. The MAP is taken in `theta`. Two prior
  forms (`GP(prior=...)`, lenz `model={"prior": ...}`): **`"botorch"` (default)** evaluates
  the lengthscale and noise LogNormal densities on the parameters themselves, as
  GPyTorch/BoTorch do, i.e. `Normal(mu - sigma^2, sigma)` on the log parameters (lengthscale
  mode `exp(mu - 3)`); `"median"` uses `Normal(mu, sigma)` on the log parameters (the
  LogNormal including the Jacobian; mode `exp(mu)`). The outputscale prior is log-space in
  both. Default chosen by a 288-run A/B (branin, hartmann6, ackley10, levy10, rosenbrock5,
  styblinski_tang8 x fixed_bo/heuristic x 8 seeds): `"botorch"` was better or tied on every
  problem (median final simple regret), significantly better on ackley10 for both policies
  (Mann-Whitney p = 0.003 / 0.007) and on levy10 for the heuristic (p = 0.015), with no
  significant loss. Box bounds on log-lengthscales
  `[ln 1e-2, ln 1e3]`, log-outputscale `[ln 1e-2, ln 1e2]`.
* Optimizer: `scipy.optimize.minimize(method="L-BFGS-B")` with **analytic gradients**
  of the negative log posterior; restarts from `warm_start` (if given), the prior
  median, and random prior draws; keep the best.
* Numerics: Cholesky with jitter escalation (1e-8 ... 1e-4). `n == 1`: no optimization,
  use prior medians. `n == 0`: raise `LenzError`.
* `posterior(..., return_grad=True)` gradients are analytic (needed by acquisition
  optimization). Matern-5/2: `dk/dx_i = -s (5/3)(1 + sqrt5 r) exp(-sqrt5 r) (x_i - x'_i)/l_i^2`.
* Sensitivity (used by lenz diagnostics) = normalized inverse lengthscales.

Tests must include finite-difference checks of the NLL gradient and of posterior
mean/var gradients, LOO vs explicit leave-one-out (same hyperparameters), and
interpolation of a smooth function.

---

## 6. `agentic_bo.lenz.acquisition`, `pareto`, `optimize`

### 6.1 Acquisition functions

All acquisition objects are built from **sign-adjusted** (maximize) GP models and
evaluated on encoded points:

```python
class Acquisition:
    name: str
    has_gradient: bool
    def __call__(self, Z: np.ndarray) -> np.ndarray                  # (m,)
    def value_and_grad(self, Z) -> tuple[np.ndarray, np.ndarray]     # (m,), (m, D); only if has_gradient

def build_acquisition(name: str,
                      objective_models: list,               # GP-like objects (maximize convention)
                      constraint_models: list | None = None,# list of (model, lower, upper) on raw metric scale
                      best_f: float | None = None,          # best observed feasible objective (maximize convention)
                      X_observed: np.ndarray | None = None, # encoded observed points (noisy_logei incumbent)
                      params: dict | None = None,           # e.g. {"beta": 4.0}, {"ref_point": [...]}, {"weights": [...]}
                      Y_observed: np.ndarray | None = None, # (n, m) observed objectives (maximize conv.) for MOO
                      rng: np.random.Generator | None = None) -> Acquisition

ACQF_NAMES  # tuple of supported names
```

Single objective:

| name | definition | notes |
|---|---|---|
| `logei` (default) | `log EI(x; best_f)` via the numerically stable `log_h` of Ament et al. 2023 (`erfcx`, `log_ndtr`) | finite for `z = -1e3` |
| `noisy_logei` | `logei` with incumbent = max posterior mean over observed points | approximation of NEI |
| `ei` | expected improvement | |
| `logpi` / `pi` | (log) probability of improvement | |
| `ucb` | `mu + sqrt(beta) * sigma`, default `beta = 4.0` | |
| `pm` | posterior mean (pure exploitation) | |
| `std` | posterior std (pure exploration / max variance) | |
| `ts` | Thompson sampling | special-cased by the optimizer (joint sample on a candidate set); not available in `score` |
| `sobol`, `random` | space filling, no model | special-cased by lenz |

Constraints (paper §5.1): metric `c` feasible iff `lower <= c <= upper` (either may be
None). `PoF(x) = prod_j P(lower_j <= c_j(x) <= upper_j)` from independent GPs.
For the EI family: `log value = logEI + sum_j log PoF_j` once a feasible point
exists; before any feasible observation the acquisition is `sum_j log PoF_j`
(feasibility phase). For `ucb`/`pm`/`std`: value in standardized objective units
plus `sum_j log PoF_j`. Gradients must remain correct in all cases.

Multi-objective (|O| >= 2):

| name | definition |
|---|---|
| `ehvi` | Monte-Carlo expected hypervolume improvement (fixed base samples, N = 128), m = 2 only (exact 2-D HVI per sample); derivative-free. While no point dominates the ref point (hypervolume 0) lenz uses the exact closed form `prod_i EI_i(x; r_i)` instead (log-space, analytic gradients; §7.2) |
| `parego` | augmented Chebyshev scalarization with weights `params["weights"]` (default: Dirichlet(1) from rng) of min-max-normalized observed objectives; logEI on a GP fitted to the scalarized values (lenz fits that GP and passes it as the single objective model) |

### 6.2 `pareto.py`

```python
pareto_mask(Y: np.ndarray) -> np.ndarray[bool]          # maximize convention, (n, m)
hypervolume(Y, ref_point) -> float                       # exact for m = 2 (and m = 3 slicing); MC (seeded) for m > 3
hvi_2d(samples (N, 2), front (k, 2), ref (2,)) -> (N,)   # exact HV improvement per sample point
default_ref_point(Y) -> np.ndarray                       # min(Y) - 0.1 * (max(Y) - min(Y)), maximize conv.
```

### 6.3 `optimize.py`

```python
def optimize_acqf(acq, space, lb, ub, rng, *, raw_samples=1024, num_restarts=10,
                  X_baseline=None, around_best_frac=0.2, maxiter=200,
                  exclude=None, exclude_tol=1e-6) -> tuple[np.ndarray, float]
    # returns (z_best (D,), value) — z_best is PROJECTED (space.project) and lies in [lb, ub].
    # raw candidates: scrambled Sobol in [lb, ub] plus Gaussian perturbations (sd 0.05*(ub-lb))
    # of the best X_baseline points; top-`num_restarts` start L-BFGS-B (analytic gradients
    # when acq.has_gradient, else finite differences); project; re-score; return the best
    # point that is farther than exclude_tol from every row of `exclude` (if possible).
    # L-BFGS-B runs on (acq - center) / scale, center = best starting value, scale = spread
    # of the starting values (then of all raw values, then their magnitude), so the result
    # does not depend on the acquisition's units; returned values are in acquisition units.

def thompson_sample(model, space, lb, ub, rng, *, n_candidates=1024, X_baseline=None,
                    exclude=None) -> tuple[np.ndarray, float]

def trust_region(center_z, radius, lb, ub, lengthscales=None) -> tuple[np.ndarray, np.ndarray]
    # half-width_i = radius * w_i, w_i = l_i / geomean(l) clipped to [0.25, 4] (TuRBO-style)
    # when lengthscales are given, else radius. Intersected with [lb, ub].
```

---

## 7. `agentic_bo.lenz` engine (paper §5.1, App. G/H)

### 7.1 Data model (`frame.py`)

```python
@dataclass class Trial:
    trial_id: int; config: dict; status: str            # "pending" | "completed" | "abandoned"
    metrics: dict | None; source: str                  # "agent", "suggest:logei", "sobol", "harness-fallback", ...
    created_seq: int; completed_seq: int | None; eval_index: int | None   # 1-based among completed
    meta: dict                                          # e.g. {"rationale": ...}
@dataclass class Event: seq: int; command: str; args: dict; ok: bool; summary: str
@dataclass class Shelf:                                 # c_t
    objectives: list[dict]   # [{"metric": str, "minimize": bool, "ref": float | None}]
    constraints: list[dict]  # [{"metric": str, "lower": float | None, "upper": float | None}]
    acqf: str; acqf_params: dict; bounds: dict | None   # active region (validated spec) or None = full
    model: dict              # {"kernel": "matern52", "noise": "learn", "n_restarts": 3, "prior": "botorch"}
@dataclass class Frame: space: SearchSpace; shelf: Shelf; trials: list[Trial]; events: list[Event]; seed: int; counters: dict
    # to_dict() / from_dict() deterministic
```

### 7.2 `Lenz` API (`engine.py`)

```python
class Lenz:
    def __init__(self, space, objectives, constraints=None, acqf=None, acqf_params=None,
                 bounds=None, model=None, seed=0): ...
        # objectives: list of {"metric", "minimize"} or a single dict or a metric name (=> maximize? NO:
        # a bare string means {"metric": name, "minimize": False}). acqf default: "logei" (1 objective) / "ehvi" (2) / "parego" (>2)
    # ---- commands (return plain dicts; raise LenzError on bad input) ----
    def create(...)  # classmethod alias of __init__ for CLI symmetry
    def suggest(self, q=1, around=None, radius=0.1, bounds=None, acqf=None, acqf_params=None) -> dict
    def submit(self, config, source="agent", meta=None) -> dict
    def observe(self, metrics, trial_id=None, config=None) -> dict
    def abandon(self, trial_id) -> dict
    def set_bounds(self, bounds) -> dict            # None or "full" resets to the full domain
    def set_acqf(self, acqf, beta=None, **params) -> dict
    def set_objectives(self, objectives) -> dict
    def set_constraints(self, constraints) -> dict
    def diagnostics(self, metric=None) -> dict
    def predict(self, configs) -> dict
    def score(self, configs, acqfs=None) -> dict
    def incumbent(self) -> dict
    def pareto(self) -> dict
    def trials(self, status=None, last=None, sort_by=None, limit=None) -> dict
    def status(self) -> dict
    # ---- infrastructure ----
    def call(self, command: str, args: dict | None = None) -> dict    # {"ok", "command", "result"} | {"ok": False, "command", "error": {"type", "message"}}
    def to_dict(self) -> dict; @classmethod from_dict(d) -> Lenz
    def save(self, path); @classmethod load(path) -> Lenz
    @property frame; n_observed; n_pending
```

Command semantics (results are the `result` field of the envelope):

* **suggest** — candidates from acquisition optimization over the active region
  (`bounds` argument overrides the region for this call only; `around` = `"incumbent"` or
  a config dict defines a trust region of normalized half-width `radius` (lengthscale
  scaled) intersected with the region). `q > 1`: sequential greedy with kriging-believer
  fantasies (posterior mean). Pending trials are always fantasized. With fewer than 2
  completed observations, or `acqf in {"sobol", "random"}`, returns space-filling points
  (continuing one scrambled Sobol sequence across calls; skipping points within 1e-6 of
  existing trials). **Records nothing** (paper: trial_id is null). Result:
  `{"candidates": [{"config", "acqf", "acquisition_values": {acqf: v}, "predicted": {metric: {"mean", "std"}}, "prob_feasible", "trial_id": None, "duplicate_of"}], "mode": "model" | "space_filling", "acqf", "acqf_params", "region": {"bounds", "around", "radius", "box"}, "n_observed", "n_pending", "feasible_found", "feasibility_phase", "notes"}`.
  Details:
  - *Trust boxes on lattices*: after `trust_region`, every int (log scale included) and
    ordinal dimension is widened to reach the nearest lattice value on each side of
    the centre, inside the active/override region and its allowed values, so a radius
    narrower than the lattice spacing never pins the dimension to the centre value.
    `region.box` reports the widened box.
  - *Feasibility phase*: with constraints and no feasible completed observation, EI-family
    and MOO acquisitions maximize `sum_j log PoF_j`. A believed point (pending trial or
    an earlier pick of the batch) whose constraint posterior means are feasible ends the
    phase for later picks and counts as an incumbent (`best_f` / Pareto front), as a
    feasible observation would, so batches do not collapse onto the first pick.
    `feasibility_phase` is true if any pick of the batch was in the phase; a note says
    when later picks used believed-feasible earlier picks as incumbents.
  - *EHVI with an empty front*: while no feasible completed (or believed) point dominates
    the reference point (hypervolume 0), `HVI(y) = prod_i (y_i - r_i)^+`, so with
    independent posteriors `EHVI = prod_i EI_i(x; r_i)` exactly. lenz optimizes its log,
    `sum_i logEI_i(x; r_i)` (+ log PoF once), with analytic gradients and reports
    `exp(log)` (the MC EHVI's scale) in suggest/score/predict; suggest adds a note naming
    the ref point. The MC EHVI is used again once any point dominates the ref.
  - The ehvi reference point: objective `ref` > `acqf_params["ref_point"]` > data default.
* **submit** — validates config, records a pending trial. `{"trial_id", "status": "pending", "config"}`.
* **observe** — attaches metrics (dict metric -> float; must include every objective
  and constraint metric currently configured; extra metrics allowed) to a pending trial
  (by `trial_id`, or by matching `config` among pending trials; if `config` matches no
  pending trial a new completed trial is created). `{"trial_id", "status": "completed", "metrics", "eval_index", "is_new_incumbent", "incumbent"}`.
* **set_bounds / set_acqf / set_objectives / set_constraints** — mutate the shelf only;
  never drop data. Results as paper App. H: set_bounds `{"bounds", "is_moo", "observed", "n_in_region"}`;
  set_acqf `{"acqf", "acqf_params", "is_moo", "observed", "notes"}` (`notes` says, e.g.,
  when an ehvi `ref_point` is ignored because objectives carry their own `ref`; suggest
  with `acqf_params={"ref_point": ...}` adds the same note); set_objectives
  `{"objectives": [{"metric", "minimize", "ref"}], "is_moo", "acqf", "acqf_params", "acqf_switched", "previous_acqf", "notes", "observed", "n_with_objectives"}`
  (acqf auto-switches between single/multi-objective families when needed and says so;
  when the ordered (metric, minimize) list changes, an ehvi `ref_point` is dropped with a
  note, ParEGO `weights` are kept since they act on normalized objectives); set_constraints
  `{"constraints": [{"metric", "lower", "upper"}], "feasible_found", "n_feasible", "observed", "is_moo"}`.
* **diagnostics** — per modelled metric (objectives + constraints):
  `{"models": {metric: {"n_observed", "cv_r2", "noise_std", "outputscale", "lengthscales": {param: l}, "sensitivity": {param: s}, "log_marginal_likelihood"}}, "objective": first objective metric}`.
  One-hot dims are aggregated per categorical param (min lengthscale, summed sensitivity).
* **predict** — per config: `{"config", "metrics": {metric: {"mean", "variance", "std"}}, "prob_feasible", "acquisition_values": {current_acqf: v}}` (raw units). Configs may lie outside the active region but must be in the domain.
* **score** — per config acquisition values for each requested acqf (default: current).
* **incumbent** — best completed feasible trial (single objective, raw direction):
  `{"incumbent": {"trial_id", "config", "metrics", "value"} | None, "best_posterior": {"trial_id", "config", "mean"} | None, "feasible_found": bool}`.
  For MOO returns the Pareto summary instead.
* **pareto** — `{"pareto": [trial dicts], "hypervolume", "ref_point"}` (feasible, completed).
* **trials** — `{"trials": [trial dicts], "n": int, "n_total": int}`; order of operations:
  filter by `status` (a name, a list, or `"all"`) -> `last` = keep the k most recent
  (before sorting) -> `sort_by in {None, "trial_id", "eval_index", "value"}` -> `limit` =
  keep the first k after sorting (`sort_by="value", limit=k` is the top k).
* **status** — `{"n_observed", "n_pending", "n_abandoned", "is_moo", "feasible_found", "config": shelf dict, "space": spec, "incumbent": ...}`.

Model cache: GPs per metric rebuilt when the data version or model settings change;
hyperparameters warm-started from the last fit. Fit only on completed trials with a
finite value for that metric.

### 7.3 CLI (paper App. H)

`python -m agentic_bo.lenz --study STUDY.json <command> [--flags]` prints exactly one
JSON line (the envelope). Commands use the paper's hyphenated names:
`create --space JSON --objectives JSON [--constraints JSON] [--acqf NAME] [--seed N]`,
`suggest [--q N] [--around incumbent|JSON] [--radius R] [--bounds JSON] [--acqf NAME] [--acqf-params JSON]`,
`submit --config JSON`, `observe (--trial-id N | --config JSON) --metrics JSON`,
`set-bounds --bounds JSON|full`, `set-acqf --acqf NAME [--beta B] [--params JSON]`,
`set-objectives --objectives JSON`, `set-constraints --constraints JSON`,
`diagnostics`, `predict --configs JSON`, `score --configs JSON [--acqf A,B]`,
`incumbent`, `pareto`, `trials [--last N] [--status S[,S]] [--sort-by K] [--limit N]`,
`status`. The state file is authoritative
(load -> run -> save on mutation). Exit code 0 on `ok`, 1 otherwise.

---

## 8. Problems (`agentic_bo.problems`) and metrics

```python
@dataclass
class Problem:
    name: str
    space: SearchSpace
    evaluate_fn: Callable[[dict], dict]   # noise-free metrics for a config
    objective: str                        # primary metric name
    minimize: bool                        # raw direction of the primary metric
    f_star: float | None                  # optimal raw value of the primary metric (None if unknown)
    constraints: list[dict] = []          # [{"metric", "lower", "upper"}]
    noise_std: float = 0.0                # Gaussian noise on observed metrics
    description: str = ""                 # natural-language brief shown to agents (no optimum leaks)
    prior: str | None = None              # optional natural-language domain knowledge
    prior_hint: dict | None = None        # optional structured prior for scripted agents: {"config": {...}} or {"region": {...}}
    default_budget: int = 50
    x_star: dict | None = None
    # multi-objective
    objectives: list[dict] | None = None  # [{"metric", "minimize"}] when |O| >= 2
    ref_point: list[float] | None = None  # raw units
    max_hv: float | None = None
    def evaluate(self, config) -> dict                 # noise-free
    def observe(self, config, rng) -> dict             # noisy (noise_std) — what the agent sees (the harness applies
                                                       # the same noise to its single evaluate call instead of calling it)
    def utility(self, metrics) -> float                # maximize-convention primary value: -v if minimize
    @property u_star -> float | None                   # maximize-convention optimum
    def is_feasible(self, metrics) -> bool
    def regret(self, metrics) -> float                 # raw u_star - utility(metrics): >= 0 (up to fp) for feasible points;
                                                       # an infeasible point that beats the constrained optimum is negative
    def brief(self) -> str                             # markdown brief: description, inputs (space.describe()), outputs, constraints,
                                                       # "## Noise": the noise std, or "Observations are noiseless" when
                                                       # noise_std == 0 and the class attribute `deterministic` is True
                                                       # (False for api.make_problem wrappers of user black boxes)
    @property is_moo -> bool

def get_problem(name: str, seed: int | None = None, randomize: bool = False,
                noise_std: float = 0.0, **kwargs) -> Problem
def list_problems() -> list[str]
```

Registered names (minimum): `branin`, `six_hump_camel`, `goldstein_price`, `hartmann3`,
`hartmann6`, `ackley{d}` (e.g. `ackley2`, `ackley10`, `ackley20`), `levy{d}`,
`rosenbrock{d}`, `styblinski_tang{d}`, `rastrigin{d}`, `griewank{d}`,
`constrained_hartmann6` (BoTorch's: constraint `1 - ||x||_2^2 >= 0`, i.e. metric
`c = ||x||^2` with `upper = 1`; the global optimum is feasible), `branin_currin`
(2 objectives, both minimized, ref point `[18, 6]`, `max_hv = 59.36011874867746`), and
`reaction_yield` (6-D toy Suzuki-type yield model in `applied.py`: temperature [0,150] °C,
pressure [1,10] bar, catalyst_loading [0.1,10] mol% (log), residence_time [1,180] min
(log), pH [6,13], water_fraction [0,1]; metrics `yield` (maximize, max 95 %) and `cost`;
smooth chemistry-flavoured structure; `prior` text with realistic chemistry knowledge
consistent with the optimum; `f_star` computed numerically at import time with dense
multi-start search and cached). Budgets follow paper App. A where they apply
(branin 50, hartmann6 100, ackley10 150, ackley20 200, constrained_hartmann6 100,
reaction_yield 40).

**Randomization** (paper App. A anti-memorization; `randomize=True`): canonicalize the
domain to `[0,1]^d`, rename params to deterministic random tokens (e.g. `"x_3fa"`), and —
only for functions whose known optimum is global over R^d (`shift_safe`) — shift the
optimum by up to a quarter of each range while keeping it inside `[0.05, 0.95]` of the
canonical box. `f_star` is unchanged; `x_star` is mapped. Descriptions of randomized
problems must not name the function.

`agentic_bo.metrics`:

```python
simple_regret_curve(utilities, u_star, feasible=None) -> np.ndarray   # r_t = u* - max_{s<=t} u_s over feasible points;
                                                                       # before the first feasible point: max(regret of the worst point so far, 0)
cumulative_regret_curve(utilities, u_star, feasible=None) -> np.ndarray  # R_t = sum_{s<=t} (u* - u_s); infeasible or failed points charged
                                                                          # max(u* - u_s, max regret seen so far, 0) (never below 0);
                                                                          # failures before the first finite observation k are charged
                                                                          # the charge at k: R_t is NaN only for t < k (all NaN if nothing is finite)
hypervolume_regret_curve(Y_true, feasible, ref_point, max_hv, minimize_flags) -> np.ndarray
                                                                       # max_hv - HV_t; with zero evaluations and a known max_hv: max_hv
summarize_curves(curves: list[np.ndarray]) -> dict   # {"median", "q25", "q75", "mean", "sem"} lists
```

---

## 9. Sara harness (`agentic_bo.sara`)

### 9.1 Tools (`tools.py`)

```python
@dataclass
class ToolSpec:
    name: str
    action_class: str        # "probe" | "reconfigure" | "propose" | "evaluate" | "stop" | "note" | "setup"
    description: str         # concise, accurate, LLM-facing
    input_schema: dict       # JSON schema, type=object, additionalProperties=False
    handler: Callable[["AgentSession", dict], dict]

def build_tools(session, toolset: str = "full") -> list[ToolSpec]   # sorted by name
TOOLSETS = {"full": [...], "llm_only": ["evaluate", "incumbent", "record_note", "status", "trials"(, "stop")]}
```

Agent-facing tools (snake_case; map 1:1 to lenz commands except the harness ones):
`status`, `trials {last?, sort_by?, limit?}` (`last` before sorting, `limit` after: top k
with `sort_by="value"`), `incumbent`, `pareto`, `diagnostics {metric?}`,
`predict {configs}` (<= 50), `score {configs, acqfs?}`, `set_bounds {bounds|null}`,
`set_acqf {acqf, beta?, params?}`, `set_objectives {objectives}`, `set_constraints {constraints}`,
`suggest {q?, around?, radius?, bounds?, acqf?, beta?, params?}` (q <= 10; `beta`/`params`
map to `lenz.suggest(acqf_params={**params, "beta": beta})` for this call only; without
`acqf` they adjust the current function), **`evaluate {config,
rationale, expectation?}`**, `stop {reason}` (only when early stopping is allowed; needs
one evaluation and is rejected once the whole budget is used), `record_note {text}`, and
`create {...}` only with `agent_setup=True`. `predict`/`score` default to the current
acquisition, or to `logei` (`ehvi`/`parego` for several objectives) when it is `ts`,
`sobol` or `random`.

Results are passed through `present`: floats keep 6 significant digits, configs and
values of int/ordinal/categorical parameters are kept exact (`round_param_value`, real
values with enough digits to be re-submitted), and metric values (floats keyed by a
metric name, `value` or `mean`) keep 10 significant digits so that their resolution does
not depend on the metric's offset. `schema.validate` compares integers beyond the float
range exactly (error messages, never `OverflowError`).

Config schemas are generated per problem (one property per param: `number`/`integer`/
`enum`; all required; `additionalProperties: false`) so tools are fixed for the whole
campaign (prompt-cache friendly). `rationale` is required: Sara must state belief and
learning goal before every submission (paper §5.2 directive 5).

### 9.2 Session and campaign (`session.py`)

```python
@dataclass
class ContextItem: t: int; source: str; text: str; data: dict | None = None   # K_t entry ("initial", "instructor", ...)

@dataclass
class CampaignConfig:
    budget: int | None = None                 # None -> problem.default_budget
    regret: str = "simple"                    # "simple" | "cumulative" (what the agent is told to minimize)
    allow_early_stop: bool = False
    soft_actions_per_step: int = 12           # beyond this, tool results carry a warning
    hard_actions_per_step: int = 30           # beyond this, every call of an open step (also unknown tools and
                                              # malformed arguments) is replaced by a fallback commit, except a
                                              # valid evaluate / stop, which still runs (fallback only if it fails)
    max_evals_per_step: int = 1               # harness-level enforcement of sequential reasoning
    context_schedule: dict = {}               # {t: str | dict | list}: exogenous instructions injected before step t+1 (t = #completed evals)
    agent_setup: bool = False
    toolset: str = "full"
    seed: int = 0
    lenz_options: dict = {}                   # passed to Lenz(...) (model, acqf, ...)
    show_prior: bool = True                   # include problem.prior in the brief
    context: Any = None                       # initial context items (str | dict | ContextItem | list)
    on_policy_error: str = "fallback"         # "fallback": record a policy exception, finish with fallbacks; "raise"

class AgentSession:
    problem; lenz: Lenz; config: CampaignConfig; tools: list[ToolSpec]
    t: int                                    # completed evaluations
    budget: int; remaining: int
    context: list[ContextItem]                # K_t (append-only)
    notebook: list[dict]
    log: list[dict]                           # H_t: {"t", "j", "tool", "action_class", "args", "ok", "result"|"error", "forced"}
    records: list[dict]                       # one per evaluation: {"t", "trial_id", "config", "observed", "true_metrics",
                                              #   "utility", "feasible", "regret", "source", "rationale", "expectation", "forced"}
    step_committed: bool; stopped: bool; actions_this_step: int
    def call(self, name: str, args: dict | None = None) -> dict   # envelope {"ok": True, "result": ...} | {"ok": False, "error": str}
                                                                  # validates args (schema.py), enforces limits, logs, never raises
    def brief(self) -> str                    # initial agent brief: problem.brief() + budget + regret goal + prior + initial context
    def status_line(self) -> str              # e.g. "Step 7/50 | remaining 44 | best y=0.41 (trial 5)"
    def begin_step(self) -> None
    def add_context(self, item: ContextItem) -> None
    def commit_fallback(self, reason: str) -> dict   # top `suggest` candidate (or Sobol), evaluated with forced=True
    def tool_definitions(self) -> list[dict]  # Anthropic tool dicts {"name","description","input_schema"} sorted by name

class Campaign:
    def __init__(self, problem, policy, config: CampaignConfig | None = None): ...
    def run(self) -> "CampaignResult"
        # policy.start(session)
        # while session.t < budget and not session.stopped:
        #     inject context_schedule items due at session.t (session.add_context + policy.on_context)
        #     if session.stopped: break                 # a stop issued while handling the context
        #     session.begin_step(); policy.step(session)
        #     if not session.step_committed: session.commit_fallback("policy ended step without evaluating")
        # policy.finish(session)
        # An exception (any Exception, not only PolicyError) raised by a policy hook (start,
        # on_context, step, finish) -> recorded as policy_error "<Type>: <message>"; remaining steps
        # use commit_fallback (config.on_policy_error="fallback"|"raise"). BaseException
        # (KeyboardInterrupt, api._Abort) propagates.

@dataclass
class CampaignResult:
    problem: str; policy: str; seed: int; budget: int; regret_mode: str
    records: list[dict]; simple_regret: list[float]; cumulative_regret: list[float]
    best_value: float | None; best_config: dict | None
    actions_per_step: list[int]; action_counts: dict; forced_commits: int
    context: list[dict]; notebook: list[dict]; log: list[dict]
    usage: dict; wallclock_s: float; stopped_early: bool; policy_error: str | None
    def to_dict(self) -> dict; def save(self, path) -> None; @classmethod load(path)
```

`evaluate` semantics: validate config against the domain (outside the active region is
allowed — the agent may override lenz — but the result carries a note); reject when
the budget is exhausted or the step already committed (`max_evals_per_step`);
`lenz.submit(source=...)` -> `problem.evaluate(config)` (noise-free; the black box runs
exactly once) -> the harness adds `N(0, noise_std^2)` per metric from the session noise
stream (`Problem.observe` semantics; `problem.observe` itself is not called) ->
`lenz.observe` -> record true (noise-free) metrics/utility/regret -> return
`{"trial_id", "config", "observed": metrics, "feasible", "is_new_best", "incumbent", "t", "remaining", "note"?}`.
Predictions never enter `D_t` (there is no agent-facing `observe`; anti-pattern guard).

Early stop (if allowed; `stop` needs at least one evaluation and is rejected once the
whole budget is used): simple regret stays at its last value; cumulative regret charges
every unused evaluation the regret of the best observed point (as if the agent exploited
it), so stopping cannot game `R_T`.

### 9.3 Policies

```python
class PolicyError(RuntimeError): ...
class Policy(ABC):
    name: str = "policy"
    def start(self, session) -> None: ...
    @abstractmethod
    def step(self, session) -> None: ...      # must lead to session.call("evaluate"|"stop", ...)
    def on_context(self, session, item: ContextItem) -> None: ...
    def finish(self, session) -> None: ...
    def usage(self) -> dict: return {}

def get_policy(name: str, **kwargs) -> Policy
# names: "sobol", "random", "fixed_bo", "heuristic", "sara" (Claude, full toolset), "llm_only" (Claude, LLM-as-optimizer ablation)
```

Policies interact **only** through `session.call(...)` (same interface as the LLM), so
traces are comparable.

* `SobolPolicy` — `suggest(acqf="sobol")` -> evaluate.
* `RandomPolicy` — uniform random configs -> evaluate.
* `FixedBOPolicy(n_init=5, acqf="logei")` — `Pi_0` (Ax-like): Sobol for `n_init` steps, then
  evaluate the top `suggest` candidate every step; never reconfigures.
* `HeuristicSaraPolicy` — a scripted agent following Sara's directives with the same
  tools: adaptive opening (prior config -> evaluate it; prior region -> `set_bounds` then
  space-filling; else a short Sobol design), per-step `diagnostics` gating (low CV R^2 =>
  don't trust the surrogate: explore), global vs trust-region (`suggest(around="incumbent")`)
  candidates compared with `score`/`predict`, TuRBO-style radius adaptation on
  success/failure streaks, stagnation handling (switch acquisition to UCB / widen or
  restart the region), duplicate avoidance, end-game exploitation, cumulative-regret mode
  (less exploration), structured `ContextItem.data` instructions applied via reconfigure
  tools, periodic `record_note`, and a rationale on every evaluation. Must be at least as
  good as `FixedBOPolicy` on the synthetic suite (median final simple regret).
  Robustness details: the TuRBO success tolerance is `max(1e-3 * min(|u|, best_u -
  worst_u), declared noise std)` (TuRBO's relative tolerance capped at 0.1% of the
  observed utility range, so an additive offset does not swamp it); the regime
  acquisition (UCB with `cumulative_beta` under cumulative regret, `noisy_logei` when the
  brief declares noise) is re-applied when an objectives change resets the acquisition;
  initial context items go through `on_context` (errors logged, `strict=True` re-raises);
  instructions whose `args` is not an object, and prior/instructed configs that
  `evaluate` rejects, are acknowledged with `record_note` instead of failing.
* `ClaudeSaraPolicy` — §10.

---

## 10. Claude policy (`sara/policies/llm.py`)

```python
class ClaudeSaraPolicy(Policy):
    def __init__(self, model="claude-opus-5-5", effort="high", max_tokens=32000,
                 thinking_display="summarized", use_fallbacks=True, use_compaction=True,
                 compaction_trigger_tokens=150_000,      # int >= MIN_COMPACTION_TRIGGER = 50_000 (API minimum)
                 cache=True, eager_input_streaming=True,
                 max_api_calls_per_step=40, max_nudges=3, client=None,
                 transcript_path=None, toolset="full", system_prompt=None, verbose=False)
```

Anthropic API contract (from the bundled claude-api skill; **do not deviate**):

* `anthropic` is an optional import. Missing package -> `PolicyError` with install hint.
  `client=None` -> `anthropic.Anthropic()` (resolves `ANTHROPIC_API_KEY`, auth token or
  `ant auth login` profile).
* Model `claude-opus-5-5`. Thinking: `{"type": "adaptive", "display": "summarized"}`
  (never `budget_tokens`, never `disabled`). Effort via `output_config={"effort": effort}`
  (Opus 5.5 default would be `medium`; the paper ran "high").
* Streaming: `with client.beta.messages.stream(**kwargs) as stream: msg = stream.get_final_message()`.
  Client tools carry `"eager_input_streaming": True`; therefore **validate every tool
  input against its schema** (session.call does) and return an `is_error` tool_result on
  failure. Guard `ValueError` raised by the stream for unparseable JSON (re-issue the
  request; cap retries); never catch the SDK's typed API errors as JSON errors.
* `tool_choice` stays `auto` (forced tool choice is a 400 on Opus 5.5); steer with the prompt.
* Betas (each toggleable): `server-side-fallback-2026-07-01` with `fallbacks="default"`;
  `compact-2026-01-12` with `context_management={"edits": [{"type": "compact_20260112",
  "trigger": {"type": "input_tokens", "value": N}, "instructions": COMPACTION.md}]}`.
* Prompt caching: frozen `system` = `[{"type":"text","text": SYSTEM.md}, {"type":"text","text": LENZ_REF.md, "cache_control": {"type": "ephemeral"}}]`,
  tools sorted by name and serialized deterministically, plus top-level
  `cache_control={"type": "ephemeral"}` for the growing tail. Nothing volatile in
  `system`/`tools`.
* **Append-only history (preserved thinking).** Append `response.content` verbatim as
  the assistant turn (keeps thinking + compaction blocks). Never edit, reorder, or delete
  earlier messages. All tool_result blocks for one assistant message go in **one** user
  message (results first, then optional text blocks such as the step status line).
  New exogenous instructions are appended as `{"role": "system", "content": "..."}` right
  after that user message (operator-authority, cache-safe). Placement rule: a system
  message must follow a user message and be last or followed by an assistant turn, so
  all context items due before the same step share **one** system message (their texts
  joined in arrival order, separated by a blank line). If the API rejects
  mid-conversation system messages (only a 400 matching the documented "role 'system' is
  not supported" wording), fall back to `<system-reminder>` text blocks inside the user
  message (one per item). The staged user/system turn is committed together with the
  reply, and only when the reply has content: an empty reply commits nothing and the
  identical body is re-sent (if the step is abandoned, the next step's texts join the
  same staged user message). Transcript request events after an empty reply repeat the
  uncommitted messages; place `new_messages` at offset `n_messages - len(new_messages)`.
* Stop reasons: `tool_use` -> execute tool_use blocks in order through `session.call`;
  `end_turn` without a committed evaluation -> append a short user nudge ("step t is still
  open: call evaluate ..."), at most `max_nudges` then return (harness fallback);
  `max_tokens` or `model_context_window_exceeded` with tool_use -> do not execute (the
  input may be truncated); return `is_error` results naming the limit and asking for a
  shorter call (without tool_use: a `[harness]` continue text); `usage.max_tokens_stops`
  counts `max_tokens` only. `pause_turn` -> re-send; `refusal` -> do not read, log or
  execute the refused content; log a `kind="policy"` note with the refusal category and
  explanation, then raise `PolicyError` (harness continues with fallbacks). Check
  `stop_reason` before reading content.
* `step()` returns as soon as a processed assistant message committed the step
  (evaluate/stop). Tool calls after the evaluate in the same message still execute
  (they count toward the next step); a second evaluate is rejected by the session.
* Usage: accumulate `input_tokens`, `output_tokens`, `cache_read_input_tokens`,
  `cache_creation_input_tokens`, API call count; cost estimate for Opus 5.5
  ($4 in / $20 out / cache read 0.05x = $0.20 / 5-min cache write 1.25x = $5 per MTok).
* Reasoning traces: summarized thinking + text blocks are appended to `session.log`
  (H_t) per step (never those of a refused turn). Optional JSONL transcript (`transcript_path`) of requests (without
  repeated system/tools) and responses (`.to_dict()` when available).
* Tests use a fake client that records each request's kwargs and replays scripted
  responses; they assert the request contract above and that every request's `messages`
  is an append-only extension of the previous request's.

Prompts (`prompts/*.md`) — reconstructed from paper §5.2 (the paper's SYSTEM.md text is not
public): agent-led, surrogate-assisted posture; one reasoning step per
suggest -> evaluate -> observe cycle; adaptive opening; trust calibration with
diagnostics; strategy revision (bounds, acquisition, targeted evaluations, reframing on
new instructions, data is never lost); explicit anti-patterns (deferring to an
untrustworthy posterior, treating predictions as observations, ignoring context,
mechanical suggest loops, duplicate evaluations, premature region collapse); state
belief + learning goal before each submission; awareness of simple vs cumulative regret.
Calm, direct language (no all-caps emphasis). `LENZ_REF.md` documents every tool,
argument, and result shape. `COMPACTION.md` says what a summary must retain (objectives
and instructions in force, lenz config, incumbent, landscape hypotheses, sensitive
params, model trust, plan, budget) and that the full trial log stays available via `trials`.

---

## 11. Experiments, API, examples

```python
# agentic_bo/api.py
def optimize(f, space, budget, *, maximize=False, policy="heuristic", seed=0, f_star=None,
             description="", prior=None, constraints=None, noise_std=0.0, context=None,
             regret="simple", policy_kwargs=None, verbose=False) -> CampaignResult
# f: config dict -> float (objective) or dict of metrics (objective under key "y").

# agentic_bo/experiment.py
def run_campaign(problem, policy, seed=0, budget=None, randomize=True, noise_std=0.0, regret="simple", **cfg) -> CampaignResult
def run_benchmark(problems, policies, seeds, out_dir, budget=None, n_jobs=1, randomize=True, noise_std=0.0) -> list[dict]
def summarize(results) -> dict          # per problem x policy: median/IQR final simple & cumulative regret, wins/ties/losses
def compare(results, reference="fixed_bo") -> list[dict]   # two-sided Mann-Whitney U per problem, p >= 0.05 = tie;
                                                           # exact permutation p-value of the midrank sum (ties included)
                                                           # whenever C(n+m, min(n,m)) <= 50,000, scipy's default above
def plot_regret(results, out_path, kind="simple", log_scale=True)  # median + 25-75% IQR bands (paper §6)
# CLI: python -m agentic_bo.experiment run --problems branin hartmann6 --policies heuristic fixed_bo sobol --seeds 5 --out results/
#      python -m agentic_bo.experiment report --results results/      # or a single run file (plots go next to it)
```

LLM policies are never run by tests or default benchmarks (they cost money); examples
show how to run them.
