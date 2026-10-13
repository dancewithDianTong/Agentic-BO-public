# lenz reference

lenz is the Bayesian backend of the campaign: a trial log, one Gaussian-process surrogate per modelled metric (Matern-5/2 kernel with a lengthscale per input, inputs normalized to [0, 1], outputs standardized), acquisition functions and an acquisition optimizer. You reach it through the tools below. Every tool except `evaluate` is free: it spends no budget, and only `evaluate` adds data. A tool that fails returns an error message and changes nothing; fix the arguments and call it again.

## Conventions

- Config: an object with a value for every parameter, inside its domain, in raw units (see the parameter table in the brief). Integers must be integers; ordinal and categorical values must match a listed value exactly. Config values in results keep enough digits to be copied back exactly.
- Metrics are reported in raw units and in their own direction: a minimized metric is better when lower. Observations carry noise when the problem is noisy. A failed measurement is reported as null and is left out of the models.
- Acquisition values are "higher is better" within one acquisition function. Values of different functions are not comparable.
- Active region (`bounds`): `{param: [lo, hi]}` for numeric parameters (raw units, inside the domain) and `{param: [allowed values]}` for ordinal and categorical parameters. Omitted parameters keep their full domain; null means the full domain. The region steers `suggest`; `evaluate` may go outside it.
- Trust region: `around` (`"incumbent"` or a config) with `radius`, a fraction of each parameter's range in (0, 1], scaled per parameter by the GP lengthscales and intersected with the active region.
- Constraints: `{metric, lower, upper}` holds when lower <= metric <= upper (null for an open side). The incumbent, the Pareto set and `sort_by="value"` treat feasible trials as better than infeasible ones.
- Two configs closer than about 1e-6 (RMS distance in normalized units) count as the same point.
- Floats in results have six significant digits; metric values (observed metrics, `value`, predicted `mean`) keep ten, so small differences on top of a large offset stay visible. Very long results are truncated with a note that says how to see the rest.

## Probes (read-only)

### status
Arguments: none.
Returns: `step`, `evaluations_done`, `budget`, `remaining`, `regret_mode`, `early_stop_allowed`, `step_committed`, `actions_this_step`, `action_limits` {soft, hard}, `n_observed`, `n_feasible`, `feasible_found`, `is_moo`, `incumbent`, `lenz_config` {objectives, constraints, acqf, acqf_params, bounds, model} and `recent_notes`.
Use it to re-orient, for example after new instructions or a long exchange.

### trials
Arguments: `last` (keep only the k most recent trials; applied before sorting), `sort_by` (`"value"`: best first, feasible before infeasible, in the objective's own direction; `"eval_index"`: evaluation order), `limit` (keep only the first k rows after sorting). For the k best trials use `sort_by="value"` with `limit=k`; `last` with `sort_by="value"` sorts only the most recent trials.
Returns: `trials` [{trial_id, eval_index, source, config, metrics, feasible}], `n` (rows shown), `n_total`.
The trial log is the single source of truth. `source` tells who chose the point: `agent` for your evaluations, `harness-fallback` for points the harness evaluated.

### incumbent
Arguments: none.
Returns: `incumbent` {trial_id, config, metrics, value} (the best feasible observed trial, or null), `best_posterior` {trial_id, config, mean} (the evaluated trial with the best posterior mean; the noise-robust choice when observations are noisy), `feasible_found`, `is_moo`, and `least_infeasible` {trial_id, config, metrics} while constraints exist and nothing is feasible.
With several objectives it returns `pareto` (trial rows), `hypervolume` and `ref_point` instead, and `incumbent` is null.

### pareto
Arguments: none.
Returns: `pareto` (non-dominated feasible trials), `hypervolume` (dominated by them, relative to `ref_point`), `ref_point` (raw units), `n_feasible`, `is_moo`.

### diagnostics
Arguments: `metric` (optional; default every objective and constraint).
Returns: `models` {metric: {role, n_observed, cv_r2, noise_std, signal_std, outputscale, lengthscales {param}, sensitivity {param}, log_marginal_likelihood}} and `objective` (the first objective).
Reading it:
- `cv_r2`: leave-one-out cross-validated R^2. Near 1: the model predicts unseen points well. Below about 0.5: treat predictions and suggestions with suspicion. Negative: worse than predicting the mean. Null with fewer than 3 observations. It is one number for all the data, dominated by the largest deviations: on a heavy-tailed objective (a few values far worse than the rest) it can be high while predictions near the best values are poor, so check the model there by comparing your expectations with the observations.
- `noise_std` and `signal_std` (raw units): a noise level close to the signal level means the model explains little of the variation. The fitted noise also absorbs model misfit, so it does not tell you whether observations are noisy (the brief does): a sizable `noise_std` on a noiseless problem means misfit, not noise.
- `lengthscales` (normalized units, a fraction of each parameter's range): small values (below about 0.1) mean the metric changes quickly along that parameter; very large values mean the parameter has had little effect so far. `sensitivity` normalizes inverse lengthscales to sum to 1. A categorical parameter's sensitivity is summed over its choices and can look larger than it is.

### predict
Arguments: `configs` (1 to 50 configs; they may lie outside the active region).
Returns: `predictions` [{config, metrics {metric: {mean, std, variance}}, prob_feasible, acquisition_values {acqf: value}, in_region}], `acqf`, `n_observed`.
The acquisition value is under the current function; when that is `ts`, `sobol` or `random` (no pointwise value) it is under `logei` (`ehvi` or `parego` for several objectives), and `acqf` says which.
Use it to test a hypothesis or a hand-made candidate before spending an evaluation, and to write a quantitative `expectation`.

### score
Arguments: `configs` (1 to 50), `acqfs` (list of pointwise acquisition functions: logei, noisy_logei, ei, logpi, pi, ucb, pm, std, ehvi, parego; default the current one, or `logei` (`ehvi`/`parego` for several objectives) when the current one is `ts`, `sobol` or `random`).
Returns: `scores` [{config, acquisition_values}], `acqfs`, `best` {acqf: index of the top config}, `notes`.
Use it to compare candidates from different sources (global, trust region, your own) on a common scale.

## Proposals

### suggest
Arguments: `q` (number of candidates, 1 to 10, default 1), `acqf` (an acquisition function for this call only), `beta` and `params` (acquisition parameters for this call only, as in `set_acqf`; without `acqf` they adjust the current function), `bounds` (a region for this call only; `"full"` = the whole domain), `around` (`"incumbent"` or a config: search a trust region around it), `radius` (trust-region half-width, default 0.1).
Returns: `candidates` [{config, acquisition_values, predicted {metric: {mean, std}}, prob_feasible, duplicate_of, trial_id: null}], `acqf`, `acqf_params`, `mode` (`"model"` or `"space_filling"`), `region` (`"full domain"`, or the bounds, around, radius and the resulting box), `n_observed`, `feasible_found`, `feasibility_phase`, `notes`.
Behaviour:
- Records nothing: candidates are proposals until you evaluate one.
- While fewer than two observations report every objective, and whenever `acqf` is `sobol` or `random`, it returns space-filling points. The Sobol design continues across calls and skips points already in the log. For `sobol` and `random`, `predicted` is null (use `predict`).
- `q > 1` builds a greedy batch: each pick is conditioned on the previous ones with fantasized outcomes, so the batch is varied. It is a shortlist; you still evaluate one point per step.
- With constraints and no feasible observation yet, the EI-type and multi-objective acquisition functions search for feasibility first (`feasibility_phase: true`). In a batch, once an earlier pick is predicted feasible, later picks treat it as the incumbent instead of piling up next to it (`notes` say so).
- `ehvi` before any feasible evaluated point dominates the reference point (hypervolume 0): the expected improvement is computed in closed form from each objective's improvement over its reference value, so it still ranks candidates; `notes` name the reference point.
- `around="incumbent"` centres on the best feasible trial; for several objectives, on a balanced Pareto point; while nothing is feasible, on the least infeasible trial. With a radius below about 0.25, categorical parameters keep the centre's value. On integer and ordinal parameters the trust region always reaches the neighbouring values of the centre (inside the region), so a small radius does not pin them; `region.box` shows the box searched.
- `duplicate_of` names an evaluated trial that a candidate repeats; this happens when the searched region holds nothing new for the acquisition function. Evaluating it wastes budget: widen the `radius`, suggest globally (without `around`), or use a more exploratory acquisition function. `notes` explain fallbacks such as space filling, the feasibility phase or duplicates.

## Reconfiguration (never drops data)

### set_bounds
Arguments: `bounds` (the new active region, or null for the full domain).
Returns: `bounds`, `observed`, `n_in_region`, `is_moo`.
Every trial stays in the models, including those outside the region.

### set_acqf
Arguments: `acqf`, `beta` (ucb only; default 4, larger explores more), `params` (advanced: ehvi `ref_point` in raw units and `n_samples`; parego `weights` and `rho`; ts `n_candidates`).
Returns: `acqf`, `acqf_params`, `is_moo`, `observed`, `notes` (for example when an ehvi `ref_point` is ignored because the objectives carry their own `ref`).
Acquisition functions:
- One objective: `logei` (default; log expected improvement), `noisy_logei` (improvement over the best posterior mean; for noisy observations), `ei`, `logpi` and `pi` (probability of improvement; more exploitative), `ucb` (mean + sqrt(beta) * std), `pm` (posterior mean; pure exploitation), `std` (posterior std; pure exploration), `ts` (Thompson sampling: a random draw from the posterior, maximized).
- Several objectives: `ehvi` (expected hypervolume improvement; exactly two objectives), `parego` (random Chebyshev scalarizations).
- Model-free: `sobol` (space filling), `random`.
Passing a parameter that the chosen function does not take is an error.

### set_objectives
Arguments: `objectives` [{metric, minimize, ref (optional hypervolume reference value, raw units)}]. Two or more objectives make the study multi-objective.
Returns: `objectives`, `is_moo`, `acqf`, `acqf_params`, `acqf_switched`, `previous_acqf`, `notes`, `observed`, `n_with_objectives`.
The acquisition function switches between the single- and multi-objective families when needed, and the result says so. When the objectives change, a stored ehvi `ref_point` (in the old objectives' units) is dropped and `notes` say so; an objective's own `ref` takes precedence over `ref_point` anyway. Only metrics that the problem reports can be used.

### set_constraints
Arguments: `constraints` [{metric, lower, upper}] (at least one bound each; `[]` removes all constraints).
Returns: `constraints`, `feasible_found`, `n_feasible`, `observed`, `is_moo`.
Acquisition functions then weight candidates by their probability of feasibility.

## Commit and notes

### evaluate
Arguments: `config`, `rationale` (required: your belief about the point and what you expect to learn), `expectation` (optional: your predicted outcome).
Returns: `trial_id`, `config`, `observed` (the measured metrics), `feasible`, `is_new_best`, `incumbent`, `t` (evaluations used), `remaining`, and `note` when the config lies outside the active region, repeats an evaluated trial, or used the last evaluation.
Spends one evaluation and ends the step. A second `evaluate` in the same step is rejected.

### record_note
Arguments: `text`.
Returns: `recorded`, `n_notes`.
Free. The most recent notes appear in `status`.

### stop
Available only when the campaign allows early stopping. Arguments: `reason`.
Returns: `stopped`, `evaluations_done`, `unused_budget`, `reason`.
Ends the campaign and forfeits the remaining evaluations. Under simple regret the result is the best point found; under cumulative regret every unused evaluation is charged as if it repeated your best point. It needs at least one evaluation and is rejected once the whole budget is used (the campaign then ends by itself).

### create
Available only when the campaign lets you set up the study. Arguments (all optional; omitted ones keep their current setting): `objectives`, `constraints`, `acqf`, `bounds`, `model` {kernel: matern52 or rbf, noise: "learn" or a fixed variance in standardized units, n_restarts}.
Rebuilds the study with these settings and carries every trial over.

## Limits

A step allows a number of free actions (see `action_limits` in `status`). Beyond the soft limit, results carry a warning. Beyond the hard limit, any other call (including one that fails, such as an unknown tool or invalid arguments) is not executed: the harness evaluates a fallback point for you and the step ends. A valid `evaluate` or `stop` is still carried out; if it fails, the fallback follows.
