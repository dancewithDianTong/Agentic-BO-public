"""Multi-seed benchmarks of agentic BO policies: runner, statistics, plots and a CLI (paper §6).

    >>> from agentic_bo.experiment import run_benchmark, summarize, plot_regret
    >>> rows = run_benchmark(["branin", "hartmann6"], ["heuristic", "fixed_bo", "sobol"], seeds=5,
    ...                      out_dir="results/", n_jobs=4)
    >>> summary = summarize(rows)                       # median / IQR final regret, wins vs fixed_bo
    >>> plot_regret(rows, "results/regret_simple.png")  # median + 25-75 % bands, one panel per problem

Command line::

    python -m agentic_bo.experiment run --problems branin hartmann6 --policies heuristic fixed_bo sobol \\
        --seeds 5 --out results/ --n-jobs 4
    python -m agentic_bo.experiment report --results results/
    python -m agentic_bo.experiment list

Protocol (paper App. A / §6): by default every seed runs on its own *randomized*
instance of a synthetic problem (canonical box, anonymous parameter names, shifted
optimum). Problems whose natural-language prior refers to the real parameter names
(``reaction_yield``) and problems with non-real parameters are not randomized. Policies
are compared per problem on the final regret with a two-sided Mann-Whitney U test
(exact permutation p-values for small samples, ties included); ``p >= 0.05`` counts as a
tie. With fewer than 4 seeds per side no difference can be significant at that level.

``run_benchmark`` writes one JSON file per run (``out_dir/runs/<problem>__<policy>__seed<k>.json``,
the full :class:`~agentic_bo.sara.CampaignResult` plus the run settings) and
``out_dir/summary.json``. Runs whose file exists are loaded instead of re-run, so an
interrupted benchmark resumes where it stopped. Non-LLM runs can use several processes;
LLM runs (``sara``, ``llm_only``) always run one at a time in the calling process.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import itertools
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .metrics import summarize_curves
from .problems import get_problem, list_problems, randomize_problem
from .problems.base import Problem
from .sara.policies import POLICY_NAMES, Policy, get_policy
from .sara.session import RESULT_FORMAT, Campaign, CampaignConfig, CampaignResult
from .space import RealParam
from .utils import LenzError, dumps, jsonable

__all__ = [
    "run_campaign", "prepare_problem", "run_benchmark", "load_results", "result_row", "summarize", "compare",
    "format_report", "plot_regret", "main", "SUMMARY_FORMAT", "RUN_FORMAT", "ALPHA", "PALETTE",
]

SUMMARY_FORMAT = "agentic_bo.experiment/summary"
RUN_FORMAT = RESULT_FORMAT
#: significance level of the Mann-Whitney U comparisons (p >= ALPHA is a tie)
ALPHA = 0.05
#: categorical colors in fixed order (one per policy, consistent across panels)
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_SURFACE, _INK, _INK_2, _MUTED, _GRID, _AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
_LLM_KEYS = {"sara", "llm_only", "claude", "llm"}
_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS")
_USAGE_KEYS = ("api_calls", "input_tokens", "output_tokens", "cache_read_input_tokens",
               "cache_creation_input_tokens", "cost_usd")


# =========================================================================== single campaigns


def _randomizable(problem: Problem) -> bool:
    return problem.prior is None and all(isinstance(p, RealParam) for p in problem.space.params)


def prepare_problem(problem: str | Problem, seed: int = 0, randomize: bool = True,
                    noise_std: float = 0.0) -> Problem:
    """The problem instance a benchmark run uses.

    A registry name is built with :func:`~agentic_bo.problems.get_problem`; with
    ``randomize`` it becomes the seed's randomized instance unless it has a
    natural-language prior or non-real parameters (both would lose their meaning).
    A :class:`Problem` is used as given. A non-zero ``noise_std`` sets the
    observation noise in both cases.
    """
    noise = float(noise_std)
    if isinstance(problem, Problem):
        return dataclasses.replace(problem, noise_std=noise) if noise else problem
    if not isinstance(problem, str):
        raise LenzError(f"problem must be a registry name or a Problem, got {type(problem).__name__}")
    base = get_problem(problem, noise_std=noise)
    if randomize and _randomizable(base):
        return randomize_problem(base, int(seed))
    return base


def _is_llm(policy: str | Policy) -> bool:
    if isinstance(policy, str):
        return policy.strip().lower().replace("-", "_") in _LLM_KEYS
    from .sara.policies.llm import ClaudeSaraPolicy
    return isinstance(policy, ClaudeSaraPolicy)


def run_campaign(problem: str | Problem, policy: str | Policy, seed: int = 0, budget: int | None = None,
                 randomize: bool = True, noise_std: float = 0.0, regret: str = "simple", *,
                 policy_kwargs: Mapping[str, Any] | None = None, **cfg: Any) -> CampaignResult:
    """Run one campaign and return its result.

    Args:
        problem: registry name or :class:`Problem` (see :func:`prepare_problem`).
        policy: policy name (``heuristic``, ``fixed_bo``, ``sobol``, ``random``, ``sara``,
            ``llm_only``) or a :class:`Policy` instance.
        seed: campaign seed; also selects the randomized problem instance.
        budget: evaluations (``None``: the problem's default budget).
        randomize: apply the anti-memorization protocol to registry problems.
        noise_std: observation noise standard deviation.
        regret: ``"simple"`` or ``"cumulative"``.
        policy_kwargs: constructor arguments when ``policy`` is a name.
        **cfg: further :class:`~agentic_bo.sara.CampaignConfig` fields.
    """
    prob = prepare_problem(problem, seed, randomize, noise_std)
    if isinstance(policy, Policy):
        if policy_kwargs:
            raise LenzError("policy_kwargs only applies when the policy is given by name")
        pol = policy
    else:
        pol = get_policy(policy, **dict(policy_kwargs or {}))
    known = {f.name for f in dataclasses.fields(CampaignConfig)}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise LenzError(f"unknown campaign settings {unknown}; CampaignConfig fields are {sorted(known)}")
    config = CampaignConfig(budget=budget, seed=seed, regret=regret, **cfg)
    return Campaign(prob, pol, config).run()


# =========================================================================== results as rows


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def result_row(result: CampaignResult, path: str | os.PathLike | None = None) -> dict:
    """Compact per-run record (no logs) used by :func:`summarize`, :func:`compare` and :func:`plot_regret`."""
    usage = {k: result.usage[k] for k in _USAGE_KEYS if k in (result.usage or {})}
    aps = result.actions_per_step
    return {
        "problem": result.problem, "policy": result.policy, "seed": int(result.seed), "budget": int(result.budget),
        "regret_mode": result.regret_mode, "is_moo": bool(result.is_moo), "n_evals": int(result.n_evals),
        "final_simple_regret": result.final_simple_regret, "final_cumulative_regret": result.final_cumulative_regret,
        "simple_regret": [float(v) for v in result.simple_regret],
        "cumulative_regret": [float(v) for v in result.cumulative_regret],
        "best_value": result.best_value, "best_config": result.best_config,
        "forced_commits": int(result.forced_commits), "wallclock_s": float(result.wallclock_s),
        "stopped_early": bool(result.stopped_early), "policy_error": result.policy_error,
        "mean_actions_per_step": float(np.mean(aps)) if aps else None,
        "action_counts": dict(result.action_counts), "usage": usage,
        "path": None if path is None else os.fspath(path), "error": None,
    }


def _row_from_dict(d: Mapping[str, Any], path: str | None = None) -> dict:
    if "records" in d and "simple_regret" in d:     # a full CampaignResult document
        return result_row(CampaignResult.from_dict(d), path)
    if "problem" in d and "policy" in d:
        row = dict(d)
        for key in ("simple_regret", "cumulative_regret"):
            row[key] = [_num(v) for v in row.get(key) or []]
        for key in ("final_simple_regret", "final_cumulative_regret"):
            row[key] = _num(row.get(key))
        row.setdefault("error", None)
        return row
    raise LenzError("not a campaign result or result row (needs 'problem' and 'policy')")


def _read_json(path: str | os.PathLike) -> Any:
    with open(os.fspath(path), encoding="utf-8") as fh:
        return json.load(fh)


def load_results(path: str | os.PathLike) -> list[dict]:
    """Rows of the run files under ``path``.

    ``path`` is a benchmark directory (its ``runs/`` folder is read), a folder of
    result JSON files, or one result file. ``summary.json`` is skipped.
    """
    p = Path(path)
    if p.is_file():
        return [_row_from_dict(_read_json(p), str(p))]
    if not p.is_dir():
        raise LenzError(f"no results at {os.fspath(p)!r}")
    folder = p / "runs" if (p / "runs").is_dir() else p
    rows = []
    for f in sorted(folder.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            d = _read_json(f)
        except (OSError, ValueError) as exc:
            raise LenzError(f"cannot read {os.fspath(f)!r}: {exc}") from None
        if isinstance(d, Mapping) and d.get("format") == SUMMARY_FORMAT:
            continue
        rows.append(_row_from_dict(d, str(f)))
    return _benchmark_order(rows, p / "summary.json")


def _benchmark_order(rows: list[dict], summary_path: Path) -> list[dict]:
    """Sort rows like the benchmark that produced them (its problem / policy order), else keep file order."""
    try:
        settings = _read_json(summary_path)["settings"]
        probs, pols = list(settings["problems"]), list(settings["policies"])
    except (OSError, ValueError, KeyError, TypeError):
        return rows

    def key(r: Mapping[str, Any]) -> tuple:
        return (probs.index(r["problem"]) if r["problem"] in probs else len(probs),
                pols.index(r["policy"]) if r["policy"] in pols else len(pols), int(r.get("seed", 0)))
    return sorted(rows, key=key)


def _rows(results: Any) -> list[dict]:
    """Normalize results (rows, CampaignResults, result dicts, paths, or a mix) to rows."""
    if isinstance(results, (str, os.PathLike)):
        return load_results(results)
    if isinstance(results, (CampaignResult, Mapping)):
        results = [results]
    out: list[dict] = []
    for r in results:
        if isinstance(r, CampaignResult):
            out.append(result_row(r))
        elif isinstance(r, Mapping):
            out.append(_row_from_dict(r))
        elif isinstance(r, (str, os.PathLike)):
            out.extend(load_results(r))
        else:
            raise LenzError(f"cannot interpret a result of type {type(r).__name__}")
    return out


def _ok_rows(results: Any) -> list[dict]:
    return [r for r in _rows(results) if not r.get("error")]


def _ordered(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


# =========================================================================== statistics


def _stats(values: Sequence[float]) -> dict:
    v = np.asarray([_num(x) for x in values], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "mean": None}
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(med), "q25": float(q25), "q75": float(q75), "mean": float(v.mean())}


def _resolve_metric(rows: Sequence[Mapping[str, Any]], metric: str | None) -> str:
    if metric is None:
        modes = {r.get("regret_mode") for r in rows}
        return "cumulative" if modes == {"cumulative"} else "simple"
    if metric not in ("simple", "cumulative"):
        raise LenzError(f"metric must be 'simple' or 'cumulative', got {metric!r}")
    return metric


#: largest number of distinct group assignments C(n + m, min(n, m)) enumerated by the exact test
_EXACT_MAX = 50_000


def _mann_whitney(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """``(U, p)`` of the two-sided Mann-Whitney U test of ``x`` against ``y``.

    scipy's default p-value is exact only without ties; with tied values it uses the
    normal approximation, which is anti-conservative for a handful of seeds (3 vs 3
    identical regrets would give p = 0.047). Whenever the full permutation distribution
    is small enough to enumerate, the p-value is instead the exact permutation p-value
    of the midrank sum, ``2 min(P(S <= s), P(S >= s))`` capped at 1 (equal to scipy's
    exact p-value when there are no ties).
    """
    from scipy.stats import mannwhitneyu, rankdata

    res = mannwhitneyu(x, y, alternative="two-sided")
    statistic = float(res.statistic)
    p_value = float(res.pvalue) if math.isfinite(res.pvalue) else 1.0
    n, m = x.size, y.size
    k = min(n, m)
    if math.comb(n + m, k) <= _EXACT_MAX:
        ranks = rankdata(np.concatenate([x, y]))
        obs = ranks[:n].sum() if n <= m else ranks[n:].sum()
        combos = np.array(list(itertools.combinations(range(n + m), k)), dtype=np.intp).reshape(-1, k)
        sums = ranks[combos].sum(axis=1)
        tol = 1e-9 * max(1.0, abs(obs))             # midrank sums are multiples of 1/2
        p_value = min(1.0, 2.0 * min(float(np.mean(sums <= obs + tol)), float(np.mean(sums >= obs - tol))))
    return statistic, p_value


def compare(results: Any, reference: str = "fixed_bo", metric: str | None = None,
            alpha: float = ALPHA) -> list[dict]:
    """Per-problem two-sided Mann-Whitney U tests of each policy against ``reference`` (paper §6).

    Compares the final regret (``metric``: ``"simple"`` or ``"cumulative"``; default:
    cumulative if every run optimized cumulative regret, else simple) over seeds.
    ``outcome`` is ``"win"`` (lower regret, ``p < alpha``), ``"loss"``, ``"tie"``
    (``p >= alpha``) or ``None`` when either side has no finite values. ``effect`` is
    ``U / (n * n_ref)``, the probability that a run of the policy has *higher* regret
    than a reference run (ties count half): below 0.5 favours the policy. Small samples
    get the exact permutation p-value even when values are tied (see ``_mann_whitney``).
    """
    rows = _ok_rows(results)
    metric = _resolve_metric(rows, metric)
    key = f"final_{metric}_regret"
    out = []
    for prob in _ordered(r["problem"] for r in rows):
        group = [r for r in rows if r["problem"] == prob]
        ref = np.array([_num(r[key]) for r in group if r["policy"] == reference], dtype=float)
        ref = ref[np.isfinite(ref)]
        for pol in _ordered(r["policy"] for r in group):
            if pol == reference:
                continue
            x = np.array([_num(r[key]) for r in group if r["policy"] == pol], dtype=float)
            x = x[np.isfinite(x)]
            entry: dict[str, Any] = {
                "problem": prob, "policy": pol, "reference": reference, "metric": metric,
                "n": int(x.size), "n_ref": int(ref.size),
                "median": float(np.median(x)) if x.size else None,
                "median_ref": float(np.median(ref)) if ref.size else None,
                "statistic": None, "p_value": None, "effect": None, "outcome": None,
            }
            if x.size and ref.size:
                statistic, p_value = _mann_whitney(x, ref)
                effect = statistic / (x.size * ref.size)
                entry.update(statistic=statistic, p_value=p_value, effect=effect,
                             outcome="tie" if p_value >= alpha else ("win" if effect < 0.5 else "loss"))
            out.append(entry)
    return out


def summarize(results: Any, reference: str = "fixed_bo", metric: str | None = None) -> dict:
    """Per problem x policy statistics of finished runs, plus wins / ties / losses vs ``reference``.

    Returns ``{"problems", "policies", "reference", "metric", "table", "totals",
    "comparisons", "n_runs", "n_failed"}`` where ``table[problem][policy]`` holds
    ``n_runs``, ``seeds``, ``budget``, ``simple`` / ``cumulative`` / ``best_value``
    (``{"n", "median", "q25", "q75", "mean"}`` of the final values), means of
    ``wallclock_s``, ``forced_commits``, ``mean_actions_per_step``, the number of
    ``policy_errors``, the summed LLM ``cost_usd`` (when reported) and ``vs_reference``
    (the :func:`compare` outcome). ``totals[policy]`` counts the outcomes over problems.
    """
    all_rows = _rows(results)
    rows = [r for r in all_rows if not r.get("error")]
    metric = _resolve_metric(rows, metric)
    problems = _ordered(r["problem"] for r in rows)
    policies = _ordered(r["policy"] for r in rows)
    comparisons = compare(rows, reference, metric) if reference in policies else []
    outcome = {(c["problem"], c["policy"]): c for c in comparisons}
    table: dict[str, dict[str, dict]] = {}
    for prob in problems:
        table[prob] = {}
        for pol in policies:
            g = [r for r in rows if r["problem"] == prob and r["policy"] == pol]
            if not g:
                continue
            costs = [_num(r.get("usage", {}).get("cost_usd")) for r in g if (r.get("usage") or {}).get("cost_usd")
                     is not None]
            aps = [_num(r.get("mean_actions_per_step")) for r in g if r.get("mean_actions_per_step") is not None]
            cmp = outcome.get((prob, pol))
            table[prob][pol] = {
                "n_runs": len(g), "seeds": sorted(int(r["seed"]) for r in g),
                "budget": max(int(r["budget"]) for r in g), "is_moo": any(r.get("is_moo") for r in g),
                "simple": _stats([r["final_simple_regret"] for r in g]),
                "cumulative": _stats([r["final_cumulative_regret"] for r in g]),
                "best_value": _stats([r.get("best_value") for r in g]),
                "wallclock_s": float(np.mean([_num(r.get("wallclock_s")) for r in g])),
                "forced_commits": float(np.mean([_num(r.get("forced_commits")) for r in g])),
                "mean_actions_per_step": float(np.mean(aps)) if aps else None,
                "policy_errors": sum(1 for r in g if r.get("policy_error")),
                "cost_usd": float(np.sum(costs)) if costs else None,
                "vs_reference": None if cmp is None else cmp["outcome"],
                "p_value": None if cmp is None else cmp["p_value"],
            }
    totals = {pol: {"wins": 0, "ties": 0, "losses": 0, "undecided": 0} for pol in policies if pol != reference}
    for c in comparisons:
        key = {"win": "wins", "tie": "ties", "loss": "losses"}.get(c["outcome"], "undecided")
        totals[c["policy"]][key] += 1
    return {"problems": problems, "policies": policies, "reference": reference if reference in policies else None,
            "metric": metric, "table": table, "totals": totals if comparisons else {}, "comparisons": comparisons,
            "n_runs": len(rows), "n_failed": len(all_rows) - len(rows)}


def _fmt(v: Any) -> str:
    x = _num(v)
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"


def _fmt_stats(s: Mapping[str, Any]) -> str:
    if not s or s.get("median") is None:
        return "n/a"
    return f"{_fmt(s['median'])} [{_fmt(s['q25'])}, {_fmt(s['q75'])}]"


def format_report(summary: Mapping[str, Any]) -> str:
    """Plain-text table of a :func:`summarize` result (median [25 %, 75 %] of final regret per problem)."""
    ref = summary.get("reference")
    lines = [f"Final regret: median [25%, 75%] over seeds"
             + (f"; vs {ref}: two-sided Mann-Whitney U, p < {ALPHA} (else tie)" if ref else "")]
    width = max([len(p) for p in summary["policies"]] + [6])
    for prob in summary["problems"]:
        entries = summary["table"][prob]
        any_entry = next(iter(entries.values()))
        simple_label = "HV regret" if any_entry.get("is_moo") else "simple regret"
        lines += ["", f"{prob} (budget {any_entry['budget']})",
                  f"  {'policy':<{width}}  runs  {simple_label:<31}  {'cumulative regret':<31}  time/run"
                  + (f"  vs {ref}" if ref else "")]
        for pol, e in entries.items():
            line = (f"  {pol:<{width}}  {e['n_runs']:>4}  {_fmt_stats(e['simple']):<31}  "
                    f"{_fmt_stats(e['cumulative']):<31}  {e['wallclock_s']:>6.1f}s")
            if ref:
                if pol == ref:
                    line += "  (reference)"
                elif e.get("vs_reference"):
                    line += f"  {e['vs_reference']} (p={e['p_value']:.3g})"
                else:
                    line += "  n/a"
            extras = []
            if e.get("forced_commits"):
                extras.append(f"{e['forced_commits']:.1f} forced commits/run")
            if e.get("policy_errors"):
                extras.append(f"{e['policy_errors']} policy error(s)")
            if e.get("cost_usd"):
                extras.append(f"${e['cost_usd']:.2f} API cost")
            if extras:
                line += "  [" + ", ".join(extras) + "]"
            lines.append(line)
    if summary.get("totals"):
        lines += ["", f"Totals vs {ref} ({summary['metric']} regret):"]
        for pol, t in summary["totals"].items():
            lines.append(f"  {pol:<{width}}  {t['wins']} win(s), {t['ties']} tie(s), {t['losses']} loss(es)"
                         + (f", {t['undecided']} undecided" if t["undecided"] else ""))
    if summary.get("n_failed"):
        lines += ["", f"{summary['n_failed']} run(s) failed and are not included."]
    return "\n".join(lines)


# =========================================================================== plots


def plot_regret(results: Any, out_path: str | os.PathLike, kind: str = "simple", log_scale: bool = True, *,
                problems: Sequence[str] | None = None, policies: Sequence[str] | None = None,
                title: str | None = None) -> str:
    """Regret curves: median and 25-75 % band over seeds, one panel per problem (paper §6).

    Args:
        results: rows, campaign results or a results directory.
        out_path: image file to write (format from the extension, e.g. ``.png``, ``.pdf``).
        kind: ``"simple"`` (hypervolume regret for multi-objective problems) or ``"cumulative"``.
        log_scale: logarithmic y axis (values <= 0 are drawn at a small positive floor).
        problems, policies: optional subsets (and order) to plot; at most 8 policies.
        title: optional figure title.

    Returns:
        ``out_path`` as a string. Requires matplotlib (imported lazily).
    """
    if kind not in ("simple", "cumulative"):
        raise LenzError(f"kind must be 'simple' or 'cumulative', got {kind!r}")
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LenzError("plot_regret needs matplotlib: pip install 'agentic-bo[plot]'") from exc

    rows = _ok_rows(results)
    probs = list(problems) if problems is not None else _ordered(r["problem"] for r in rows)
    pols = list(policies) if policies is not None else _ordered(r["policy"] for r in rows)
    rows = [r for r in rows if r["problem"] in probs and r["policy"] in pols]
    if not rows:
        raise LenzError("no finished runs to plot")
    probs = [p for p in probs if any(r["problem"] == p for r in rows)]
    if len(pols) > len(PALETTE):
        raise LenzError(f"at most {len(PALETTE)} policies per plot (got {len(pols)}); pass policies=[...]")
    color = {pol: PALETTE[i] for i, pol in enumerate(pols)}     # color follows the policy, not its rank
    key = f"{kind}_regret"

    ncols = min(3, len(probs))
    nrows = math.ceil(len(probs) / ncols)
    fig = Figure(figsize=(4.4 * ncols, 3.3 * nrows + (0.8 if title else 0.5)), facecolor=_SURFACE)
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    handles: dict[str, Any] = {}
    for i, prob in enumerate(probs):
        ax = axes[i // ncols][i % ncols]
        group = [r for r in rows if r["problem"] == prob]
        is_moo = any(r.get("is_moo") for r in group)
        curves = {}
        for pol in pols:
            runs = [np.asarray(r[key], dtype=float) for r in group if r["policy"] == pol and len(r[key])]
            if runs:
                s = summarize_curves(runs)
                curves[pol] = (np.asarray(s["median"]), np.asarray(s["q25"]), np.asarray(s["q75"]), len(runs))
        floor = None
        if log_scale:
            vals = np.concatenate([np.concatenate(c[:3]) for c in curves.values()]) if curves else np.zeros(0)
            pos = vals[np.isfinite(vals) & (vals > 0)]
            floor = float(pos.min()) / 2 if pos.size else 1e-12
        for pol, (med, q25, q75, n) in curves.items():
            x = np.arange(1, med.size + 1)
            if floor is not None:
                med, q25, q75 = (np.maximum(v, floor) for v in (med, q25, q75))
            ax.fill_between(x, q25, q75, color=color[pol], alpha=0.16, linewidth=0)
            (line,) = ax.plot(x, med, color=color[pol], linewidth=1.8, solid_capstyle="round")
            handles[pol] = line
        if log_scale:
            ax.set_yscale("log")
        n_seeds = max((c[3] for c in curves.values()), default=0)
        ax.set_title(f"{prob}  (n={n_seeds})", loc="left", fontsize=10, color=_INK)
        ax.set_facecolor(_SURFACE)
        ax.grid(True, axis="y", which="major", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_AXIS)
        ax.tick_params(colors=_MUTED, labelcolor=_INK_2, labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if i % ncols == 0:
            label = ("hypervolume regret" if is_moo else "simple regret") if kind == "simple" else (
                "cumulative HV regret" if is_moo else "cumulative regret")
            ax.set_ylabel(label, color=_INK_2, fontsize=9)
        if i // ncols == nrows - 1 or i + ncols >= len(probs):
            ax.set_xlabel("evaluation", color=_INK_2, fontsize=9)
    for j in range(len(probs), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    labels = [p for p in pols if p in handles]
    # header rows (inches): an optional title, then the legend, then the panels
    height = fig.get_figheight()
    y = 1 - 0.08 / height
    if title:
        fig.suptitle(title, x=0.01, y=y, ha="left", va="top", fontsize=11, color=_INK)
        y -= 0.32 / height
    fig.legend([handles[p] for p in labels], labels, loc="upper center", ncol=min(len(labels), 8), frameon=False,
               fontsize=9, labelcolor=_INK_2, bbox_to_anchor=(0.5, y))
    fig.tight_layout(rect=(0, 0, 1, y - 0.24 / height))
    out = os.fspath(out_path)
    folder = os.path.dirname(os.path.abspath(out))
    os.makedirs(folder, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    return out


# =========================================================================== benchmark runner


def _write_json(path: str | os.PathLike, obj: Any) -> None:
    """Atomic JSON write (sorted keys, no NaN)."""
    path = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(dumps(obj) + "\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "x"


def _policy_specs(policies: Sequence[Any]) -> list[dict]:
    """``[{"label", "policy" (name | Policy), "kwargs"}]`` from names, Policy instances or spec dicts."""
    specs = []
    for p in policies:
        if isinstance(p, str):
            get_policy(p)  # validate the name early
            specs.append({"label": p.strip(), "policy": p.strip(), "kwargs": {}})
        elif isinstance(p, Policy):
            specs.append({"label": p.name, "policy": p, "kwargs": {}})
        elif isinstance(p, Mapping):
            unknown = sorted(set(p) - {"name", "label", "kwargs"})
            if unknown or not isinstance(p.get("name"), str):
                raise LenzError(f"policy spec must be {{'name', 'label'?, 'kwargs'?}}, got {dict(p)!r}")
            kwargs = dict(p.get("kwargs") or {})
            get_policy(p["name"], **kwargs)   # validate name and arguments early
            specs.append({"label": str(p.get("label") or p["name"]).strip(), "policy": p["name"],
                          "kwargs": kwargs})
        else:
            raise LenzError(f"cannot interpret policy {p!r}; use a name, a Policy or {{'name', 'label', 'kwargs'}}")
    labels = [s["label"] for s in specs]
    dup = sorted({x for x in labels if labels.count(x) > 1})
    if dup:
        raise LenzError(f"duplicate policy labels {dup}; give each policy spec a distinct 'label'")
    return specs


def _problem_specs(problems: Sequence[Any]) -> list[dict]:
    specs = []
    for p in problems:
        if isinstance(p, str):
            specs.append({"label": get_problem(p).name, "problem": p})
        elif isinstance(p, Problem):
            specs.append({"label": p.name, "problem": p})
        else:
            raise LenzError(f"cannot interpret problem {p!r}; use a registry name or a Problem")
    labels = [s["label"] for s in specs]
    dup = sorted({x for x in labels if labels.count(x) > 1})
    if dup:
        raise LenzError(f"duplicate problems {dup}")
    return specs


def _seed_list(seeds: int | Iterable[int]) -> list[int]:
    if isinstance(seeds, bool):
        raise LenzError("seeds must be a count or a list of integers")
    if isinstance(seeds, (int, np.integer)):
        if seeds < 1:
            raise LenzError(f"seeds must be >= 1, got {seeds}")
        return list(range(int(seeds)))
    out = [int(s) for s in seeds]
    if not out or len(set(out)) != len(out):
        raise LenzError(f"seeds must be a non-empty list of distinct integers, got {out}")
    return out


def _run_task(task: Mapping[str, Any]) -> dict:
    """Run one benchmark task and write its JSON file (worker entry point; never raises)."""
    try:
        kwargs = dict(task["policy_kwargs"])
        if task["transcript"] and isinstance(task["policy"], str) and "transcript_path" not in kwargs:
            kwargs["transcript_path"] = task["transcript"]
        result = run_campaign(task["problem"], task["policy"], seed=task["seed"], budget=task["budget"],
                              randomize=task["randomize"], noise_std=task["noise_std"], regret=task["regret"],
                              policy_kwargs=kwargs or None, **task["campaign_options"])
        result.policy = task["policy_label"]
        _write_json(task["path"], {**result.to_dict(), "run_spec": task["spec"]})
        return result_row(result, task["path"])
    except Exception as exc:  # noqa: BLE001 - one failed run must not lose the others
        return {"problem": task["problem_label"], "policy": task["policy_label"], "seed": task["seed"],
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "path": None}


@contextlib.contextmanager
def _single_threaded_blas() -> Iterator[None]:
    """Worker processes inherit single-threaded BLAS settings (avoids oversubscribing the cores)."""
    saved = {k: os.environ.get(k) for k in _BLAS_VARS}
    for k in _BLAS_VARS:
        os.environ.setdefault(k, "1")
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_benchmark(problems: Sequence[str | Problem], policies: Sequence[Any], seeds: int | Iterable[int],
                  out_dir: str | os.PathLike, budget: int | None = None, n_jobs: int = 1, randomize: bool = True,
                  noise_std: float = 0.0, *, regret: str = "simple", reference: str = "fixed_bo",
                  campaign_options: Mapping[str, Any] | None = None, verbose: bool = False) -> list[dict]:
    """Run every problem x policy x seed campaign, with per-run JSON files and a summary.

    Args:
        problems: registry names or :class:`Problem` objects (picklable for ``n_jobs > 1``).
        policies: names, :class:`Policy` instances or ``{"name", "label"?, "kwargs"?}``
            specs (the label names the policy in results; it must be unique).
        seeds: a count ``k`` (seeds ``0..k-1``) or an explicit list of seeds.
        out_dir: output folder: ``runs/*.json`` (one per run), ``summary.json``, and
            ``transcripts/*.jsonl`` for LLM policies.
        budget: evaluations per run (``None``: each problem's default budget).
        n_jobs: worker processes for non-LLM runs (LLM runs stay sequential).
        randomize, noise_std, regret: as in :func:`run_campaign`.
        reference: policy label the comparisons in ``summary.json`` use.
        campaign_options: further :class:`~agentic_bo.sara.CampaignConfig` fields.
        verbose: print one line per finished run.

    Returns:
        One row per run (see :func:`result_row`) in problem x policy x seed order;
        runs that crashed have an ``error`` field and no file (a later call retries them).

    Raises:
        LenzError: invalid arguments, or an existing run file produced with different
        settings (use a fresh ``out_dir``).
    """
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise LenzError(f"n_jobs must be an integer >= 1, got {n_jobs!r}")
    options = dict(campaign_options or {})
    clash = sorted({"budget", "seed", "regret"} & set(options))
    if clash:
        raise LenzError(f"campaign_options must not repeat the arguments {clash}")
    prob_specs, pol_specs, seed_list = _problem_specs(problems), _policy_specs(policies), _seed_list(seeds)
    out = Path(out_dir)
    tasks: list[dict] = []
    for ps in prob_specs:
        for qs in pol_specs:
            llm = _is_llm(qs["policy"])
            for seed in seed_list:
                stem = f"{_safe(ps['label'])}__{_safe(qs['label'])}__seed{seed}"
                spec = jsonable({
                    "problem": ps["label"], "policy": qs["label"],
                    "policy_name": qs["policy"] if isinstance(qs["policy"], str) else repr(qs["policy"]),
                    "policy_kwargs": {k: v if isinstance(v, (str, int, float, bool, type(None), list, dict))
                                      else repr(v) for k, v in qs["kwargs"].items()},
                    "seed": seed, "budget": budget, "randomize": bool(randomize), "noise_std": float(noise_std),
                    "regret": regret, "campaign_options": {k: repr(v) if callable(v) else v
                                                           for k, v in options.items()},
                })
                tasks.append({
                    "problem": ps["problem"], "problem_label": ps["label"], "policy": qs["policy"],
                    "policy_label": qs["label"], "policy_kwargs": qs["kwargs"], "seed": seed, "budget": budget,
                    "randomize": bool(randomize), "noise_std": float(noise_std), "regret": regret,
                    "campaign_options": options, "path": str(out / "runs" / f"{stem}.json"), "spec": spec,
                    "transcript": str(out / "transcripts" / f"{stem}.jsonl") if llm else None, "llm": llm,
                })

    rows: list[dict | None] = [None] * len(tasks)
    todo: list[int] = []
    for i, task in enumerate(tasks):
        if not os.path.exists(task["path"]):
            todo.append(i)
            continue
        try:
            doc = _read_json(task["path"])
            stored = doc.get("run_spec")
            row = _row_from_dict(doc, task["path"])
        except (OSError, ValueError) as exc:
            if verbose:
                print(f"[benchmark] re-running {task['path']} (unreadable: {exc})", flush=True)
            todo.append(i)
            continue
        if stored is not None and dumps(stored) != dumps(task["spec"]):
            raise LenzError(f"{task['path']} was produced with different settings ({stored}); use a fresh "
                            f"out_dir or delete the old runs")
        rows[i] = row

    n_total, done = len(tasks), len(tasks) - len(todo)
    started = time.perf_counter()

    def report(i: int, row: dict) -> None:
        nonlocal done
        rows[i] = row
        done += 1
        if verbose:
            t = tasks[i]
            head = f"[benchmark] {done}/{n_total} {t['problem_label']} | {t['policy_label']} | seed {t['seed']}"
            if row.get("error"):
                print(f"{head}: FAILED {row['error']}", flush=True)
            else:
                print(f"{head}: simple regret {_fmt(row['final_simple_regret'])}, cumulative "
                      f"{_fmt(row['final_cumulative_regret'])} ({row['wallclock_s']:.1f} s)", flush=True)

    if verbose and done:
        print(f"[benchmark] {done}/{n_total} runs loaded from {os.fspath(out / 'runs')}", flush=True)
    pool_ids = [i for i in todo if not tasks[i]["llm"]]
    serial_ids = [i for i in todo if tasks[i]["llm"]]
    if n_jobs > 1 and len(pool_ids) > 1:
        import multiprocessing

        with _single_threaded_blas(), ProcessPoolExecutor(
                max_workers=min(n_jobs, len(pool_ids)), mp_context=multiprocessing.get_context("spawn")) as ex:
            futures = {ex.submit(_run_task, tasks[i]): i for i in pool_ids}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001 - e.g. an unpicklable problem or a dead worker
                    row = {"problem": tasks[i]["problem_label"], "policy": tasks[i]["policy_label"],
                           "seed": tasks[i]["seed"], "error": f"{type(exc).__name__}: {exc}", "path": None}
                report(i, row)
    else:
        serial_ids = pool_ids + serial_ids
    for i in serial_ids:
        report(i, _run_task(tasks[i]))

    final = [r for r in rows if r is not None]
    ok = [r for r in final if not r.get("error")]
    failed = [{k: r.get(k) for k in ("problem", "policy", "seed", "error")} for r in final if r.get("error")]
    summary = summarize(ok, reference=reference)
    _write_json(out / "summary.json", {
        "format": SUMMARY_FORMAT, "version": 1,
        "settings": {"problems": [s["label"] for s in prob_specs], "policies": [s["label"] for s in pol_specs],
                     "seeds": seed_list, "budget": budget, "randomize": bool(randomize),
                     "noise_std": float(noise_std), "regret": regret, "reference": reference},
        "runs": [{k: v for k, v in r.items() if k not in ("simple_regret", "cumulative_regret", "best_config")}
                 for r in ok],
        "failed": failed, "summary": summary,
    })
    if verbose:
        print(f"[benchmark] finished {len(ok)}/{n_total} runs in {time.perf_counter() - started:.1f} s"
              + (f"; {len(failed)} failed" if failed else "") + f"; summary: {os.fspath(out / 'summary.json')}",
              flush=True)
    return final


# =========================================================================== CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agentic_bo.experiment",
                                     description="Benchmark agentic BO policies (multi-seed runs, statistics, plots).")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = sub.add_parser("run", help="run a benchmark (resumes from existing run files)")
    run.add_argument("--problems", nargs="+", required=True, help="registry names, e.g. branin hartmann6 ackley10")
    run.add_argument("--policies", nargs="+", default=["heuristic", "fixed_bo", "sobol"],
                     help=f"policy names ({', '.join(POLICY_NAMES)}); default: heuristic fixed_bo sobol")
    run.add_argument("--seeds", type=int, default=5, help="number of seeds (default 5)")
    run.add_argument("--seed-start", type=int, default=0, help="first seed (default 0)")
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--budget", type=int, default=None, help="evaluations per run (default: problem default)")
    run.add_argument("--n-jobs", type=int, default=1, help="worker processes for non-LLM runs")
    run.add_argument("--no-randomize", action="store_true", help="use the original (non-anonymized) problems")
    run.add_argument("--noise-std", type=float, default=0.0, help="observation noise standard deviation")
    run.add_argument("--regret", choices=("simple", "cumulative"), default="simple")
    run.add_argument("--policy-kwargs", default=None,
                     help='JSON {policy: {kwarg: value}}, e.g. \'{"fixed_bo": {"n_init": 10}}\'')
    run.add_argument("--reference", default="fixed_bo", help="reference policy of the comparisons")
    run.add_argument("--no-plot", action="store_true", help="do not write regret plots")
    run.add_argument("--quiet", action="store_true", help="no per-run progress lines")

    rep = sub.add_parser("report", help="summarize and plot the runs in a results directory")
    rep.add_argument("--results", required=True,
                     help="benchmark output directory, a folder of run files, or one run file")
    rep.add_argument("--reference", default="fixed_bo")
    rep.add_argument("--metric", choices=("simple", "cumulative"), default=None)
    rep.add_argument("--no-plot", action="store_true")
    rep.add_argument("--json", action="store_true", help="print the summary as JSON instead of a table")

    sub.add_parser("list", help="list registered problems and policies")
    return parser


def _plots(rows: list[dict], folder: Path) -> list[str]:
    paths = []
    try:
        paths.append(plot_regret(rows, folder / "regret_simple.png", kind="simple", log_scale=True))
        paths.append(plot_regret(rows, folder / "regret_cumulative.png", kind="cumulative", log_scale=False))
    except LenzError as exc:
        print(f"[benchmark] plots skipped: {exc}", file=sys.stderr)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``run`` | ``report`` | ``list``); returns the exit code."""
    ns = _build_parser().parse_args(argv)
    try:
        if ns.command == "list":
            print("problems: " + " ".join(list_problems()))
            print("          (families take any dimension: ackley{d}, levy{d}, rosenbrock{d}, styblinski_tang{d}, "
                  "rastrigin{d}, griewank{d})")
            print("policies: " + " ".join(POLICY_NAMES))
            return 0
        if ns.command == "run":
            kw = json.loads(ns.policy_kwargs) if ns.policy_kwargs else {}
            if not isinstance(kw, dict):
                raise LenzError("--policy-kwargs must be a JSON object {policy: {kwarg: value}}")
            unknown = sorted(set(kw) - set(ns.policies))
            if unknown:
                raise LenzError(f"--policy-kwargs names policies that are not run: {unknown}")
            pols = [{"name": p, "kwargs": kw[p]} if p in kw else p for p in ns.policies]
            seeds = list(range(ns.seed_start, ns.seed_start + ns.seeds))
            rows = run_benchmark(ns.problems, pols, seeds, ns.out, budget=ns.budget, n_jobs=ns.n_jobs,
                                 randomize=not ns.no_randomize, noise_std=ns.noise_std, regret=ns.regret,
                                 reference=ns.reference, verbose=not ns.quiet)
            ok = [r for r in rows if not r.get("error")]
            print()
            print(format_report(summarize(ok, reference=ns.reference)))
            if ok and not ns.no_plot:
                for p in _plots(ok, Path(ns.out)):
                    print(f"wrote {p}")
            return 0 if len(ok) == len(rows) else 1
        rows = load_results(ns.results)
        ok = [r for r in rows if not r.get("error")]
        if not ok:
            raise LenzError(f"no finished runs in {ns.results!r}")
        summary = summarize(ok, reference=ns.reference, metric=ns.metric)
        print(dumps(summary, indent=2) if ns.json else format_report(summary))
        if not ns.no_plot:
            results = Path(ns.results)      # a directory, or a single run file (plots go next to it)
            for p in _plots(ok, results if results.is_dir() else results.parent):
                print(f"wrote {p}", file=sys.stderr if ns.json else sys.stdout)
        return 0
    except (LenzError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    # Run through the importable module so that worker processes pickle
    # ``agentic_bo.experiment._run_task`` rather than a function of ``__main__``.
    from agentic_bo.experiment import main as _main

    raise SystemExit(_main())
