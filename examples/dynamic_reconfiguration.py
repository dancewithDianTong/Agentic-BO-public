"""Dynamic reconfiguration: the experimenter changes the goal in the middle of a campaign.

The problem is the reaction-yield toy model (a continuous-flow Suzuki coupling with six
process conditions, a natural-language prior, budget 40). Before step 15 the
experimenter adds an instruction: the cost index of a run must now stay at or below
3.0. The unconstrained optimum (about 94 % yield) costs about 6.7, so the agent has to
move to a different region (hotter, shorter residence time, less palladium), where
about 92 % is still possible.

Sara reconfigures lenz instead of starting over: ``set_constraints`` keeps every trial
(the cost of all earlier runs is already known), so the constraint model starts with
14 observations. The scripted agent reads the instruction's structured payload
(``data = {"op": "set_constraints", ...}``); Claude reads its text.

    python examples/dynamic_reconfiguration.py                 # scripted Sara agent (free)
    python examples/dynamic_reconfiguration.py --llm           # Claude (needs anthropic + credentials)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import agentic_bo  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_bo.experiment import run_campaign
from agentic_bo.problems import get_problem
from agentic_bo.space import sobol_points


def instruction(limit: float) -> dict:
    """The mid-campaign instruction: text for an LLM agent, structured data for the scripted one."""
    return {
        "source": "instructor",
        "text": (f"New requirement from the process team: from now on every run must have a cost index of at most "
                 f"{limit:g} (metric `cost`, lower is cheaper). Runs above that limit are not acceptable for "
                 f"scale-up, however high their yield. Keep maximizing yield under this limit; all earlier runs "
                 f"remain valid data."),
        "data": {"op": "set_constraints", "args": {"constraints": [{"metric": "cost", "upper": limit}]}},
    }


def constrained_optimum(limit: float, n: int = 2 ** 14, n_starts: int = 12) -> tuple[float, dict]:
    """Reference: best yield with cost <= limit (dense Sobol search, then SLSQP polishing).

    The cost limit is active at the constrained optimum, so a raw Sobol maximum lies a few
    points below it; the best feasible designs are refined with the constraint enforced.
    """
    from scipy.optimize import minimize

    problem = get_problem("reaction_yield")
    lo = np.array([p.lower for p in problem.space.params])
    hi = np.array([p.upper for p in problem.space.params])
    log = np.array([p.log_scale for p in problem.space.params])

    def to_raw(U: np.ndarray) -> np.ndarray:      # encoded unit cube -> raw (log axes for log params)
        U = np.clip(np.atleast_2d(U), 0.0, 1.0)
        a = np.where(log, np.log(np.where(log, lo, 1.0)), lo)
        b = np.where(log, np.log(np.where(log, hi, 1.0)), hi)
        Z = a + U * (b - a)
        return np.where(log, np.exp(Z), Z)

    def metrics(u: np.ndarray) -> dict:
        return problem.evaluate_array(to_raw(u))

    U = sobol_points(n, problem.space.d, 0)
    m = metrics(U)
    score = np.where(m["cost"] <= limit, m["yield"], -np.inf)
    best_v, best_x = float(score.max()), to_raw(U[int(np.argmax(score))])[0]
    for i in np.argsort(-score)[:n_starts]:
        if not np.isfinite(score[i]):
            break
        res = minimize(lambda u: -metrics(u)["yield"][0], U[i], method="SLSQP", bounds=[(0.0, 1.0)] * len(lo),
                       constraints=[{"type": "ineq", "fun": lambda u: limit - metrics(u)["cost"][0]}],
                       options={"maxiter": 500, "ftol": 1e-10})
        x = to_raw(res.x)
        mm = problem.evaluate_array(x)
        if mm["cost"][0] <= limit and mm["yield"][0] > best_v:
            best_v, best_x = float(mm["yield"][0]), x[0]
    return best_v, dict(zip(problem.space.names, best_x.tolist()))


def make_llm_policy(effort: str):
    from agentic_bo.sara.policies.llm import INSTALL_HINT, ClaudeSaraPolicy
    try:
        import anthropic
    except ImportError:
        sys.exit(f"error: --llm needs the 'anthropic' package: {INSTALL_HINT}")
    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 - typically: no credentials configured
        sys.exit(f"error: could not create an Anthropic client ({type(exc).__name__}: {exc}); set "
                 f"ANTHROPIC_API_KEY or run `ant auth login`. No API calls were made.")
    # recent SDKs build a client without credentials and fail only at the first request
    if not any(getattr(client, a, None) for a in ("api_key", "auth_token", "credentials", "_token_cache")):
        sys.exit("error: no Anthropic credentials found; set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) or run "
                 "`ant auth login`. No API calls were made.")
    return ClaudeSaraPolicy(client=client, effort=effort)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mid-campaign cost constraint on the reaction-yield problem.")
    parser.add_argument("--llm", action="store_true", help="use Claude (Sara) instead of the scripted agent")
    parser.add_argument("--effort", default="high", help="Claude effort (with --llm)")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--at-step", type=int, default=15, help="the instruction arrives before this step")
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    policy = make_llm_policy(args.effort) if args.llm else "heuristic"
    t_inject = args.at_step - 1          # context_schedule keys count completed evaluations
    print(f"Running {'Claude Sara' if args.llm else 'the scripted Sara agent'} on reaction_yield "
          f"(budget {args.budget}); the cost limit {args.cost_limit:g} arrives before step {args.at_step}.")
    result = run_campaign("reaction_yield", policy, seed=args.seed, budget=args.budget, randomize=False,
                          context_schedule={t_inject: instruction(args.cost_limit)})

    recs = result.records
    before, after = recs[:t_inject], recs[t_inject:]

    def best(rows: list[dict], limit: float | None = None) -> tuple[float | None, dict | None]:
        ok = [r for r in rows if r["true_metrics"].get("yield") is not None
              and (limit is None or r["true_metrics"]["cost"] <= limit)]
        if not ok:
            return None, None
        r = max(ok, key=lambda r: r["true_metrics"]["yield"])
        return r["true_metrics"]["yield"], r

    y0, r0 = best(before)
    y1, r1 = best(after, args.cost_limit)
    n_ok = sum(r["true_metrics"].get("cost", np.inf) <= args.cost_limit for r in after)
    ref, ref_cfg = constrained_optimum(args.cost_limit)

    print()
    if r0 is not None:
        print(f"before the instruction ({len(before)} runs): best yield {y0:.1f} % at cost "
              f"{r0['true_metrics']['cost']:.2f}")
    reconf = [e for e in result.log if e.get("tool") == "set_constraints"]
    for e in reconf:
        print(f"step {e['step']}: the agent called set_constraints({e['args']}) -> "
              f"{'ok' if e['ok'] else e.get('error')}")
    if not reconf:
        print("the agent did not call set_constraints")
    print(f"after the instruction ({len(after)} runs): {n_ok} within the cost limit; best feasible yield "
          + (f"{y1:.1f} % at cost {r1['true_metrics']['cost']:.2f}, config {r1['config']}" if r1 else "none"))
    print(f"reference: best yield with cost <= {args.cost_limit:g} is about {ref:.1f} % (dense search + local "
          f"polishing), at {({k: round(v, 3) for k, v in ref_cfg.items()})}")
    notes = [n for n in result.notebook if n["t"] >= t_inject]
    if notes:
        print(f"\nnotebook entry at t={notes[0]['t']}: {notes[0]['text'][:400]}")
    if result.usage.get("api_calls"):
        print(f"\nAPI calls {result.usage['api_calls']}, estimated cost ${result.usage.get('cost_usd', 0):.2f}")


if __name__ == "__main__":
    main()
