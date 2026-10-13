"""Run Sara (Claude Opus 5.5 driving lenz) on a benchmark problem and report regret, tokens and cost.

This calls the Anthropic API and costs money (roughly a few US dollars for the
default budget of 20 at effort "high"; the script prints the actual usage). It needs
the optional SDK and credentials:

    pip install -e '.[llm]'
    export ANTHROPIC_API_KEY=...          # or ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile
    python examples/run_sara_claude.py                       # randomized branin, budget 20
    python examples/run_sara_claude.py --problem hartmann6 --budget 40 --effort medium
    python examples/run_sara_claude.py --toolset llm_only    # LLM-as-optimizer ablation (no lenz tools)

Every request and response is appended to a JSONL transcript, and the full campaign
result (records, tool log with Claude's reasoning summaries, notebook) is saved as JSON.
A FixedBO run with the same seed is printed for reference (free, no API calls).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import agentic_bo  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_bo.experiment import prepare_problem, run_campaign
from agentic_bo.sara.policies.llm import DEFAULT_MODEL, EFFORT_LEVELS, INSTALL_HINT, ClaudeSaraPolicy


#: client attributes through which the SDK resolves authentication (any one set is enough)
_CREDENTIAL_ATTRS = ("api_key", "auth_token", "credentials", "_token_cache")


def make_client():
    """An Anthropic client, or ``None`` after printing why it cannot be created."""
    try:
        import anthropic
    except ImportError:
        print(f"error: the 'anthropic' package is not installed. Install it with:\n  {INSTALL_HINT}",
              file=sys.stderr)
        return None
    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 - typically: no credentials configured
        print("error: could not create an Anthropic client "
              f"({type(exc).__name__}: {exc}).\nSet ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN), or log in with "
              "`ant auth login`, then run this script again. No API calls were made.", file=sys.stderr)
        return None
    # Recent SDKs construct a client without credentials and fail only at the first request;
    # refuse here instead of starting a campaign that would end in harness fallbacks.
    if not any(getattr(client, a, None) for a in _CREDENTIAL_ATTRS):
        print("error: no Anthropic credentials found.\nSet ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN), or log in "
              "with `ant auth login`, then run this script again. No API calls were made.", file=sys.stderr)
        return None
    return client


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Claude Sara on a benchmark problem.")
    parser.add_argument("--problem", default="branin", help="registry name (default: branin)")
    parser.add_argument("--budget", type=int, default=20, help="evaluations (default: 20)")
    parser.add_argument("--effort", default="high", choices=EFFORT_LEVELS, help="output_config.effort")
    parser.add_argument("--seed", type=int, default=0, help="seed (also selects the randomized instance)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--toolset", default="full", choices=("full", "llm_only"))
    parser.add_argument("--regret", default="simple", choices=("simple", "cumulative"))
    parser.add_argument("--no-randomize", action="store_true", help="show the original problem (not anonymized)")
    parser.add_argument("--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    parser.add_argument("--no-compaction", action="store_true", help="disable server-side compaction")
    parser.add_argument("--no-baseline", action="store_true", help="skip the FixedBO reference run")
    parser.add_argument("--out", default="results/sara", help="folder for the transcript and result JSON")
    parser.add_argument("--verbose", action="store_true", help="print Claude's reasoning and tool calls")
    args = parser.parse_args()

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("note: neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set; trying other credential sources.",
              file=sys.stderr)
    client = make_client()
    if client is None:
        return 2

    randomize = not args.no_randomize
    problem = prepare_problem(args.problem, seed=args.seed, randomize=randomize)
    budget = args.budget or problem.default_budget
    stem = f"{problem.name}_{args.toolset}_seed{args.seed}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    transcript = out / f"{stem}.jsonl"
    policy = ClaudeSaraPolicy(model=args.model, effort=args.effort, client=client, toolset=args.toolset,
                              use_fallbacks=not args.no_fallbacks, use_compaction=not args.no_compaction,
                              transcript_path=transcript, verbose=args.verbose)
    print(f"Sara ({args.model}, effort {args.effort}, toolset {args.toolset}) on {problem.name}"
          f"{' (randomized)' if randomize and problem.meta.get('randomized') else ''}: budget {budget}, "
          f"{args.regret} regret. This makes real API calls.")

    result = run_campaign(problem, policy, seed=args.seed, budget=budget, regret=args.regret)
    result.save(out / f"{stem}.json")

    u = result.usage
    print()
    if result.policy_error:
        print(f"policy error: {result.policy_error}")
        print("(the harness finished the remaining steps with fallback evaluations)")
    print(f"best value     : {result.best_value}   at {result.best_config}")
    print(f"simple regret  : {result.final_simple_regret:.4g}")
    print(f"cumulative     : {result.final_cumulative_regret:.4g}")
    print(f"forced commits : {result.forced_commits} of {result.n_evals} evaluations")
    print(f"actions/step   : {sum(result.actions_per_step) / max(len(result.actions_per_step), 1):.1f} "
          f"(by tool: {result.action_counts})")
    print(f"API calls      : {u.get('api_calls', 0)}   stop reasons: {u.get('stop_reasons', {})}")
    print(f"tokens         : input {u.get('input_tokens', 0):,} (uncached), output {u.get('output_tokens', 0):,}, "
          f"cache read {u.get('cache_read_input_tokens', 0):,}, cache write "
          f"{u.get('cache_creation_input_tokens', 0):,}")
    print(f"estimated cost : ${u.get('cost_usd', 0.0):.2f}   (compactions: {u.get('compactions', 0)}, "
          f"fallback responses: {u.get('fallback_responses', 0)})")
    print(f"wall clock     : {result.wallclock_s:.0f} s")
    print(f"saved          : {out / (stem + '.json')} and {transcript}")

    if not args.no_baseline:
        ref = run_campaign(problem, "fixed_bo", seed=args.seed, budget=budget, regret=args.regret)
        print(f"\nFixedBO reference (same seed): simple regret {ref.final_simple_regret:.4g}, "
              f"cumulative {ref.final_cumulative_regret:.4g}")
    return 1 if result.policy_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
