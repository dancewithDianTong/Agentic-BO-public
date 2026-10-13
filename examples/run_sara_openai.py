"""Run Sara on an OpenAI-compatible chat model: the OpenAI API or a self-hosted server.

OpenAI (needs OPENAI_API_KEY):

    python examples/run_sara_openai.py --model <openai-model-id> --budget 20

A self-hosted Qwen model behind vLLM (tool calling must be enabled on the server, e.g.
``vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes``):

    python examples/run_sara_openai.py --model Qwen/Qwen3-32B --base-url http://localhost:8000/v1

Ollama serves the same API at http://localhost:11434/v1 (raise its context length:
the Sara prompts alone are ~8k tokens).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_bo.experiment import run_campaign  # noqa: E402
from agentic_bo.sara.policies.openai_llm import OpenAISaraPolicy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Sara on an OpenAI-compatible chat model.")
    ap.add_argument("--model", required=True, help="model name on the endpoint")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default: the OpenAI API)")
    ap.add_argument("--problem", default="branin")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reasoning-effort", default=None, help="sent as reasoning_effort (reasoning models only)")
    ap.add_argument("--toolset", default="full", choices=("full", "llm_only"))
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Qwen3 on vLLM/SGLang: chat_template_kwargs.enable_thinking=true")
    ap.add_argument("--out", default="results/sara_openai")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.problem}_seed{a.seed}"
    policy = OpenAISaraPolicy(
        model=a.model, base_url=a.base_url, reasoning_effort=a.reasoning_effort, toolset=a.toolset,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}} if a.enable_thinking else None,
        transcript_path=out / f"{tag}.transcript.jsonl", verbose=a.verbose)
    res = run_campaign(a.problem, policy, seed=a.seed, budget=a.budget)
    res.save(out / f"{tag}.json")

    if res.policy_error:
        print(f"policy error: {res.policy_error}", file=sys.stderr)
    usage = res.usage or {}
    print(f"{a.problem}: {res.n_evals} evaluations, final simple regret {res.final_simple_regret:.3g}, "
          f"{res.forced_commits} harness fallbacks")
    print("usage:", json.dumps({k: usage.get(k) for k in ("api_calls", "input_tokens", "cache_read_input_tokens",
                                                           "output_tokens", "reasoning_tokens", "nudges",
                                                           "invalid_arguments")}))
    print(f"result and transcript in {out}/")
    return 1 if res.policy_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
