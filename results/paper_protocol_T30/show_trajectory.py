"""Print an LLM agent's trajectory step by step: its reasoning, tool calls and evaluations.

Usage (from the repo root):
    python results/paper_protocol_T30/show_trajectory.py results/paper_protocol_T30/qwen/runs/hartmann6__qwen_sara__seed1.json
    python results/paper_protocol_T30/show_trajectory.py <run.json> --step 5      # one step only
    python results/paper_protocol_T30/show_trajectory.py <run.json> --full        # untruncated text
"""
import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="result JSON of one campaign")
    ap.add_argument("--step", type=int, default=None, help="show only this step")
    ap.add_argument("--full", action="store_true", help="do not truncate long reasoning")
    a = ap.parse_args()
    r = json.load(open(a.run))
    print(f"{r['problem']} | policy {r['policy']} | seed {r['seed']} | budget {r['budget']} | "
          f"final simple regret {r['simple_regret'][-1]:.4g}\n")
    last_step = None
    for e in r["log"]:
        step = e.get("step")
        if a.step is not None and step != a.step:
            continue
        if step != last_step:
            print(f"\n{'=' * 30} step {step} {'=' * 30}")
            last_step = step
        if e.get("action_class") == "reasoning":
            text = e.get("text", "").strip()
            if not a.full and len(text) > 1500:
                text = text[:1500] + f" ... [{len(text) - 1500} more chars, use --full]"
            print(f"\n[{e.get('kind')}]\n{text}")
        else:
            args = json.dumps(e.get("args"), ensure_ascii=False)
            status = "ok" if e.get("ok") else f"ERROR: {e.get('error')}"
            line = f"\n> {e.get('tool')}({args[:300]}) -> {status}"
            if e.get("tool") == "evaluate" and e.get("ok"):
                res = e.get("result") or {}
                line += f"\n  observed {res.get('observed')}  new best: {res.get('is_new_best')}"
            print(line)


if __name__ == "__main__":
    main()
