"""Run all missing Sara-Qwen jobs (budget 30) with N parallel subprocesses. Resumable: finished runs are skipped.

Usage (from the repo root):  DLS_2026_API_KEY=... python results/paper_protocol_T30/qwen_driver.py 8
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "qwen")
PROBS = ["branin", "hartmann6", "ackley10", "ackley20", "constrained_hartmann6", "levy10", "styblinski_tang8",
         "reaction_yield", "branin_currin"]
REUSE = {("branin", s) for s in range(5)}      # already run at T=30 (results/sara_qwen/branin_T30_seed*.json)
jobs = [(p, s) for s in range(10) for p in PROBS if (p, s) not in REUSE
        and not os.path.exists(f"{OUT}/runs/{p}__qwen_sara__seed{s}.json")]
print(f"{len(jobs)} jobs to run", flush=True)

def run(job):
    p, s = job
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "qwen_job.py"), p, str(s), "30", OUT],
                       capture_output=True, text=True)
    done = [l for l in r.stdout.splitlines() if l.startswith("DONE")]
    print(done[-1] if done else f"FAILED {p} seed{s} rc={r.returncode}: {r.stderr.strip().splitlines()[-1:]}", flush=True)

with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
    list(ex.map(run, jobs))
print("ALL_DONE", flush=True)
