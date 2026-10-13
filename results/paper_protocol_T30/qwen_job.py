"""One Sara-Qwen campaign: python qwen_job.py <problem> <seed> <budget> <outdir>."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from agentic_bo.experiment import run_campaign
from agentic_bo.sara.policies.openai_llm import OpenAISaraPolicy
problem, seed, budget, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(f"{out}/runs", exist_ok=True); os.makedirs(f"{out}/transcripts", exist_ok=True)
tag = f"{problem}__qwen_sara__seed{seed}"
pol = OpenAISaraPolicy(model="qwen3.8-27b", base_url=os.environ.get("DLS_2026_BASE_URL", "http://209.182.219.81:4010/v1"), name="qwen_sara",
                       api_key=os.environ["DLS_2026_API_KEY"], max_completion_tokens=16384,
                       transcript_path=f"{out}/transcripts/{tag}.jsonl")
t0 = time.time()
res = run_campaign(problem, pol, seed=seed, budget=budget)
res.save(f"{out}/runs/{tag}.json.tmp"); os.replace(f"{out}/runs/{tag}.json.tmp", f"{out}/runs/{tag}.json")
print(f"DONE {tag} evals={res.n_evals} forced={res.forced_commits} regret={res.final_simple_regret:.4g} "
      f"err={res.policy_error!r} wall={time.time()-t0:.0f}s", flush=True)
