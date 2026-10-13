#!/bin/bash
# Final benchmark following the paper's protocol (App. A): randomized problems, 10 seeds, paper budgets.
cd "$(dirname "$0")/.."
PY=${PY:-python3}
$PY -m agentic_bo.experiment run \
  --problems branin hartmann6 ackley10 ackley20 constrained_hartmann6 levy10 styblinski_tang8 reaction_yield branin_currin \
  --policies heuristic fixed_bo sobol random --seeds 10 --n-jobs 8 --reference fixed_bo \
  --out results/paper_protocol --quiet
$PY -m agentic_bo.experiment run \
  --problems branin hartmann6 --policies heuristic fixed_bo sobol --seeds 10 --n-jobs 8 \
  --regret cumulative --reference fixed_bo --out results/cumulative --quiet
echo ALL_DONE
