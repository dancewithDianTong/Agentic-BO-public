Summarize this optimization campaign so far so that it can continue from the summary alone. The full trial log stays available through the `trials` tool, so do not copy every trial; keep what is needed to make the next decisions well. Use exact numbers and exact configs where you keep them, and do not invent values that are not in the conversation.

The summary must retain:

1. Task and goal: what is optimized and in which direction, the regret mode (simple or cumulative), and the constraints.
2. Instructions in force: every instruction or new piece of context received during the campaign (operator system messages, user context), when it arrived (after how many evaluations), and what it changed. Mark instructions that were later superseded.
3. Budget: evaluations used and remaining, and the current step.
4. Current lenz configuration: objectives, constraints, acquisition function and its parameters, active region (bounds), and model settings, with the reasons for the latest changes.
5. Incumbent: trial id, exact config and observed metrics; for several objectives, the Pareto set and hypervolume. Also the few next-best trials when they lie in different regions.
6. Landscape hypotheses: what the data suggests about the shape of the function (promising regions, boundary optima, several basins, noise), which hypotheses were confirmed and which were refuted.
7. Sensitive parameters: which inputs matter most and in which direction values improve, with the supporting evidence (lengthscales, sensitivities, controlled comparisons).
8. Model trust: the latest diagnostics (cv_r2 per metric and its trend) and how well recent predictions or expectations matched observations.
9. Plan: the current strategy, the next intended actions, and any open questions to test.
10. Lab-notebook notes that are still relevant.

Keep the summary compact and factual, organized under these headings. Write it as text only, with no tool call.
