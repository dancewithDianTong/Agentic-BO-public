"""The scripted Sara agent (DESIGN.md §9.3): agentic BO without an LLM.

:class:`HeuristicSaraPolicy` follows Sara's directives (paper §5.2) with exactly the
tools an LLM agent gets: it acts only through :meth:`AgentSession.call` and reads only
what an agent may read (tool results, the brief, the context, the campaign settings).
Its traces are therefore comparable with Claude's, and it is deterministic given the
campaign seed. A campaign runs through these phases:

* **Opening (adaptive).** A structured prior config (``prior_hint.config``) is
  evaluated first. A prior region (``prior_hint.region``) becomes the active region
  (``set_bounds``) and the opening fills it. Otherwise the opening is a short Sobol
  design. Under simple regret it is extended, up to ``max_init`` points, while the
  surrogate is not predictive (low cross-validated R^2). Under cumulative regret it is
  shorter and never extended.
* **Trust calibration.** Every model step starts with ``diagnostics`` (CV R^2). A
  trusted surrogate (CV R^2 >= ``trust_threshold``) may move the search globally. A
  surrogate distrusted for ``distrust_patience`` steps in a row stops steering
  exclusively: every few steps the policy explores instead of refining locally.
* **Global vs local.** Each step proposes a local candidate (``suggest`` in a trust
  region around the incumbent) and, when trusted, a global one (``suggest`` over the
  active region). ``score`` puts both, plus the incumbent, on a common scale. The global
  candidate wins only if the posterior mean predicts an improvement there, or (simple
  regret) its expected improvement is ``global_ei_ratio`` times larger. Failed global
  moves make the next one wait longer (credit assignment).
* **Trust region (TuRBO-style).** The radius doubles after ``success_tol`` successive
  improvements and halves after ``failure_tol`` successive failures. An improvement
  counts when it exceeds TuRBO's ``1e-3 |y|``, capped at ``1e-3`` of the observed range
  (an additive offset of the objective must not hide real progress). Under simple regret
  a radius below ``radius_min`` restarts the region from a global candidate.
* **Stagnation.** After ``stagnation_steps`` evaluations without improvement, a prior
  region is widened to the full domain. Otherwise the step evaluates one exploratory UCB
  candidate with a large ``beta`` (``set_acqf``, ``suggest``, then restore the
  acquisition).
* **Region checks.** A prior region is widened when the incumbent sits at one of its
  interior faces, or when the posterior mean predicts a better value outside it.
* **End game.** The last ``endgame_frac`` of the budget only exploits: local candidates
  around the incumbent with a radius of at most ``endgame_radius``.
* **Cumulative regret.** Every evaluation counts, so: a short opening, UCB with a small
  ``beta``, a smaller trust region, no restarts, no stagnation or distrust exploration.
  Global moves happen only when the posterior mean predicts an improvement elsewhere.
* **Noise.** When the brief states an observation noise level, logEI becomes
  noisy_logei and improvements smaller than the noise do not count as successes.
  These regime acquisitions are re-applied when an objectives change resets the
  acquisition (e.g. back from multi-objective to single-objective).
* **Context.** Structured instructions (``ContextItem.data = {"op", "args"}``) are
  applied with the matching reconfigure tool (``set_constraints``, ``set_objectives``,
  ``set_bounds``, ``set_acqf``). An instructed acquisition or region is kept and never
  overridden. ``{"op": "evaluate", "args": {"config"}}`` queues a targeted evaluation,
  whose outcome is first predicted with ``predict``. Plain-text context, malformed
  instructions and instructions that could not be applied (or evaluated) are
  acknowledged with ``record_note``.

Every evaluation carries a rationale (the kind of step, why, how far the surrogate is
trusted) and an expectation (the predicted outcome). Every ``note_every`` evaluations the
policy writes a ``record_note`` summary of its beliefs. Duplicates of evaluated trials
are not committed when an alternative exists. Optional actions are skipped when they
would exceed the soft action limit; a typical step takes 3-5 actions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from ...lenz.acquisition import DEFAULT_UCB_BETA, POINTWISE_ACQFS
from ...space import CategoricalParam, IntParam, OrdinalParam, RealParam, SearchSpace
from ...utils import LenzError
from .base import Policy, PolicyError

if TYPE_CHECKING:  # pragma: no cover
    from ..session import AgentSession, ContextItem

__all__ = ["HeuristicSaraPolicy", "INSTRUCTION_OPS", "space_from_tools", "declared_noise_std"]

#: structured instruction ops (``ContextItem.data["op"]``) the policy applies
INSTRUCTION_OPS = ("set_constraints", "set_objectives", "set_bounds", "set_acqf", "evaluate", "record_note")
_RECONFIGURE_OPS = ("set_constraints", "set_objectives", "set_bounds", "set_acqf")
#: acquisition functions whose values are log-scale expected improvement / probabilities
_LOG_ACQFS = ("logei", "noisy_logei", "logpi")
_NOISE_RE = re.compile(r"noise[^.]{0,80}?standard deviation\s+(?:of\s+)?([0-9][0-9.eE+-]*)", re.IGNORECASE)


# =========================================================================== helpers


def _num(v: Any) -> float | None:
    """``v`` as a finite float, else ``None``."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    x = float(v)
    return x if math.isfinite(x) else None


def _fmt(v: Any, digits: int = 4) -> str:
    x = _num(v)
    return "n/a" if x is None else f"{x:.{digits}g}"


def _to_unit(p: RealParam | IntParam, v: float) -> float:
    if p.log_scale:
        return (math.log(v) - math.log(p.lower)) / (math.log(p.upper) - math.log(p.lower))
    return (v - p.lower) / (p.upper - p.lower) if p.upper > p.lower else 0.0


def _from_unit(p: RealParam | IntParam, u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    if p.log_scale:
        return math.exp(math.log(p.lower) + u * (math.log(p.upper) - math.log(p.lower)))
    return p.lower + u * (p.upper - p.lower)


def space_from_tools(session: "AgentSession") -> SearchSpace | None:
    """The search space as the agent sees it: rebuilt from the ``evaluate`` tool's config schema.

    Numbers become real parameters and integers become integer parameters, with
    ``log_scale`` taken from the description. Enums are ordinal when described as such,
    categorical otherwise. Returns ``None`` if the schema cannot be read.
    """
    try:
        spec = next(d for d in session.tool_definitions() if d["name"] == "evaluate")
        props = spec["input_schema"]["properties"]["config"]["properties"]
        params: list = []
        for name, ps in props.items():
            desc = str(ps.get("description", ""))
            log = "log scale" in desc
            if "enum" in ps:
                cls = OrdinalParam if desc.startswith("ordinal") else CategoricalParam
                params.append(cls(name, list(ps["enum"])))
            elif ps.get("type") == "integer":
                params.append(IntParam(name, int(ps["minimum"]), int(ps["maximum"]), log_scale=log))
            else:
                params.append(RealParam(name, float(ps["minimum"]), float(ps["maximum"]), log_scale=log))
        return SearchSpace(params)
    except (StopIteration, KeyError, TypeError, ValueError, LenzError):
        return None


def declared_noise_std(brief: str) -> float:
    """Observation noise std stated in a problem brief ("... noise with standard deviation 0.5 ..."), else 0."""
    m = _NOISE_RE.search(brief or "")
    if not m:
        return 0.0
    try:
        v = float(m.group(1).rstrip("."))
    except ValueError:
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


@dataclass
class _Candidate:
    """A proposal with what the surrogate believes about it."""

    config: dict
    kind: str                 # local | global | explore | restart | sobol | prior | instructed | moo
    acqf: str | None = None
    value: float | None = None
    mean: float | None = None
    std: float | None = None
    pof: float | None = None
    duplicate_of: int | None = None


@dataclass
class _State:
    """Per-campaign beliefs and bookkeeping (reset by :meth:`HeuristicSaraPolicy.start`)."""

    space: SearchSpace | None = None
    d: int = 1
    cumulative: bool = False
    metric: str = "y"
    minimize: bool = False
    is_moo: bool = False
    n_constraints: int = 0
    acqf: str = "logei"
    acqf_params: dict = field(default_factory=dict)
    acqf_locked: bool = False          # an instruction chose the acquisition: keep it
    bounds: dict | None = None
    bounds_owner: str | None = None    # "prior" (may be widened) | "instruction" (kept)
    region_checked_t: int = 0
    n_init: int = 5
    max_init: int = 5
    opening: bool = True
    model_phase: bool = False
    planned: list = field(default_factory=list)     # [(config, kind, rationale)]
    best_u: float | None = None        # best feasible observed utility (maximize convention)
    worst_u: float | None = None       # worst feasible observed utility (with best_u: the observed range)
    best_tid: int | None = None
    best_cfg: dict | None = None
    last_global_t: int = 0             # last evaluation of a global / exploration candidate
    global_fails: int = 0              # global / exploration steps in a row that did not improve
    distrust_steps: int = 0            # model steps in a row with a surrogate below distrust_threshold
    last_improve_t: int = 0
    last_stagnation_t: int = 0
    radius: float = 0.2
    succ: int = 0
    fail: int = 0
    center: dict | None = None         # None: centre the trust region on the incumbent
    local_best: float | None = None
    restarts: int = 0
    cv_r2: float | None = None
    sensitivity: dict | None = None
    noise_std: float = 0.0             # observation noise std stated in the brief (raw units)
    evals: dict = field(default_factory=dict)       # kind -> evaluations
    wins: dict = field(default_factory=dict)        # kind -> improvements of the incumbent
    last_note_t: int = 0


# =========================================================================== policy


class HeuristicSaraPolicy(Policy):
    """Scripted agentic BO policy: trust-calibrated global/local search through the agent tools.

    Every knob is a keyword argument. ``None`` means a default that depends on the
    problem: its dimension ``d``, the budget ``T`` and the regret mode.

    Args:
        n_init: base opening size under simple regret (default ``clip(d + 1, 4, T // 5)``).
        max_init: the opening is extended up to this size while the surrogate is not
            predictive (default ``clip(2 (d + 1), n_init, T // 4)``).
        n_init_cumulative: opening size under cumulative regret
            (default ``clip(ceil((d + 1) / 2), 2, T // 10)``).
        trust_threshold: CV R^2 at or above which the surrogate is trusted to make
            global moves.
        distrust_threshold: CV R^2 below which the opening is extended.
        radius_init, radius_min, radius_max: trust-region half-width (normalized units,
            scaled per parameter by the GP lengthscales): its start value, the collapse
            threshold that triggers a restart, and its maximum.
        success_tol, failure_tol: successive local improvements / failures that double /
            halve the radius (``failure_tol`` default ``max(4, d)``).
        global_margin: the global candidate wins when its standardized posterior mean
            exceeds the reference by more than this.
        global_ref: the reference for that comparison: ``"best"`` = the larger of the local
            candidate's and the incumbent's posterior mean (the model must predict an
            improvement elsewhere), ``"local"`` = the local candidate's only.
        global_ei_ratio: under simple regret the global candidate also wins when its
            expected improvement is this many times larger (``None`` disables the rule).
        stagnation_steps: evaluations without improvement before a stagnation step
            (default ``max(8, T // 8)``).
        explore_beta: UCB ``beta`` of stagnation (exploration) steps.
        endgame_frac: fraction of the budget at the end that is pure exploitation.
        endgame_radius: largest trust-region radius in the end game.
        cumulative_beta: UCB ``beta`` used throughout under cumulative regret.
        cumulative_radius: initial trust-region radius under cumulative regret.
        note_every: evaluations between ``record_note`` summaries (default ``max(5, T // 5)``).
        region_check_every: evaluations between checks of a prior region's boundaries.
        edge_frac: the incumbent is "at the boundary" of a prior region when it lies
            within this fraction of the region's width from an interior face.
        widen_frac: a widened face moves out by this fraction of the region's width.
        global_cooldown: after ``k`` failed global moves in a row the next one waits
            ``2 ** min(k, global_cooldown)`` evaluations (0 disables the cooldown).
        distrust_patience, distrust_explore_every: once the surrogate has stayed below
            ``distrust_threshold`` for ``distrust_patience`` model steps in a row, every
            ``distrust_explore_every``-th step explores instead of refining locally (simple
            regret only; ``distrust_explore_every=0`` disables this). Brief dips in CV R^2
            do not trigger it.
        distrust_explore: what such a step evaluates: ``"global"`` = the acquisition's
            optimum over the active region, ``"sobol"`` = a space-filling point.
        strict: re-raise internal errors instead of leaving the step to the harness
            fallback (useful in tests).
    """

    name = "heuristic"
    toolset = "full"

    def __init__(self, *, n_init: int | None = None, max_init: int | None = None,
                 n_init_cumulative: int | None = None, trust_threshold: float = 0.5,
                 distrust_threshold: float = 0.2, radius_init: float = 0.2, radius_min: float = 0.01,
                 radius_max: float = 0.8, success_tol: int = 3, failure_tol: int | None = None,
                 global_margin: float = 0.0, global_ref: str = "best", global_ei_ratio: float | None = 10.0,
                 stagnation_steps: int | None = None, explore_beta: float = 9.0,
                 endgame_frac: float = 0.1, endgame_radius: float = 0.1, cumulative_beta: float = 1.0,
                 cumulative_radius: float = 0.1, note_every: int | None = None, region_check_every: int = 5,
                 edge_frac: float = 0.05, widen_frac: float = 0.5, global_cooldown: int = 4,
                 distrust_patience: int = 6, distrust_explore_every: int = 3, distrust_explore: str = "global",
                 strict: bool = False):
        for name, v in (("n_init", n_init), ("max_init", max_init), ("n_init_cumulative", n_init_cumulative),
                        ("failure_tol", failure_tol), ("stagnation_steps", stagnation_steps),
                        ("note_every", note_every)):
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
                raise LenzError(f"{name} must be an integer >= 1 or None, got {v!r}")
        for name, v in (("success_tol", success_tol), ("region_check_every", region_check_every)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise LenzError(f"{name} must be an integer >= 1, got {v!r}")
        if not 0 < radius_min <= radius_init <= radius_max <= 1:
            raise LenzError(f"need 0 < radius_min <= radius_init <= radius_max <= 1, got "
                            f"{radius_min}, {radius_init}, {radius_max}")
        if not 0 < cumulative_radius <= 1 or not 0 < endgame_radius <= 1:
            raise LenzError("cumulative_radius and endgame_radius must lie in (0, 1]")
        if not 0 <= endgame_frac < 1:
            raise LenzError(f"endgame_frac must lie in [0, 1), got {endgame_frac!r}")
        if global_ref not in ("best", "local"):
            raise LenzError(f"global_ref must be 'best' or 'local', got {global_ref!r}")
        if global_ei_ratio is not None and global_ei_ratio <= 1:
            raise LenzError(f"global_ei_ratio must be > 1 or None, got {global_ei_ratio!r}")
        if explore_beta < 0 or cumulative_beta < 0:
            raise LenzError("explore_beta and cumulative_beta must be >= 0")
        if not (0 < edge_frac < 0.5 and widen_frac > 0):
            raise LenzError("need 0 < edge_frac < 0.5 and widen_frac > 0")
        for name, v in (("global_cooldown", global_cooldown), ("distrust_patience", distrust_patience),
                        ("distrust_explore_every", distrust_explore_every)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise LenzError(f"{name} must be an integer >= 0, got {v!r}")
        if distrust_explore not in ("global", "sobol"):
            raise LenzError(f"distrust_explore must be 'global' or 'sobol', got {distrust_explore!r}")
        self.n_init = n_init
        self.max_init = max_init
        self.n_init_cumulative = n_init_cumulative
        self.trust_threshold = float(trust_threshold)
        self.distrust_threshold = float(distrust_threshold)
        self.radius_init = float(radius_init)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.success_tol = success_tol
        self.failure_tol = failure_tol
        self.global_margin = float(global_margin)
        self.global_ref = global_ref
        self.global_ei_ratio = None if global_ei_ratio is None else float(global_ei_ratio)
        self.stagnation_steps = stagnation_steps
        self.explore_beta = float(explore_beta)
        self.endgame_frac = float(endgame_frac)
        self.endgame_radius = float(endgame_radius)
        self.cumulative_beta = float(cumulative_beta)
        self.cumulative_radius = float(cumulative_radius)
        self.note_every = note_every
        self.region_check_every = region_check_every
        self.edge_frac = float(edge_frac)
        self.widen_frac = float(widen_frac)
        self.global_cooldown = global_cooldown
        self.distrust_patience = distrust_patience
        self.distrust_explore_every = distrust_explore_every
        self.distrust_explore = distrust_explore
        self.strict = bool(strict)
        self._st = _State()
        self._eval_error: Any = None    # error of the last rejected evaluate call

    # ------------------------------------------------------------------ derived settings
    def _defaults(self, session: "AgentSession") -> None:
        st, T, d = self._st, session.budget, self._st.d
        if st.cumulative:
            n = self.n_init_cumulative or min(max(math.ceil((d + 1) / 2), 2), max(T // 10, 2))
            st.n_init = st.max_init = max(1, min(n, T))
            st.radius = self.cumulative_radius
        else:
            n = self.n_init or min(max(d + 1, 4), max(T // 5, 2))
            st.n_init = max(1, min(n, T))
            m = self.max_init or min(max(2 * (d + 1), st.n_init), max(T // 4, st.n_init))
            st.max_init = max(st.n_init, min(m, T))
            st.radius = self.radius_init
        self._failure_tol = self.failure_tol or max(4, d)
        self._stagnation_steps = self.stagnation_steps or max(8, T // 8)
        self._note_every = self.note_every or max(5, T // 5)
        self._endgame_n = int(round(self.endgame_frac * T))

    # ------------------------------------------------------------------ policy interface
    def start(self, session: "AgentSession") -> None:
        st = self._st = _State()
        st.cumulative = session.config.regret == "cumulative"
        st.space = space_from_tools(session)
        st.d = st.space.d if st.space is not None else 1
        st.noise_std = declared_noise_std(session.brief())
        self._defaults(session)
        env = session.call("status")
        if env.get("ok"):
            res = env["result"]
            self._sync_config(res.get("lenz_config") or {}, bool(res.get("is_moo")))
        for item in list(session.context):
            if item.source != "problem":
                self.on_context(session, item)

    def on_context(self, session: "AgentSession", item: "ContextItem") -> None:
        try:
            self._apply_context(session, item)
        except PolicyError:
            raise
        except Exception as exc:  # noqa: BLE001 - a scripted agent must not crash the campaign
            if self.strict:
                raise
            session.log_reasoning(f"heuristic policy could not apply context: {type(exc).__name__}: {exc}",
                                  kind="error")

    def step(self, session: "AgentSession") -> None:
        try:
            self._step(session)
        except PolicyError:
            raise
        except Exception as exc:  # noqa: BLE001 - leave the step to the harness fallback
            if self.strict:
                raise
            session.log_reasoning(f"heuristic policy internal error: {type(exc).__name__}: {exc}", kind="error")

    def finish(self, session: "AgentSession") -> None:
        st = self._st
        if session.t > st.last_note_t and not session.stopped:
            self._note(session, final=True)

    # ------------------------------------------------------------------ tool plumbing
    @staticmethod
    def _result(env: Mapping[str, Any]) -> dict | None:
        return env["result"] if env.get("ok") and isinstance(env.get("result"), dict) else None

    @staticmethod
    def _room(session: "AgentSession", n: int) -> bool:
        """True if ``n`` more actions plus the evaluation keep this step within the soft action limit."""
        return session.actions_this_step + n + 1 <= session.config.soft_actions_per_step

    def _sync_config(self, cfg: Mapping[str, Any], is_moo: bool) -> None:
        """Mirror the lenz configuration (objective, constraints, acquisition, region)."""
        st = self._st
        objs = cfg.get("objectives") or []
        if objs:
            st.metric = str(objs[0].get("metric"))
            st.minimize = bool(objs[0].get("minimize"))
        st.is_moo = bool(is_moo) or len(objs) >= 2
        st.n_constraints = len(cfg.get("constraints") or [])
        if cfg.get("acqf"):
            st.acqf = str(cfg["acqf"])
            st.acqf_params = dict(cfg.get("acqf_params") or {})
        st.bounds = cfg.get("bounds") or None

    def _utility(self, metrics: Mapping[str, Any] | None, feasible: Any = True) -> float | None:
        if not metrics or feasible is False:
            return None
        v = _num(metrics.get(self._st.metric))
        if v is None:
            return None
        return -v if self._st.minimize else v

    def _raw(self, u: float | None) -> float | None:
        return None if u is None else (-u if self._st.minimize else u)

    def _tol(self, u: float | None) -> float:
        """Smallest improvement over ``u`` that counts as a success.

        TuRBO's relative tolerance ``1e-3 |u|``, capped at ``1e-3`` of the observed utility
        range: an additive offset of the objective must not turn every real improvement
        into a failure. Never below the observation noise level stated in the brief.
        """
        st = self._st
        scale = 0.0
        if u is not None:
            scale = abs(u)
            if st.best_u is not None and st.worst_u is not None:
                scale = min(scale, st.best_u - st.worst_u)
        return max(1e-3 * scale, st.noise_std)

    def _refresh_incumbent(self, session: "AgentSession") -> None:
        """Re-read the incumbent (after objectives or constraints changed)."""
        st = self._st
        res = self._result(session.call("incumbent"))
        if res is None:
            return
        inc = res.get("incumbent") if not res.get("is_moo") else None
        if inc:
            st.best_u = self._utility(inc.get("metrics"))
            st.best_tid, st.best_cfg = inc.get("trial_id"), dict(inc.get("config") or {}) or None
        else:
            st.best_u, st.best_tid, st.best_cfg = None, None, None
        st.worst_u = st.best_u  # the observed range of the new utility is re-learned from here
        st.local_best, st.center, st.succ, st.fail = st.best_u, None, 0, 0

    def _suggest(self, session: "AgentSession", kind: str, **args: Any) -> _Candidate | None:
        env = session.call("suggest", {k: v for k, v in args.items() if v is not None})
        res = self._result(env)
        if res is None or not res.get("candidates"):
            return None
        c = res["candidates"][0]
        acqf = res.get("acqf")
        pred = (c.get("predicted") or {}).get(self._st.metric) or {}
        return _Candidate(config=dict(c["config"]), kind=kind, acqf=acqf,
                          value=_num((c.get("acquisition_values") or {}).get(acqf)),
                          mean=_num(pred.get("mean")), std=_num(pred.get("std")),
                          pof=_num(c.get("prob_feasible")), duplicate_of=c.get("duplicate_of"))

    def _predicted(self, session: "AgentSession", config: dict, kind: str) -> _Candidate:
        """``config`` as a candidate, with the surrogate's prediction once there is data to predict from."""
        cand = _Candidate(config=dict(config), kind=kind)
        if session.t < 2:
            return cand
        res = self._result(session.call("predict", {"configs": [cand.config]}))
        preds = (res or {}).get("predictions") or []
        if preds:
            m = (preds[0].get("metrics") or {}).get(self._st.metric) or {}
            cand.mean, cand.std, cand.pof = _num(m.get("mean")), _num(m.get("std")), _num(preds[0].get("prob_feasible"))
        return cand

    def _set_acqf(self, session: "AgentSession", acqf: str, params: Mapping[str, Any] | None = None) -> bool:
        args: dict[str, Any] = {"acqf": acqf}
        params = dict(params or {})
        if "beta" in params:
            args["beta"] = params.pop("beta")
        if params:
            args["params"] = params
        res = self._result(session.call("set_acqf", args))
        if res is None:
            return False
        self._st.acqf, self._st.acqf_params = res["acqf"], dict(res.get("acqf_params") or {})
        return True

    def _set_bounds(self, session: "AgentSession", bounds: dict | None, owner: str | None) -> bool:
        res = self._result(session.call("set_bounds", {"bounds": bounds}))
        if res is None:
            return False
        self._st.bounds = res.get("bounds") or None
        self._st.bounds_owner = owner if self._st.bounds else None
        return True

    def _note(self, session: "AgentSession", text: str | None = None, final: bool = False) -> None:
        """Write ``text`` (default: a belief summary, which also resets the summary clock) to the notebook."""
        if text is None:
            text = self._belief_summary(session, final)
            self._st.last_note_t = session.t
        session.call("record_note", {"text": text[:3900]})

    # ------------------------------------------------------------------ context
    def _apply_context(self, session: "AgentSession", item: "ContextItem") -> None:
        st = self._st
        data = item.data or {}
        hint = data.get("prior_hint")
        if isinstance(hint, Mapping):
            self._apply_prior(session, hint)
            return
        op = data.get("op")
        if op is None:
            if item.source != "prior":
                self._note(session, f"Context received at t={session.t} ({item.source}): {item.text.strip()[:600]} "
                                    f"-- no structured action attached; keeping the current strategy.")
            return
        args = data.get("args")
        if args is None:
            args = {}
        elif not isinstance(args, Mapping):
            self._note(session, f"Instruction {_short(op, 60)} not applied: its 'args' must be an object of tool "
                                f"arguments, got {_short(args, 120)}. {item.text.strip()[:400]}")
            return
        args = dict(args)
        if op in _RECONFIGURE_OPS:
            env = session.call(op, args)
            res = self._result(env)
            if res is None:
                self._note(session, f"Instruction {op} could not be applied: {env.get('error')}")
                return
            if op == "set_bounds":
                st.bounds = res.get("bounds") or None
                st.bounds_owner = "instruction" if st.bounds else None
            elif op == "set_acqf":
                st.acqf, st.acqf_params = res["acqf"], dict(res.get("acqf_params") or {})
                st.acqf_locked = True
            else:
                if op == "set_objectives":
                    objs = res.get("objectives") or []
                    if objs:
                        st.metric, st.minimize = str(objs[0]["metric"]), bool(objs[0]["minimize"])
                    st.is_moo = bool(res.get("is_moo"))
                    st.acqf = str(res.get("acqf") or st.acqf)
                    st.acqf_params = dict(res.get("acqf_params") or {})
                    if res.get("acqf_switched") and st.model_phase:
                        self._regime_acqf(session)  # e.g. back from multi-objective: lenz chose logei
                else:
                    st.n_constraints = len(res.get("constraints") or [])
                self._refresh_incumbent(session)
                st.radius = max(st.radius, self.cumulative_radius if st.cumulative else self.radius_init)
                st.last_improve_t = session.t
            self._note(session, f"Instruction applied at t={session.t}: {op}({_short(args)}). "
                                f"{item.text.strip()[:400]}")
        elif op == "evaluate" and isinstance(args.get("config"), Mapping):
            st.planned.append((dict(args["config"]), "instructed",
                               f"instructed evaluation (context at t={session.t}): {item.text.strip()[:300]}"))
        elif op == "record_note":
            self._note(session, str(args.get("text") or item.text))
        else:
            self._note(session, f"Instruction with unsupported op {op!r} noted, not applied: {item.text[:400]}")

    def _apply_prior(self, session: "AgentSession", hint: Mapping[str, Any]) -> None:
        st = self._st
        cfg = hint.get("config")
        if isinstance(cfg, Mapping):
            st.planned.append((dict(cfg), "prior", "prior knowledge: the domain prior names this configuration "
                                                   "as promising; testing it first anchors the search"))
        region = hint.get("region")
        if isinstance(region, Mapping) and region and st.bounds_owner != "instruction":
            if self._set_bounds(session, dict(region), "prior"):
                self._note(session, f"Prior region adopted as the active region: {_short(st.bounds)}. "
                                    f"The opening fills it; it will be widened if the optimum appears to lie "
                                    f"at its boundary.")

    # ------------------------------------------------------------------ the step
    def _step(self, session: "AgentSession") -> None:
        st = self._st
        if session.is_done:
            return
        while st.planned:
            cfg, kind, why = st.planned.pop(0)
            if self._commit(session, self._predicted(session, cfg, kind), why, extra=""):
                return
            error = str(self._eval_error or "unknown error")[:600]
            self._note(session, f"The {kind} configuration {_config_text(cfg)} could not be evaluated: {error}. "
                                f"Continuing with the regular strategy.")
        if st.is_moo:
            self._moo_step(session)
            return
        if st.opening and session.t < st.n_init:
            self._opening_step(session, "space-filling opening design")
            return
        cv = self._diagnose(session)
        if st.opening:
            if (not st.cumulative and session.t < st.max_init and (cv is None or cv < self.distrust_threshold)):
                self._opening_step(session, f"opening extended: the surrogate is not predictive yet "
                                            f"(CV R^2 = {_fmt(cv, 3)})")
                return
            st.opening = False
        if not st.model_phase:
            self._enter_model_phase(session, cv)
        self._model_step(session, cv)

    def _diagnose(self, session: "AgentSession") -> float | None:
        st = self._st
        res = self._result(session.call("diagnostics", {"metric": st.metric}))
        model = ((res or {}).get("models") or {}).get(st.metric) or {}
        st.cv_r2 = _num(model.get("cv_r2"))
        st.sensitivity = model.get("sensitivity") or None
        return st.cv_r2

    def _regime_acqf(self, session: "AgentSession") -> None:
        """Use the acquisition the campaign calls for: UCB with a small beta under cumulative
        regret, noisy_logei instead of logEI on noisy data (never over an instructed choice)."""
        st = self._st
        if st.acqf_locked or st.is_moo:
            return
        if st.cumulative:
            self._set_acqf(session, "ucb", {"beta": self.cumulative_beta})
        elif st.noise_std and st.acqf == "logei" and self._set_acqf(session, "noisy_logei"):
            self._note(session, f"The brief says observations are noisy (std {_fmt(st.noise_std, 3)}): using "
                                f"noisy_logei, and improvements smaller than the noise do not count as successes.")

    def _enter_model_phase(self, session: "AgentSession", cv: float | None) -> None:
        st = self._st
        st.model_phase = True
        self._regime_acqf(session)
        self._note(session, f"Opening done after {session.t} evaluations (surrogate CV R^2 = {_fmt(cv, 3)}). "
                            f"Switching to model-guided search: local trust-region candidates around the "
                            f"incumbent, global candidates when the surrogate is trusted"
                            + (", UCB with a small beta for cumulative regret." if st.cumulative else "."))

    def _opening_step(self, session: "AgentSession", why: str) -> None:
        cand = self._suggest(session, "sobol", acqf="sobol")
        if cand is None:
            return
        region = "inside the prior region" if self._st.bounds_owner == "prior" else \
            ("inside the active region" if self._st.bounds else "over the full domain")
        self._commit(session, cand, f"{why}: Sobol point {session.t + 1} {region}, to learn the "
                                    f"landscape's broad structure before trusting a surrogate", extra="")

    def _moo_step(self, session: "AgentSession") -> None:
        st = self._st
        if session.t < st.n_init:
            self._opening_step(session, "space-filling opening design (multi-objective)")
            return
        cand = self._suggest(session, "moo")
        why = f"multi-objective step: top {cand.acqf if cand else st.acqf} candidate"
        if cand is not None and cand.acqf == "ehvi" and (cand.value is None or cand.value <= 1e-12):
            alt = self._suggest(session, "moo", acqf="parego")
            if alt is not None:
                cand, why = alt, ("expected hypervolume improvement is flat (nothing dominates the reference "
                                  "point yet): ParEGO scalarization candidate instead")
        if cand is None or cand.duplicate_of is not None:
            alt = self._suggest(session, "sobol", acqf="sobol")
            if alt is not None:
                cand, why = alt, "acquisition proposal repeats an evaluated trial: space-filling point instead"
        if cand is not None:
            self._commit(session, cand, why, extra="")

    # ------------------------------------------------------------------ model phase
    def _model_step(self, session: "AgentSession", cv: float | None) -> None:
        st = self._st
        trusted = cv is not None and cv >= self.trust_threshold
        endgame = session.remaining <= self._endgame_n
        trust_txt = f"surrogate CV R^2 = {_fmt(cv, 3)} ({'trusted' if trusted else 'not trusted for global moves'})"

        if st.bounds_owner == "prior" and not endgame and self._room(session, 6):
            self._check_region(session)

        stagnating = (not st.cumulative and not endgame
                      and session.t - st.last_improve_t >= self._stagnation_steps
                      and session.t - st.last_stagnation_t >= self._stagnation_steps)
        if stagnating:
            st.last_stagnation_t = session.t
            if st.bounds_owner == "prior":
                if self._set_bounds(session, None, None):
                    self._note(session, f"No improvement for {session.t - st.last_improve_t} evaluations inside "
                                        f"the prior region: widening the search to the full domain.")
            elif self._room(session, 3) and self._explore_step(session, trusted, trust_txt):
                return

        if not st.cumulative and not endgame and st.radius < self.radius_min:
            if self._restart_step(session, trust_txt):
                return
            st.radius = self.radius_init  # restart failed to propose: keep going locally

        distrusted = cv is None or cv < self.distrust_threshold
        st.distrust_steps = st.distrust_steps + 1 if distrusted else 0
        if (distrusted and not st.cumulative and not endgame and self.distrust_explore_every
                and st.distrust_steps >= self.distrust_patience
                and (st.distrust_steps - self.distrust_patience) % self.distrust_explore_every == 0
                and self._distrust_step(session, trust_txt)):
            return
        radius = min(st.radius, self.endgame_radius) if endgame else st.radius
        around: Any = "incumbent" if (st.center is None or endgame) else st.center
        local = self._suggest(session, "endgame" if endgame else "local", around=around, radius=radius)

        glob = None
        # credit assignment: after failed global moves, wait 2^failures steps before the next one
        cooled = session.t - st.last_global_t >= 2 ** min(st.global_fails, self.global_cooldown)
        if not endgame and trusted and cooled and self._room(session, 3):
            glob = self._suggest(session, "global")
        cand, why = self._choose(session, local, glob, around, radius)
        if cand is None:
            cand = self._suggest(session, "sobol", acqf="sobol")
            why = "no model-based candidate available: space-filling point instead"
            if cand is None:
                return
        where = "the incumbent" if around == "incumbent" else "the restart centre"
        if cand.kind in ("local", "endgame"):
            phase = "end game, exploit" if endgame else "local trust-region step"
            why = (f"{phase}: {cand.acqf} candidate within radius {_fmt(radius, 3)} of {where} "
                   f"({st.succ} successes / {st.fail} failures in a row); {why}")
        self._commit(session, cand, why, extra=trust_txt)

    def _choose(self, session: "AgentSession", local: _Candidate | None, glob: _Candidate | None,
                around: Any, radius: float) -> tuple[_Candidate | None, str]:
        """Pick between the local and the global candidate (``score`` on a common scale).

        Duplicates of evaluated trials are avoided: an exhausted trust region (its optimum
        repeats a trial) falls back to a random point in a doubled region, then to the
        global acquisition optimum, then to a space-filling point. A duplicate is returned
        only when every alternative repeats a trial too (an exhausted discrete space).
        """
        st = self._st
        repeat = None
        if local is not None and local.duplicate_of is not None:
            alt = self._suggest(session, local.kind, acqf="sobol", around=around,
                                radius=min(1.0, max(2 * radius, 0.05)))
            if alt is not None and alt.duplicate_of is None:
                return alt, (f"the local optimum repeats trial {local.duplicate_of}: random point in a wider "
                             f"trust region")
            repeat, local = local, None
        if glob is not None and glob.duplicate_of is not None:
            repeat, glob = repeat or glob, None
        elif glob is None and local is None:
            glob = self._suggest(session, "global")
            if glob is not None and glob.duplicate_of is not None:
                repeat, glob = repeat or glob, None
        if local is None and glob is None:
            alt = self._suggest(session, "sobol", acqf="sobol")
            if alt is not None and alt.duplicate_of is None:
                return alt, "every model-based proposal repeats an evaluated trial: space-filling point instead"
            return repeat or alt, "every proposal repeats an evaluated trial (the space may be exhausted)"
        if glob is None:
            return local, "no better global alternative"
        if local is None:
            return glob, (f"the trust region is exhausted (its proposals repeat trial {repeat.duplicate_of}): "
                          f"global candidate instead" if repeat else "no local candidate: global candidate instead")
        names = ["pm"] + ([st.acqf] if st.acqf in POINTWISE_ACQFS and st.acqf != "pm" else [])
        configs = [local.config, glob.config] + ([st.best_cfg] if st.best_cfg else [])
        res = self._result(session.call("score", {"configs": configs, "acqfs": names}))
        vals = [s.get("acquisition_values") or {} for s in (res or {}).get("scores") or []]
        if len(vals) != len(configs):
            return local, "score failed: staying local"
        pm_l, pm_g = _num(vals[0].get("pm")), _num(vals[1].get("pm"))
        pm_i = _num(vals[2].get("pm")) if len(vals) > 2 else None
        a_l, a_g = _num(vals[0].get(st.acqf)), _num(vals[1].get(st.acqf))
        pool = (pm_l, pm_i) if self.global_ref == "best" else (pm_l,)
        ref = max([v for v in pool if v is not None] or [math.inf])
        if pm_g is not None and pm_g > ref + self.global_margin:
            rivals = "both the local candidate and the incumbent" if self.global_ref == "best" else \
                "the local candidate"
            return glob, (f"global candidate: the surrogate's posterior mean there beats {rivals} "
                          f"(predicted {st.metric} ~ {_fmt(glob.mean)} vs {_fmt(local.mean)} at the local "
                          f"candidate)")
        if (not st.cumulative and self.global_ei_ratio is not None and st.acqf in _LOG_ACQFS
                and a_l is not None and a_g is not None and a_g - a_l > math.log(self.global_ei_ratio)):
            return glob, (f"global candidate: its {st.acqf} is {_fmt(math.exp(min(a_g - a_l, 700)), 3)}x the "
                          f"local candidate's (threshold {_fmt(self.global_ei_ratio, 3)}x)")
        return local, (f"score: local {st.acqf}={_fmt(a_l, 3)} (predicted {st.metric} ~ {_fmt(local.mean)}) vs "
                       f"global {_fmt(a_g, 3)} (~ {_fmt(glob.mean)}): the global candidate is not convincingly "
                       f"better")

    def _explore_step(self, session: "AgentSession", trusted: bool, trust_txt: str) -> bool:
        """Stagnation: one UCB step with a large beta (``set_acqf`` -> ``suggest`` -> restore)."""
        st = self._st
        stalled = session.t - st.last_improve_t
        prev = (st.acqf, dict(st.acqf_params))
        beta = self.explore_beta
        if st.acqf_locked:  # keep the instructed acquisition; UCB for this call only (default beta)
            beta = DEFAULT_UCB_BETA
            cand = self._suggest(session, "explore", acqf="ucb")
        else:
            if not self._set_acqf(session, "ucb", {"beta": self.explore_beta}):
                return False
            if trusted:
                cand = self._suggest(session, "explore")
            else:
                cand = self._suggest(session, "explore", around="incumbent",
                                     radius=min(1.0, max(2 * st.radius, 2 * self.radius_init)))
            self._set_acqf(session, *prev)
        if cand is None or cand.duplicate_of is not None:
            return False
        scope = "over the active region" if trusted else "in a wide region around the incumbent"
        return self._commit(session, cand, f"stagnation: no improvement for {stalled} evaluations; one exploratory "
                                           f"UCB step (beta {_fmt(beta, 3)}) {scope}", extra=trust_txt)

    def _distrust_step(self, session: "AgentSession", trust_txt: str) -> bool:
        """Explore away from the incumbent while the surrogate is not predictive."""
        st = self._st
        cand = None
        if self.distrust_explore == "global":
            cand = self._suggest(session, "distrust")
            what = f"the global {cand.acqf if cand else st.acqf} optimum"
        if cand is None or cand.duplicate_of is not None:
            cand = self._suggest(session, "distrust", acqf="sobol")
            what = "a space-filling point"
        if cand is None:
            return False
        return self._commit(session, cand, f"distrust: the surrogate has not been predictive for {st.distrust_steps} "
                                           f"steps, so instead of refining its local proposals this step explores "
                                           f"{what}", extra=trust_txt)

    def _restart_step(self, session: "AgentSession", trust_txt: str) -> bool:
        """The trust region collapsed: restart it from a global candidate (Sobol if that repeats a trial)."""
        cand = self._suggest(session, "restart")
        if cand is None or cand.duplicate_of is not None:
            cand = self._suggest(session, "restart", acqf="sobol")
        if cand is None:
            return False
        cand.kind = "restart"
        return self._commit(session, cand, f"trust region collapsed (radius < {_fmt(self.radius_min, 3)}) after "
                                           f"converging locally: restarting from a global {cand.acqf} candidate",
                            extra=trust_txt)

    def _check_region(self, session: "AgentSession") -> None:
        """Widen a prior region whose interior faces the optimum seems to lie at."""
        st = self._st
        if st.space is None or not st.bounds or session.t - st.region_checked_t < self.region_check_every:
            return
        st.region_checked_t = session.t
        inc = self._result(session.call("incumbent"))
        inc_cfg = ((inc or {}).get("incumbent") or {}).get("config")
        new, why = None, ""
        if inc_cfg:
            new = self._widen_at_edges(inc_cfg)
            why = "the incumbent lies at an interior face of the prior region"
        if new is None:
            cand = self._suggest(session, "probe", acqf="pm", bounds="full")
            if (cand is not None and cand.mean is not None and st.best_u is not None
                    and not st.space.contains(cand.config, st.bounds)
                    and (-cand.mean if st.minimize else cand.mean) > st.best_u + self._tol(st.best_u)):
                new = self._include(cand.config)
                why = (f"the posterior mean predicts {st.metric}={_fmt(cand.mean)} outside the prior region, "
                       f"better than the best observed {_fmt(self._raw(st.best_u))}")
        if new is not None and new != st.bounds and self._set_bounds(session, new, "prior"):
            self._note(session, f"Widened the active region ({why}): {_short(st.bounds)}.")

    def _widen_at_edges(self, cfg: Mapping[str, Any]) -> dict | None:
        st = self._st
        space, bounds = st.space, dict(st.bounds or {})
        changed = False
        for name, lim in list(bounds.items()):
            if name not in space or name not in cfg:
                continue
            p = space[name]
            if not isinstance(p, (RealParam, IntParam)):
                continue
            lo, hi, v = _to_unit(p, float(lim[0])), _to_unit(p, float(lim[1])), _to_unit(p, float(cfg[name]))
            width = hi - lo
            if width <= 0:
                continue
            new_lo, new_hi = lo, hi
            if lo > 1e-9 and v - lo <= self.edge_frac * width:
                new_lo = max(0.0, lo - self.widen_frac * width)
            if hi < 1 - 1e-9 and hi - v <= self.edge_frac * width:
                new_hi = min(1.0, hi + self.widen_frac * width)
            if (new_lo, new_hi) != (lo, hi):
                bounds[name] = _raw_interval(p, new_lo, new_hi)
                changed = True
        return bounds if changed else None

    def _include(self, cfg: Mapping[str, Any]) -> dict:
        """The active region grown to contain ``cfg`` with a margin."""
        st = self._st
        space, bounds = st.space, dict(st.bounds or {})
        for name, lim in list(bounds.items()):
            if name not in space or name not in cfg:
                continue
            p = space[name]
            v = cfg[name]
            if isinstance(p, (RealParam, IntParam)):
                lo, hi, u = _to_unit(p, float(lim[0])), _to_unit(p, float(lim[1])), _to_unit(p, float(v))
                margin = 0.1 * max(hi - lo, 0.05)
                if u < lo:
                    lo = max(0.0, u - margin)
                if u > hi:
                    hi = min(1.0, u + margin)
                bounds[name] = _raw_interval(p, lo, hi)
            elif v not in lim:
                bounds[name] = list(lim) + [v]
        return bounds

    # ------------------------------------------------------------------ commit and learn
    def _commit(self, session: "AgentSession", cand: _Candidate, why: str, extra: str) -> bool:
        """Evaluate ``cand`` with a rationale and expectation; update beliefs from the outcome."""
        st = self._st
        best = "no feasible point yet" if st.best_u is None else \
            f"best so far {st.metric}={_fmt(self._raw(st.best_u))} (trial {st.best_tid})"
        rationale = f"{why}. {extra + '; ' if extra else ''}{best}."
        expectation = None
        if cand.mean is not None:
            expectation = f"{st.metric} ~ {_fmt(cand.mean)} +/- {_fmt(cand.std, 3)} (GP posterior)"
            if cand.pof is not None and st.n_constraints:
                expectation += f", P(feasible) = {_fmt(cand.pof, 3)}"
        args: dict[str, Any] = {"config": cand.config, "rationale": rationale[:3900]}
        if expectation:
            args["expectation"] = expectation
        env = session.call("evaluate", args)
        res = self._result(env)
        if res is None:
            self._eval_error = env.get("error")
            return False
        self._learn(session, cand, res)
        if session.t - st.last_note_t >= self._note_every and not session.is_done:
            self._note(session)
        return True

    def _learn(self, session: "AgentSession", cand: _Candidate, res: Mapping[str, Any]) -> None:
        st = self._st
        kind = cand.kind
        st.evals[kind] = st.evals.get(kind, 0) + 1
        if st.is_moo:
            if res.get("is_new_best"):
                st.last_improve_t = session.t
                st.wins[kind] = st.wins.get(kind, 0) + 1
            return
        u = self._utility(res.get("observed"), res.get("feasible"))
        improved = u is not None and (st.best_u is None or u > st.best_u + self._tol(st.best_u))
        if u is not None and (st.best_u is None or u > st.best_u):
            st.best_u, st.best_tid, st.best_cfg = u, res.get("trial_id"), dict(res["config"])
        if u is not None and (st.worst_u is None or u < st.worst_u):
            st.worst_u = u
        if improved:
            st.last_improve_t = session.t
            st.wins[kind] = st.wins.get(kind, 0) + 1
        if kind in ("global", "explore"):
            st.last_global_t = session.t
            st.global_fails = 0 if improved else st.global_fails + 1
        if not st.model_phase or kind in ("sobol", "prior", "instructed", "probe"):
            if st.center is None:  # a restart run keeps measuring against its own best
                st.local_best = st.best_u
            return
        if kind in ("local", "endgame"):
            if u is not None and (st.local_best is None or u > st.local_best + self._tol(st.local_best)):
                st.succ, st.fail = st.succ + 1, 0
                st.local_best = u
                if st.center is not None:  # follow the restart run; rejoin the incumbent once it is the best
                    st.center = None if st.best_u is None or u >= st.best_u else dict(res["config"])
            else:
                st.succ, st.fail = 0, st.fail + 1
            if st.succ >= self.success_tol:
                st.radius, st.succ = min(2 * st.radius, self.radius_max), 0
            elif st.fail >= self._failure_tol:
                st.radius, st.fail = st.radius / 2, 0
                if st.cumulative:
                    st.radius = max(st.radius, self.radius_min)
        elif kind == "restart":
            st.restarts += 1
            st.center = dict(res["config"]) if u is not None and st.best_u is not None and u < st.best_u else None
            st.local_best = u if st.center is not None else st.best_u
            st.radius, st.succ, st.fail = self.radius_init, 0, 0
        elif kind in ("global", "explore", "distrust") and improved:
            st.center, st.local_best = None, st.best_u
            st.radius = max(st.radius, self.cumulative_radius if st.cumulative else self.radius_init)
            st.succ, st.fail = 0, 0

    # ------------------------------------------------------------------ notes
    def _belief_summary(self, session: "AgentSession", final: bool) -> str:
        st = self._st
        parts = [f"{'Final summary' if final else 'Summary'} at t={session.t}/{session.budget}:"]
        if st.is_moo:
            parts.append("multi-objective study; following the acquisition's Pareto-front proposals.")
        else:
            parts.append("no feasible point yet." if st.best_u is None else
                         f"best {st.metric}={_fmt(self._raw(st.best_u))} (trial {st.best_tid}), last improved at "
                         f"t={st.last_improve_t}.")
        if st.cv_r2 is not None:
            parts.append(f"Surrogate CV R^2 = {_fmt(st.cv_r2, 3)}.")
        if st.sensitivity:
            top = sorted(((k, _num(v) or 0.0) for k, v in st.sensitivity.items()), key=lambda kv: -kv[1])[:3]
            parts.append("Most influential inputs: " + ", ".join(f"{k} ({v:.2f})" for k, v in top) + ".")
        if st.model_phase and not st.is_moo:
            parts.append(f"Trust region radius {_fmt(st.radius, 3)} centred on "
                         f"{'the incumbent' if st.center is None else 'a restart point'}; {st.restarts} restart(s).")
        if st.evals:
            parts.append("Evaluations by kind: " + ", ".join(
                f"{k} {n} ({st.wins.get(k, 0)} improved)" for k, n in sorted(st.evals.items())) + ".")
        if st.bounds:
            parts.append(f"Active region: {_short(st.bounds)} ({st.bounds_owner or 'set'}).")
        if not final:
            left = session.remaining
            plan = ("exploit around the incumbent until the end" if left <= self._endgame_n else
                    "keep exploiting locally; move globally only when the surrogate predicts a better mean"
                    if st.cumulative else
                    "alternate local refinement with global moves when the trusted surrogate points elsewhere")
            parts.append(f"Plan ({left} evaluations left): {plan}.")
        return " ".join(parts)


def _raw_interval(p: RealParam | IntParam, lo_u: float, hi_u: float) -> list:
    lo, hi = _from_unit(p, lo_u), _from_unit(p, hi_u)
    if isinstance(p, IntParam):
        return [int(max(p.lower, math.floor(lo + 1e-9))), int(min(p.upper, math.ceil(hi - 1e-9)))]
    return [max(p.lower, lo), min(p.upper, hi)]


def _config_text(cfg: Mapping[str, Any], n: int = 300) -> str:
    text = ", ".join(f"{k}={_fmt(v, 6) if _num(v) is not None else v}" for k, v in cfg.items())
    return text if len(text) <= n else text[: n - 3] + "..."


def _short(obj: Any, n: int = 300) -> str:
    text = repr(obj)
    return text if len(text) <= n else text[: n - 3] + "..."
