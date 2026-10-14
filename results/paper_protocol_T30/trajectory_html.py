"""Render one campaign's full trajectory (reasoning, tool calls, evaluations) as a standalone HTML page.

Usage (from the repo root):
    python results/paper_protocol_T30/trajectory_html.py <run.json> [-o out.html]

The page is self-contained (no external scripts or fonts) and does not include the
endpoint address or any credentials.
"""
import argparse
import html
import json
import math
import os

CSS = """
:root{--bg:#fbfbf9;--panel:#ffffff;--ink:#16161a;--ink2:#55555c;--muted:#8a8a91;--line:#e4e3dd;
--accent:#2a6fdb;--think:#f3f1ea;--tool:#eef4fd;--eval:#eaf7ef;--best:#1f9d55;--warn:#c2410c;--code:#f6f6f3}
@media (prefers-color-scheme:dark){:root{--bg:#141417;--panel:#1c1c20;--ink:#ececf0;--ink2:#b4b4bc;--muted:#85858d;
--line:#2e2e34;--accent:#7aa7ff;--think:#232227;--tool:#1b2433;--eval:#17291f;--best:#4ade80;--warn:#fb923c;--code:#202024}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 10px}
.sub{color:var(--ink2);margin-bottom:18px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:16px 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.stat .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.stat .v{font-size:18px;font-weight:600}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
details>summary{cursor:pointer;color:var(--ink2)}pre,code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
pre{white-space:pre-wrap;word-break:break-word;background:var(--code);border-radius:8px;padding:10px;margin:8px 0 0}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 4px}
.chip{display:inline-block;min-width:34px;text-align:center;padding:3px 7px;border-radius:999px;border:1px solid var(--line);
background:var(--panel);color:var(--ink2);text-decoration:none;font-size:12.5px}.chip.best{border-color:var(--best);color:var(--best);font-weight:600}
.step{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin:14px 0;overflow:hidden}
.step>header{display:flex;justify-content:space-between;align-items:baseline;gap:10px;padding:10px 16px;border-bottom:1px solid var(--line)}
.step>header .t{font-weight:650}.step>header .r{color:var(--ink2);font-size:13px}
.items{padding:6px 16px 14px}.item{margin-top:10px;border-radius:8px;padding:8px 12px}
.think{background:var(--think)}.tool{background:var(--tool)}.eval{background:var(--eval);border-left:3px solid var(--best)}
.note{background:var(--think);border-left:3px solid var(--muted)}
.label{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:2px}
.txt{white-space:pre-wrap}.err{color:var(--warn)}.best-tag{color:var(--best);font-weight:650}
.kv{display:grid;grid-template-columns:110px 1fr;gap:2px 10px}.kv div:nth-child(odd){color:var(--muted)}
svg text{fill:var(--muted);font-size:11px}svg .grid{stroke:var(--line)}svg .curve{stroke:var(--accent);fill:none;stroke-width:2}
svg .pt{fill:var(--accent)}svg .pt.best{fill:var(--best)}svg a:hover .pt{r:6}
"""


def esc(x):
    return html.escape(str(x), quote=True)


def fmt(v, digits=4):
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def compact(obj, limit=None):
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return s if limit is None or len(s) <= limit else s[:limit] + " …"


def regret_svg(regrets, best_steps):
    w, h, l, r, t, b = 900, 230, 56, 12, 12, 30
    vals = [max(v, 1e-12) for v in regrets]
    lo, hi = math.floor(math.log10(min(vals))), math.ceil(math.log10(max(vals)))
    hi = hi if hi > lo else lo + 1
    n = len(vals)
    X = lambda i: l + (w - l - r) * (i / max(n - 1, 1))
    Y = lambda v: t + (h - t - b) * (hi - math.log10(v)) / (hi - lo)
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="simple regret per evaluation">']
    for e in range(lo, hi + 1):
        y = Y(10 ** e)
        parts.append(f'<line class="grid" x1="{l}" x2="{w - r}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text x="{l - 6}" y="{y + 4:.1f}" text-anchor="end">1e{e}</text>')
    for i in range(0, n, max(1, n // 10)):
        parts.append(f'<text x="{X(i):.1f}" y="{h - 10}" text-anchor="middle">{i + 1}</text>')
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
    parts.append(f'<polyline class="curve" points="{pts}"/>')
    for i, v in enumerate(vals):
        cls = "pt best" if (i + 1) in best_steps else "pt"
        parts.append(f'<a href="#eval-{i + 1}"><circle class="{cls}" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4">'
                     f'<title>evaluation {i + 1}: simple regret {v:.4g}</title></circle></a>')
    parts.append("</svg>")
    return "".join(parts)


def render(run):
    log, records = run["log"], run["records"]
    usage = run.get("usage") or {}
    model = usage.get("model", run["policy"])
    rec_by_t = {r["t"]: r for r in records}
    best_evals = {r["t"] for r in records if r.get("is_new_best")}
    brief = next((c["text"] for c in run.get("context", []) if c.get("source") == "problem"), "")

    steps = {}
    for e in log:
        steps.setdefault(e.get("step"), []).append(e)

    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Sara trajectory: {esc(run['problem'])}</title><style>{CSS}</style></head><body><div class='wrap'>"]
    out.append(f"<h1>Agentic BO trajectory · {esc(run['problem'])}</h1>"
               f"<div class='sub'>Sara driven by <b>{esc(model)}</b> · policy <code>{esc(run['policy'])}</code> · "
               f"seed {run['seed']} · budget {run['budget']} · {run['regret_mode']} regret</div>")
    stats = [("final simple regret", fmt(run["simple_regret"][-1])), ("best value", fmt(run.get("best_value"))),
             ("evaluations", run["n_evals"]), ("harness fallbacks", run["forced_commits"]),
             ("API calls", usage.get("api_calls", "–")), ("reasoning tokens", f"{usage.get('reasoning_tokens', 0):,}"),
             ("wall time", f"{run['wallclock_s'] / 60:.1f} min")]
    out.append("<div class='stats'>" + "".join(f"<div class='stat'><div class='k'>{esc(k)}</div><div class='v'>{esc(v)}</div></div>"
                                               for k, v in stats) + "</div>")
    counts = ", ".join(f"{k} ×{v}" for k, v in sorted(run["action_counts"].items(), key=lambda kv: -kv[1]))
    out.append(f"<div class='sub'>Tool use: {esc(counts)}</div>")

    out.append("<h2>Simple regret after each evaluation</h2><div class='panel'>"
               + regret_svg(run["simple_regret"], best_evals)
               + "<div class='sub' style='margin:6px 0 0'>Green points are new incumbents; click a point to jump to that evaluation.</div></div>")
    out.append("<h2>Steps</h2><div class='chips'>" + "".join(
        f"<a class='chip{' best' if t in best_evals else ''}' href='#eval-{t}'>{t}</a>" for t in sorted(rec_by_t)) + "</div>")
    out.append(f"<h2>Problem brief (what the agent sees)</h2><details class='panel'><summary>Show brief</summary><pre>{esc(brief)}</pre></details>")
    if run.get("notebook"):
        notes = "".join(f"<div class='item note'><div class='label'>note · step {n['step']}</div><div class='txt'>{esc(n['text'])}</div></div>"
                        for n in run["notebook"])
        out.append(f"<h2>Lab notebook</h2><details class='panel'><summary>{len(run['notebook'])} notes written by the agent</summary>{notes}</details>")

    for step in sorted(k for k in steps if k is not None):
        items, eval_rec = [], None
        for e in steps[step]:
            if e.get("action_class") == "reasoning":
                kind = e.get("kind", "thinking")
                label = {"thinking": "reasoning", "text": "message", "policy": "harness note"}.get(kind, kind)
                items.append(f"<div class='item {'note' if kind == 'policy' else 'think'}'><div class='label'>{esc(label)}</div>"
                             f"<div class='txt'>{esc(e.get('text', '').strip())}</div></div>")
                continue
            tool, ok = e.get("tool"), e.get("ok")
            if tool == "evaluate":
                a = e.get("args") or {}
                res = e.get("result") or {}
                eval_rec = rec_by_t.get(res.get("t"))
                tag = " <span class='best-tag'>★ new best</span>" if res.get("is_new_best") else ""
                forced = " (harness fallback)" if e.get("forced") else ""
                rows = [("rationale", a.get("rationale", "")), ("expectation", a.get("expectation", "—")),
                        ("config", compact(a.get("config"))), ("observed", compact(res.get("observed")))]
                if eval_rec:
                    rows.append(("regret after", fmt(eval_rec.get("regret"))))
                body = "".join(f"<div>{esc(k)}</div><div>{esc(v)}</div>" for k, v in rows)
                anchor = f" id='eval-{res.get('t')}'" if res.get("t") else ""
                err = "" if ok else f"<div class='err'>{esc(e.get('error'))}</div>"
                items.append(f"<div class='item eval'{anchor}><div class='label'>evaluate{esc(forced)}{tag}</div>"
                             f"<div class='kv'>{body}</div>{err}</div>")
            else:
                res = e.get("result") if ok else e.get("error")
                status = "ok" if ok else "<span class='err'>error</span>"
                items.append(f"<div class='item tool'><div class='label'>tool · {esc(tool)} · {status}</div>"
                             f"<code>{esc(compact(e.get('args') or {}, 400))}</code>"
                             f"<details><summary>result</summary><pre>{esc(json.dumps(res, indent=1, ensure_ascii=False)[:20000])}</pre></details></div>")
        head_r = ""
        if eval_rec:
            obs = eval_rec.get("observed") or {}
            head_r = (f"y = {fmt(next(iter(obs.values()), None))} · regret {fmt(run['simple_regret'][eval_rec['t'] - 1])}"
                      + (" · <span class='best-tag'>★</span>" if eval_rec.get("is_new_best") else ""))
        out.append(f"<section class='step'><header><span class='t'>Step {step}</span><span class='r'>{head_r}</span></header>"
                   f"<div class='items'>{''.join(items)}</div></section>")
    out.append("<div class='sub' style='margin-top:30px'>Generated from the campaign result file by "
               "results/paper_protocol_T30/trajectory_html.py (agentic_bo).</div></div></body></html>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    run = json.load(open(a.run))
    out = a.out or os.path.splitext(a.run)[0] + ".html"
    open(out, "w", encoding="utf-8").write(render(run))
    print(out)


if __name__ == "__main__":
    main()
