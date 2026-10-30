#!/usr/bin/env python3
"""
HTML report of one validation run: ci/make_report.py <run dir> [--history <csv>]

Reads <run dir>/results.json (written by tests/conftest.py) and writes <run dir>/report.html: summary, known issues
(still present / fixed), one section per module with every test, its metrics against tolerance and its figures, and
the failure messages. With --history, the trend metrics and the outcome counts are appended to the csv and the last
runs of the same tier are plotted (trends.png).
"""
import argparse
import csv
import html
import json
import os
import re
from collections import Counter, defaultdict

OUTCOMES = ['failed', 'error', 'xpassed', 'passed', 'xfailed', 'skipped']
LABEL = dict(failed='FAILED', error='ERROR', xpassed='KNOWN ISSUE FIXED?', passed='passed', xfailed='known issue',
             skipped='skipped')
CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e2e2e6; --card:#f6f6f8; --ok:#1a7f37; --bad:#c62828;
        --warn:#b26a00; --info:#3a5ccc; }
@media (prefers-color-scheme: dark) { :root { --bg:#16161a; --fg:#e8e8ea; --muted:#9a9aa2; --line:#2c2c33;
        --card:#1f1f25; --ok:#4cc26a; --bad:#ff6b6b; --warn:#f0a83a; --info:#8aa2ff; } }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; margin:0 auto; max-width:1200px; padding:16px; }
h1 { font-size:22px; margin:8px 0 4px; } h2 { font-size:17px; margin:28px 0 8px; border-bottom:1px solid var(--line); }
.muted { color:var(--muted); } table { border-collapse:collapse; width:100%; margin:6px 0 14px; }
td, th { border-bottom:1px solid var(--line); padding:5px 6px; vertical-align:top; text-align:left; }
th { font-weight:600; color:var(--muted); font-size:12px; }
.badge { display:inline-block; padding:1px 7px; border-radius:9px; font-size:12px; font-weight:600; white-space:nowrap; }
.passed { color:var(--ok); } .failed, .error { color:var(--bad); } .xpassed { color:var(--warn); }
.xfailed { color:var(--info); } .skipped { color:var(--muted); }
.cards { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0; }
.card { background:var(--card); border-radius:8px; padding:8px 14px; min-width:110px; }
.card b { font-size:20px; display:block; } .m { font-family:ui-monospace,monospace; font-size:12px; }
.bad { color:var(--bad); } .good { color:var(--ok); }
img.thumb { max-height:190px; max-width:100%; border:1px solid var(--line); border-radius:4px; margin:4px 4px 0 0; background:#fff; }
pre { background:var(--card); padding:8px; overflow-x:auto; font-size:12px; }
details summary { cursor:pointer; }
"""


def esc(x):
    return html.escape(str(x))


def fmt(v):
    if isinstance(v, list):
        if v and any(isinstance(x, list) for x in v):          # nested arrays: summarise the flattened values
            import numpy as np
            flat = np.asarray(v, dtype=float).ravel()
            return f'[{flat.min():.4g} .. {flat.max():.4g}] ({flat.size})'
        if len(v) <= 4:
            return '[' + ', '.join(fmt(x) for x in v) + ']'
        return f'[{min(v):.4g} .. {max(v):.4g}] ({len(v)})'
    if isinstance(v, float):
        return f'{v:.4g}'
    return esc(v)


def criterion(m):
    parts = []
    if m.get('expected') is not None:
        tol = []
        if m.get('atol'):
            tol.append(f'± {fmt(m["atol"])}')
        if m.get('rtol'):
            tol.append(f'± {fmt(m["rtol"])} (rel)')
        parts.append(f'{fmt(m["expected"])} {" + ".join(tol)}')
    if m.get('lo') is not None or m.get('hi') is not None:
        parts.append(f'[{fmt(m.get("lo")) if m.get("lo") is not None else "-inf"}, '
                     f'{fmt(m.get("hi")) if m.get("hi") is not None else "inf"}]')
    return ' & '.join(parts)


def parse_issue(reason):
    m = re.match(r'\[([^\]]+)\]\s*(.*?)(?:\s*\(([^()]*)\))?$', reason or '')
    return (m.group(1), m.group(2), m.group(3) or '') if m else (reason, '', '')


def update_history(path, info, tests):
    rows = []
    base = dict(date=info['date'], sha=info['sha'], dirty=info['dirty'], tier=info['tier'])
    counts = Counter(t['outcome'] for t in tests)
    for o in OUTCOMES:
        rows.append({**base, 'test': '_summary', 'metric': o, 'value': counts.get(o, 0)})
    for t in tests:
        for m in t['metrics']:
            if m.get('trend') and isinstance(m['value'], (int, float)):
                rows.append({**base, 'test': t['nodeid'], 'metric': m['name'], 'value': m['value']})
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def plot_trends(path, tier, out_png, last=30):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['tier'] == tier and r['test'] != '_summary':
                series[(r['test'].split('::')[-1], r['metric'])].append((r['date'], float(r['value'])))
    series = {k: v[-last:] for k, v in series.items() if len(v) >= 2}
    if not series:
        return False
    n = len(series)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.4 * rows), squeeze=False)
    for ax, ((test, metric), pts) in zip(axes.ravel(), sorted(series.items())):
        ax.plot(range(len(pts)), [p[1] for p in pts], 'o-', ms=3)
        ax.set_title(f'{test}\n{metric}', fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description='HTML report of a SAOS validation run')
    ap.add_argument('run_dir')
    ap.add_argument('--history', default=None, help='csv with the trend metrics of all runs (appended)')
    a = ap.parse_args()
    with open(os.path.join(a.run_dir, 'results.json')) as f:
        res = json.load(f)
    info, tests = res['info'], res['tests']
    counts = Counter(t['outcome'] for t in tests)

    trends = False
    if a.history:
        update_history(a.history, info, tests)
        trends = plot_trends(a.history, info['tier'], os.path.join(a.run_dir, 'trends.png'))

    h = [f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>SAOS validation {esc(info["tier"])} {esc(info["sha"])}</title><style>{CSS}</style></head><body>']
    verdict = 'FAIL' if (counts.get('failed') or counts.get('error') or counts.get('xpassed') or info.get('exitstatus', 0) != 0
                         or not tests) else 'PASS'
    h.append(f'<h1>SAOS validation: <span class="{"bad" if verdict == "FAIL" else "good"}">{verdict}</span></h1>')
    h.append(f'<div class="muted">tier <b>{esc(info["tier"])}</b> · commit {esc(info["sha"])}{" (dirty)" if info["dirty"] else ""} '
             f'on {esc(info["branch"])} · {esc(info["date"])} · {esc(info["host"])} · Python {esc(info["python"])}, '
             f'numpy {esc(info["numpy"])}, torch {esc(info["torch"])} '
             f'({"GPU " + esc(info["gpu"]) if info["cuda"] else "CPU"}) · {info["duration"]:.0f} s</div>')
    h.append('<div class="cards">' + ''.join(
        f'<div class="card"><b class="{o}">{counts.get(o, 0)}</b>{LABEL[o]}</div>' for o in OUTCOMES if counts.get(o) or o in ('passed', 'failed')) + '</div>')
    if info.get('exitstatus', 0) not in (0, 1):
        h.append(f'<p class="bad">pytest session ended abnormally (exit status {info.get("exitstatus")}): '
                 'interrupted, collection error or no tests selected. The results below are incomplete.</p>')
    h.append('<p class="muted">Tests labelled with a known issue state the physically correct behaviour and fail while '
             'the SAOS bug exists. A labelled test that passes means the bug is no longer reproduced and the label can be removed.</p>')

    issues = defaultdict(list)
    for t in tests:
        if t.get('known_issue'):
            issues[parse_issue(t['known_issue'])].append(t)
    if issues:
        h.append('<h2>Known issues</h2><table><tr><th>id</th><th>description</th><th>location</th><th>tests</th><th>status</th></tr>')
        for (iid, desc, loc), ts in sorted(issues.items()):
            c = Counter(t['outcome'] for t in ts)
            parts = []
            if c.get('failed') or c.get('error'):
                parts.append(f'<span class="failed">reproduced in {c.get("failed", 0) + c.get("error", 0)}/{len(ts)}</span>')
            if c.get('passed'):
                parts.append(f'<span class="xpassed">not reproduced in {c["passed"]}/{len(ts)}: fixed?</span>')
            if c.get('skipped'):
                parts.append(f'<span class="skipped">not run in {c["skipped"]}/{len(ts)}</span>')
            status = '<br>'.join(parts)
            h.append(f'<tr><td class="m">{esc(iid)}</td><td>{esc(desc)}</td><td class="m">{esc(loc)}</td>'
                     f'<td class="m">{"<br>".join(esc(t["name"]) for t in ts)}</td><td>{status}</td></tr>')
        h.append('</table>')

    if trends:
        h.append('<h2>Trends (last runs of this tier)</h2><img class="thumb" style="max-height:none" src="trends.png">')

    by_module = defaultdict(list)
    for t in tests:
        by_module[t['module']].append(t)
    for mod in sorted(by_module):
        ts = by_module[mod]
        c = Counter(t['outcome'] for t in ts)
        h.append(f'<h2>{esc(mod)} <span class="muted" style="font-size:13px">'
                 + ' · '.join(f'{c[o]} {LABEL[o]}' for o in OUTCOMES if c.get(o)) + '</span></h2>')
        h.append('<table><tr><th style="width:24%">test</th><th>outcome</th><th>metrics (value · criterion)</th><th>figures</th></tr>')
        for t in ts:
            ms = []
            for m in t['metrics']:
                cls = '' if m['ok'] is None else ('good' if m['ok'] else 'bad')
                crit = criterion(m)
                ms.append(f'<div class="m"><span class="{cls}">{"✓" if m["ok"] else ("✗" if m["ok"] is False else "·")}</span> '
                          f'{esc(m["name"])} = <b>{fmt(m["value"])}</b> {esc(m.get("unit", ""))}'
                          f'{" · " + esc(crit) if crit else ""}</div>')
            for n in t.get('notes', []):
                ms.append(f'<div class="muted">{esc(n)}</div>')
            figs = ''.join(f'<a href="{esc(p)}"><img class="thumb" src="{esc(p)}"></a>' for p in t.get('figures', []))
            issue = f'<div class="m">{esc(parse_issue(t["known_issue"])[0])}</div>' if t.get('known_issue') else ''
            msg = (f'<details><summary class="{t["outcome"]}">message</summary><pre>{esc(t["message"])}</pre></details>'
                   if t['outcome'] in ('failed', 'error', 'xpassed', 'skipped') and t.get('message') else '')
            h.append(f'<tr><td><b class="m">{esc(t["name"])}</b><div class="muted">{esc(t["doc"])}</div></td>'
                     f'<td><span class="badge {t["outcome"]}">{LABEL[t["outcome"]]}</span>{issue}'
                     f'<div class="muted">{t["duration"]:.2f} s</div>{msg}</td><td>{"".join(ms)}</td><td>{figs}</td></tr>')
        h.append('</table>')
    h.append('</body></html>')
    with open(os.path.join(a.run_dir, 'report.html'), 'w') as f:
        f.write('\n'.join(h))
    print(f'{os.path.join(a.run_dir, "report.html")}: ' + ', '.join(f'{counts[o]} {o}' for o in OUTCOMES if counts.get(o)))


if __name__ == '__main__':
    main()
