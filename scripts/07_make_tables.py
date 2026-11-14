#!/usr/bin/env python3
"""Build results tables, paired statistics and figures from pipeline outputs. CPU only, no network.

python scripts/07_make_tables.py --runs results/hfjob --out results
Expects <runs>/run/baseline, <runs>/run2/{after,naive_fitted_gate,sweep}, optional <runs>/run3/{ablation_ungated,retention_summary.json}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

LANGS = ["eng", "hau", "ibo", "yor"]
NAMES = {"eng": "English", "hau": "Hausa", "ibo": "Igbo", "yor": "Yoruba", None: "All four"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b (1->0) and c (0->1)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_boot(x, y, iters=5000, seed=0):
    """95% bootstrap CI of mean(y) - mean(x) for paired 0/1 lists."""
    rnd = random.Random(seed)
    n = len(x)
    ds = []
    for _ in range(iters):
        idx = [rnd.randrange(n) for _ in range(n)]
        ds.append(sum(y[i] - x[i] for i in idx) / n)
    ds.sort()
    return ds[int(0.025 * iters)], ds[int(0.975 * iters)]


def load(d: Path):
    return {r["sample_id"]: r for r in (json.loads(l) for l in (d / "records.jsonl").open(encoding="utf-8"))}


def subset(rows, lang, harmful):
    return [r for r in rows.values() if (lang is None or r["language"] == lang) and ((r["category"] != "benign") == harmful)]


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/hfjob"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    out = a.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    R = a.runs
    stages = {"Baseline": load(R / "run" / "baseline")}
    if (R / "run2" / "after" / "records.jsonl").exists():
        stages["CAST tuned gate (submitted)"] = load(R / "run2" / "after")
    if (R / "run2" / "naive_fitted_gate" / "records.jsonl").exists():
        stages["CAST default gate, alpha 0.5"] = load(R / "run2" / "naive_fitted_gate")
    if (R / "run3" / "ablation_ungated" / "records.jsonl").exists():
        stages["Ungated CAA, alpha 0.25"] = load(R / "run3" / "ablation_ungated")
    base = stages["Baseline"]
    md, rows_csv = [], []

    # Baseline gap table
    eng_h = subset(base, "eng", True)
    eng_hc = sum(r["complied"] for r in eng_h) / len(eng_h)
    md += ["### Table 1. Baseline harmful compliance and over-refusal by language (held-out eval split)\n",
           "| Language | Harmful compliance [95% CI] | Gap vs English | Direct harm | Culturally adapted | Over-refusal on benign [95% CI] | n harmful / benign |",
           "|---|---|---|---|---|---|---|"]
    for l in LANGS:
        h, b = subset(base, l, True), subset(base, l, False)
        k, kb = sum(r["complied"] for r in h), sum(r["refused"] for r in b)
        hh = [r for r in h if r["category"] == "harm"]
        hj = [r for r in h if r["category"] == "jailbreak"]
        lo, hi = wilson(k, len(h))
        blo, bhi = wilson(kb, len(b))
        gap = "" if l == "eng" else f"+{100 * (k / len(h) - eng_hc):.1f} pts"
        md.append(f"| {NAMES[l]} | {pct(k / len(h))} [{100*lo:.0f}, {100*hi:.0f}] | {gap} | {pct(sum(r['complied'] for r in hh)/len(hh))} | "
                  f"{pct(sum(r['complied'] for r in hj)/len(hj))} | {pct(kb / len(b))} [{100*blo:.0f}, {100*bhi:.0f}] | {len(h)} / {len(b)} |")
    md.append("")

    # All stages table
    names = list(stages)
    md += ["### Table 2. Same model before and after, same held-out prompts (harmful compliance / over-refusal, lower is better)\n",
           "| Language | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for l in LANGS + [None]:
        cells = []
        for s in names:
            h, b = subset(stages[s], l, True), subset(stages[s], l, False)
            hc, orr = sum(r["complied"] for r in h) / len(h), sum(r["refused"] for r in b) / len(b)
            cells.append(f"{pct(hc)} / {pct(orr)}")
            rows_csv.append({"stage": s, "language": l or "all", "harmful_compliance": round(hc, 4),
                             "over_refusal": round(orr, 4), "n_harmful": len(h), "n_benign": len(b)})
        md.append(f"| {NAMES[l]} | " + " | ".join(cells) + " |")
    md.append("")

    # Paired stats baseline vs submitted
    sub = "CAST tuned gate (submitted)"
    stats_json = {}
    if sub in stages:
        md += ["### Table 3. Paired change, baseline to submitted repair (same prompts; 95% paired bootstrap CI; exact McNemar p)\n",
               "| Language | Harmful compliance change [95% CI] | McNemar p | Over-refusal change [95% CI] | McNemar p |",
               "|---|---|---|---|---|"]
        for l in LANGS + [None]:
            res = {}
            for harmful, key in ((True, "complied"), (False, "refused")):
                ids = [r["sample_id"] for r in subset(base, l, harmful)]
                x = [int(base[i][key]) for i in ids]
                y = [int(stages[sub][i][key]) for i in ids]
                b_ = sum(1 for i, j in zip(x, y) if i == 1 and j == 0)
                c_ = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)
                lo, hi = paired_boot(x, y)
                res[key] = {"delta": (sum(y) - sum(x)) / len(x), "ci": [lo, hi], "p": mcnemar_exact(b_, c_), "b": b_, "c": c_}
            stats_json[l or "all"] = res
            f = lambda d: f"{100*d['delta']:+.1f} pts [{100*d['ci'][0]:+.1f}, {100*d['ci'][1]:+.1f}]"
            md.append(f"| {NAMES[l]} | {f(res['complied'])} | {res['complied']['p']:.2g} | {f(res['refused'])} | {res['refused']['p']:.2g} |")
        md.append("")
        g = json.loads((R / "run2" / "after" / "summary.json").read_text()).get("gate", {})
        md += ["### Table 4. Where the gate fires on the eval split (submitted repair)\n",
               "| Language | Fires on harmful | Fires on benign |", "|---|---|---|"]
        for l in LANGS:
            md.append(f"| {NAMES[l]} | {pct(g[l]['gate_fire_rate_harmful'])} | {pct(g[l]['gate_fire_rate_benign'])} |")
        md.append("")

    # Belebele
    ret = R / "run3" / "retention_summary.json"
    bel = None
    if ret.exists():
        bel = json.loads(ret.read_text())["belebele"]
        md += ["### Table 5. Capability retention: Belebele reading comprehension accuracy (200 parallel questions per language, higher is better)\n",
               "| Language | Baseline | CAST tuned gate (submitted) | Ungated CAA | Gate fires on Belebele |", "|---|---|---|---|---|"]
        for l in LANGS:
            bb, cc, uu = bel["base"][l], bel["cast"][l], bel["ungated"][l]
            md.append(f"| {NAMES[l]} | {pct(bb['accuracy'])} | {pct(cc['accuracy'])} | {pct(uu['accuracy'])} | {pct(cc.get('gate_fire_rate'))} |")
        md.append("")

    # Code-mixed / cross-lingual wrappers
    cm = R / "run4" / "codemix_summary.json"
    if cm.exists():
        c = json.loads(cm.read_text())["stages"]
        md += ["### Table 7. Code-mixed and cross-lingual wrappers (eval split, harmful compliance / over-refusal, base → submitted CAST)\n",
               "`en_frame`: local-language prompt inside an English instruction (\"Please answer the following in English.\"). "
               "`ask_local`: the English source prompt plus \"Please answer in <language>.\" Plain prompts from Table 2 for reference.\n",
               "| Language | Plain prompt | en_frame | ask_local |", "|---|---|---|---|"]
        for l in ["hau", "ibo", "yor"]:
            h, b = subset(base, l, True), subset(base, l, False)
            plain_b = f"{pct(sum(r['complied'] for r in h)/len(h))} / {pct(sum(r['refused'] for r in b)/len(b))}"
            if sub in stages:
                h2, b2 = subset(stages[sub], l, True), subset(stages[sub], l, False)
                plain_b += f" → {pct(sum(r['complied'] for r in h2)/len(h2))} / {pct(sum(r['refused'] for r in b2)/len(b2))}"
            cells = []
            for var in ("en_frame", "ask_local"):
                x, y = c["base"][f"{var}|{l}"], c["cast"][f"{var}|{l}"]
                cells.append(f"{pct(x['harmful_compliance_rate'])} / {pct(x['over_refusal_rate'])} → {pct(y['harmful_compliance_rate'])} / {pct(y['over_refusal_rate'])}")
            md.append(f"| {NAMES[l]} | {plain_b} | " + " | ".join(cells) + " |")
        md.append("")

    # Sweep table
    sw = R / "run2" / "sweep" / "sweep.json"
    sweep = json.loads(sw.read_text()) if sw.exists() else None
    if sweep:
        md += ["### Table 6. Calibration-split sweep used to choose the repair (in-sample; chosen before any eval-split steering)\n",
               "| Gate | alpha | Harmful compliance | Over-refusal | Eligible |", "|---|---|---|---|---|",
               f"| none (baseline) | 0 | {pct(sweep['base']['harmful_compliance_rate'])} | {pct(sweep['base']['over_refusal_rate'])} | |"]
        for k, v in sweep["grid"].items():
            gname, al = k.split("|")
            md.append(f"| {gname} | {al} | {pct(v['harmful_compliance_rate'])} | {pct(v['over_refusal_rate'])} | {'yes' if v['eligible'] else 'no'} |")
        md.append(f"\nChosen: gate {sweep['chosen']['gate']}, alpha {sweep['chosen']['alpha']}. Rule: {sweep['rule']}.\n")

    (out / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "paired_stats.json").write_text(json.dumps(stats_json, indent=2))
    with (out / "metrics_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    # Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.8 / len(names)
    colors = ["#8a8a8a", "#1f77b4", "#ff7f0e", "#d62728"]
    for ax, harmful, key, title in ((axes[0], True, "complied", "Harmful compliance (lower is better)"),
                                    (axes[1], False, "refused", "Over-refusal on benign prompts (lower is better)")):
        for si, s in enumerate(names):
            vals, errs_lo, errs_hi = [], [], []
            for l in LANGS:
                rr = subset(stages[s], l, harmful)
                k = sum(r[key] for r in rr)
                p = k / len(rr)
                lo, hi = wilson(k, len(rr))
                vals.append(100 * p)
                errs_lo.append(100 * (p - lo))
                errs_hi.append(100 * (hi - p))
            xs = [i + (si - (len(names) - 1) / 2) * width for i in range(len(LANGS))]
            ax.bar(xs, vals, width, yerr=[errs_lo, errs_hi], capsize=2, label=s, color=colors[si % len(colors)])
        ax.set_xticks(range(len(LANGS)))
        ax.set_xticklabels([NAMES[l] for l in LANGS])
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.set_ylim(0, 100)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Tiny Aya Earth on TukaBench held-out eval split (95% Wilson CIs)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "before_after_by_language.png", dpi=150)
    fig.savefig(out / "figures" / "before_after_by_language.svg")

    if sweep:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter([100 * sweep["base"]["over_refusal_rate"]], [100 * sweep["base"]["harmful_compliance_rate"]], c="k", label="baseline")
        for k, v in sweep["grid"].items():
            ax.scatter(100 * v["over_refusal_rate"], 100 * v["harmful_compliance_rate"], c="#1f77b4" if v["eligible"] else "#bbbbbb")
            ax.annotate(k.replace("|", " a="), (100 * v["over_refusal_rate"], 100 * v["harmful_compliance_rate"]), fontsize=7)
        ax.set_xlabel("Over-refusal on benign (%)")
        ax.set_ylabel("Harmful compliance (%)")
        ax.set_title("Calibration sweep: safety vs over-refusal (grey = ineligible)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figures" / "calibration_tradeoff.png", dpi=150)
        fig.savefig(out / "figures" / "calibration_tradeoff.svg")
    print("\n".join(md))


if __name__ == "__main__":
    main()
