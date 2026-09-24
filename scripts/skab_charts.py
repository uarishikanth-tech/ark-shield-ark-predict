"""
Charts for the SKAB results (slides / README), from the dashboard results file.

    python scripts/skab_charts.py app/services/anomaly/results/skab_results.json docs/img/skab
Needs matplotlib (pip install matplotlib).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG, PANEL, GRID, TEXT, DIM, FAINT = "#0D1117", "#161D27", "#28323F", "#E7EDF4", "#93A2B4", "#5B6B7D"
C = {"fixed": "#E8703B", "unsup": "#A78BFA", "sup": "#4E8CFF", "ours": "#33D482", "lb": "#5B6B7D",
     "urgent": "#F0473C", "monitor": "#F0A83B", "truth": "#8B9BB4", "line": "#4E8CFF"}
FAM = {"baseline": C["fixed"], "statistical": "#8B9BB4", "classical ML": "#4E8CFF", "deep learning": "#A78BFA",
       "ensemble": "#2DD4BF", "supervised ML": "#E879F9", "ours": "#FACC15", "ours (hybrid)": "#33D482"}

plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": DIM,
                     "xtick.color": DIM, "ytick.color": DIM, "text.color": TEXT, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def headline(res, out):
    R = {r["key"]: r for r in res["rows"]}
    systems = [("Fixed limit\n(today)", R["fixed"], C["fixed"]),
               ("Conv-AE\n(deep learning)", R["conv_ae+"], C["unsup"]),
               ("v2 URGENT\nalarms only", R["supervised_past"], C["sup"]),
               ("ARK v2\n(all alarms)", R["hybrid_past"], C["ours"])]
    metrics = [("F1 on real faults  ↑", "f1", "{:.2f}", 1.0), ("False-alarm rate %  ↓", "far", "{:.0f}%", None),
               ("False-alarm episodes / hour  ↓", "false_alarm_episodes_per_h", "{:.0f}", None),
               ("Faults caught (of 34)  ↑", "faults_caught", "{:.0f}", 34)]
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
    for ax, (title, k, fmt, top) in zip(axs, metrics):
        vals = [s[1][k] for s in systems]
        bars = ax.bar(range(4), vals, color=[s[2] for s in systems], width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color=TEXT)
        ax.set_xticks(range(4), [s[0] for s in systems], fontsize=10)
        ax.set_title(title, fontsize=13, color=TEXT, loc="left", pad=14)
        ax.set_ylim(0, (top or max(vals)) * 1.18)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9, colors=FAINT)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def scatter(res, out):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for r in res["leaderboard"]:
        ax.scatter(r["far"], r["mar"], s=50, facecolor="none", edgecolor=FAINT, lw=1.5, zorder=2)
        ax.annotate(r["label"], (r["far"], r["mar"]), xytext=(6, -3), textcoords="offset points", fontsize=8.5, color=FAINT)
    R = {r["key"]: r for r in res["rows"]}
    for r in res["rows"]:
        if r["key"].endswith("+") and r["key"][:-1] in R:
            p = R[r["key"][:-1]]
            ax.annotate("", (r["far"], r["mar"]), (p["far"], p["mar"]),
                        arrowprops=dict(arrowstyle="->", color=FAM.get(r["family"], DIM), lw=0.8, alpha=0.35))
    for r in res["rows"]:
        c = FAM.get(r["family"], DIM)
        big = r["family"].startswith("ours")
        filled = r["mode"] != "plain"
        ax.scatter(r["far"], r["mar"], s=150 if big else 60, facecolor=c if filled else BG, edgecolor=c, lw=2, zorder=3,
                   marker="*" if r["key"] == "hybrid_past" else "o")
    h, f = R.get("hybrid_past"), R["fixed"]
    if h:
        ax.annotate("ARK Predict v2", (h["far"], h["mar"]), xytext=(10, -18), textcoords="offset points", fontsize=11,
                    color=C["ours"], fontweight="bold")
    ax.annotate("Fixed 3σ limit (today)", (f["far"], f["mar"]), xytext=(8, 6), textcoords="offset points", fontsize=11,
                color=C["fixed"], fontweight="bold")
    if "hybrid_lfo" in R:
        r = R["hybrid_lfo"]
        ax.annotate("v2 on never-seen fault types", (r["far"], r["mar"]), xytext=(12, -4), textcoords="offset points",
                    fontsize=9.5, color=C["ours"])
    if "supervised_past" in R:
        r = R["supervised_past"]
        ax.annotate("v2 URGENT tier only\n(supervised)", (r["far"], r["mar"]), xytext=(-6, -26), textcoords="offset points",
                    fontsize=9.5, color=FAM["supervised ML"], ha="left")
    if "ark" in R:
        r = R["ark"]
        ax.annotate("ARK v1 (live)", (r["far"], r["mar"]), xytext=(9, 4), textcoords="offset points", fontsize=9.5,
                    color=FAM["ours"])
    from matplotlib.lines import Line2D
    fams = []
    for r in res["rows"]:
        if r["family"] not in fams:
            fams.append(r["family"])
    handles = [Line2D([], [], marker="o", ls="", color=FAM.get(fm, DIM), markersize=8, label=fm) for fm in fams]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=FAINT, markersize=8,
                          label="published leaderboard"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.86), frameon=False, fontsize=9)
    ax.set_xlabel("False-alarm rate %  (normal seconds flagged)")
    ax.set_ylabel("Missed-alarm rate %  (fault seconds missed)")
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 90)
    ax.grid(color=GRID, lw=0.5)
    ax.text(73, 87, "lower-left is better\nhollow = plain model · filled = improved\ngrey = published SKAB leaderboard",
            ha="right", va="top", fontsize=9, color=DIM)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def auc(res, out, labels):
    a = res["auc"]
    keys = [k for k in labels if k in a and len(a[k]) > 1]
    raw = [a[k]["raw"] for k in keys]
    rob = [max(v for kk, v in a[k].items() if kk != "raw") for k in keys]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = range(len(keys))
    ax.barh([i + 0.2 for i in y], raw, height=0.38, color=GRID, label="raw sensor values")
    ax.barh([i - 0.2 for i in y], rob, height=0.38, color=C["ours"], label="drift-robust inputs (ours)")
    for i, (r0, r1) in enumerate(zip(raw, rob)):
        ax.text(r1 + 0.004, i - 0.2, f"{r1:.3f}", va="center", fontsize=10, color=TEXT)
        ax.text(r0 + 0.004, i + 0.2, f"{r0:.3f}", va="center", fontsize=9, color=DIM)
    if "supervised" in a:
        ax.axvline(a["supervised"]["raw"], color=C["sup"], ls="--", lw=1.2)
        ax.text(a["supervised"]["raw"], len(keys) - 0.4, f" supervised {a['supervised']['raw']:.3f}", color=C["sup"], fontsize=10)
    ax.set_yticks(list(y), [labels[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0.7, 0.95)
    ax.set_xlabel("ROC-AUC (threshold-free: 1.0 = perfect ranking of faulty vs normal seconds)")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def replay(res, exp, out):
    r = res["replay"][exp]
    n = r["rows"]
    names = list(r["sensors"])
    tracks = [("Labelled fault", r["fault"], C["truth"]), ("Fixed 3σ limit", r["alarms"].get("fixed", []), C["fixed"]),
              ("ARK v2 · URGENT", r["alarms"].get("urgent", []), C["urgent"]),
              ("ARK v2 · MONITOR", r["alarms"].get("monitor", []), C["monitor"])]
    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(len(names) + 1, 1, height_ratios=[1] * len(names) + [1.25], hspace=0.12)
    for k, s in enumerate(names):
        ax = fig.add_subplot(gs[k])
        ax.plot(r["sensors"][s], color=C["line"], lw=0.9)
        ax.axvspan(0, r["train_rows"], color=FAINT, alpha=0.12, lw=0)
        for a, b in r["fault"]:
            ax.axvspan(a, b, color=C["urgent"], alpha=0.12, lw=0)
        ax.set_xlim(0, n)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(s.replace("RMS", "").replace("Volume ", ""), rotation=0, ha="right", va="center", fontsize=10.5, color=DIM)
        if k == 0:
            ax.text(r["train_rows"] / 2, ax.get_ylim()[1], "training (healthy)", ha="center", va="bottom", fontsize=9, color=DIM)
    ax = fig.add_subplot(gs[-1])
    for i, (lab, segs, c) in enumerate(tracks):
        yy = len(tracks) - 1 - i
        ax.broken_barh([(0, n)], (yy + 0.15, 0.7), color=PANEL)
        ax.broken_barh([(a, b - a) for a, b in segs], (yy + 0.15, 0.7), color=c)
    ax.set_yticks([len(tracks) - 1 - i + 0.5 for i in range(len(tracks))], [t[0] for t in tracks], fontsize=10.5)
    ax.set_xlim(0, n)
    ax.set_xlabel("seconds")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    fig.suptitle(f"Real SKAB experiment {exp}: fixed limit raised {len(r['alarms'].get('fixed', []))} separate alarms, "
                 f"ARK Predict v2 raised {len(r['alarms'].get('urgent', []))} urgent + {len(r['alarms'].get('monitor', []))} monitor",
                 fontsize=13, color=TEXT, x=0.01, ha="left")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    res = json.load(open(src))
    labels = {"fixed": "Fixed 3σ limit", "pca": "PCA T²+Q", "iforest": "Isolation Forest", "lof": "Local Outlier Factor",
              "ocsvm": "One-Class SVM", "dense_ae": "Dense autoencoder", "conv_ae": "Conv-1D autoencoder",
              "lstm_ae": "LSTM autoencoder", "ensemble": "Ensemble"}
    headline(res, os.path.join(out, "headline.png"))
    scatter(res, os.path.join(out, "far_vs_mar.png"))
    auc(res, os.path.join(out, "auc_drift_robust.png"), labels)
    for exp in sys.argv[3:] or ["valve1/1"]:
        replay(res, exp, os.path.join(out, f"replay_{exp.replace('/', '_')}.png"))
    print("charts written to", out)


if __name__ == "__main__":
    main()
