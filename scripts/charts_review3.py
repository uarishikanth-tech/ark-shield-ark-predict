"""Charts for Review 3 (called by scripts/review3.py). Needs matplotlib."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG, PANEL, GRID, TEXT, DIM, FAINT = "#0D1117", "#161D27", "#28323F", "#E7EDF4", "#93A2B4", "#5B6B7D"
FIXED, UNSUP, SUP, OURS, V1 = "#E8703B", "#A78BFA", "#4E8CFF", "#33D482", "#8B9BB4"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": DIM,
                     "xtick.color": DIM, "ytick.color": DIM, "text.color": TEXT, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
LABEL = {"drift": "sensor drift", "dropout": "sensor dropout", "stuck": "stuck sensor", "bearing_wear": "bearing wear",
         "hydraulic_leak": "hydraulic leak", "battery_fade": "battery fade", "oscillation": "oscillation",
         "overload": "overload lift"}


def classifier_accuracy(ev, out):
    items = [(LABEL.get(k, k), v["correct"], v["faults"]) for k, v in ev["perClass"].items()]
    items.sort(key=lambda x: x[1] / x[2])
    fig, ax = plt.subplots(figsize=(9, 5))
    y = range(len(items))
    ax.barh(list(y), [f for _, _, f in items], color=GRID, height=0.6, label="faults tested")
    ax.barh(list(y), [c for _, c, _ in items], color=OURS, height=0.6, label="named correctly")
    for i, (_, c, f) in enumerate(items):
        ax.text(f + 0.15, i, f"{c}/{f}", va="center", fontsize=11, color=TEXT)
    ax.set_yticks(list(y), [n for n, _, _ in items])
    ax.set_xlabel("faults in simulator runs the model never saw")
    ax.set_title(f"Fault classifier: right fault type for {ev['faultTypeAccuracy']:.0%} of {ev['faultsTested']} unseen faults",
                 loc="left", fontsize=13, color=TEXT, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def live_v1_v2(live, out):
    v1, v2 = live["v1"], live["v2"]
    panels = [("Severity tier exactly right", "exactTier", "{:.0%}", 1.0),
              ("Median seconds to an URGENT page  ↓", "medianSecondsToUrgent", "{:.0f} s", None),
              ("False alarms per hour  ↓", "falseAlarmsPerHour", "{:.1f}", None),
              ("Alerted faults named correctly", "diagnosisAccuracy", "{:.0%}", 1.0)]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (t, k, fmt, top) in zip(axs, panels):
        vals = [v1.get(k) or 0, v2.get(k) or 0]
        bars = ax.bar([0, 1], vals, color=[V1, OURS], width=0.55)
        for b, v, raw in zip(bars, vals, [v1.get(k), v2.get(k)]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), "—" if raw is None or (k == "diagnosisAccuracy" and b is bars[0]) else fmt.format(v),
                    ha="center", va="bottom", fontsize=14, fontweight="bold", color=TEXT)
        ax.set_xticks([0, 1], ["v1", "v2 (+ classifier)"])
        ax.set_title(t, loc="left", fontsize=12, color=TEXT, pad=12)
        ax.set_ylim(0, (top or max(vals + [1])) * 1.2)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9, colors=FAINT)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def benchmark(res, out, title):
    R = {r["key"]: r for r in res["rows"]}
    unsup = [r for r in res["rows"] if r["key"].endswith("+") and r["family"] in ("statistical", "classical ML", "deep learning", "ensemble")]
    best = max(unsup, key=lambda r: r["f1"]) if unsup else None
    short = {"pca+": "PCA", "iforest+": "Isolation Forest", "lof+": "LOF", "ocsvm+": "One-Class SVM", "dense_ae+": "Dense AE",
             "conv_ae+": "Conv-AE", "lstm_ae+": "LSTM-AE", "ensemble+": "Ensemble"}
    systems = [("Fixed limit\n(today)", R.get("fixed"), FIXED),
               (f"Best unsup.\n({short.get(best['key'], '?') if best else ''})", best, UNSUP),
               ("v2 URGENT\nonly", R.get("supervised_past"), SUP), ("ARK v2\n(all alarms)", R.get("hybrid_past"), OURS)]
    systems = [s for s in systems if s[1]]
    metrics = [("F1  ↑", "f1", "{:.2f}", 1.0), ("False-alarm rate %  ↓", "far", "{:.0f}%", None),
               ("False-alarm episodes / hour  ↓", "false_alarm_episodes_per_h", "{:.0f}", None),
               (f"Faults caught (of {systems[0][1]['faults']})  ↑", "faults_caught", "{:.0f}", systems[0][1]["faults"])]
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
    for ax, (t, k, fmt, top) in zip(axs, metrics):
        vals = [s[1][k] for s in systems]
        bars = ax.bar(range(len(systems)), vals, color=[s[2] for s in systems], width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color=TEXT)
        ax.set_xticks(range(len(systems)), [s[0] for s in systems], fontsize=10)
        ax.set_title(t, fontsize=13, color=TEXT, loc="left", pad=14)
        ax.set_ylim(0, (top or max(vals + [1])) * 1.18)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9, colors=FAINT)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, color=TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def make_all(results: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    p = lambda n: os.path.join(results, n)
    if os.path.exists(p("fault_classifier_eval.json")):
        classifier_accuracy(json.load(open(p("fault_classifier_eval.json"))), os.path.join(out_dir, "classifier_accuracy.png"))
    if os.path.exists(p("v2_live_eval.json")):
        live_v1_v2(json.load(open(p("v2_live_eval.json"))), os.path.join(out_dir, "live_v1_vs_v2.png"))
    if os.path.exists(p("forklift_results.json")):
        benchmark(json.load(open(p("forklift_results.json"))), os.path.join(out_dir, "forklift_benchmark.png"),
                  "Forklift runs: today's fixed limit vs ARK Predict")
    if os.path.exists(p("skab_results.json")):
        benchmark(json.load(open(p("skab_results.json"))), os.path.join(out_dir, "skab_benchmark.png"),
                  "Real pump data (SKAB): today's fixed limit vs ARK Predict")
