"""
Review 3 — rebuild every ML result with ONE command.

    python scripts/review3.py            # everything (~15–40 min, depending on the laptop)
    python scripts/review3.py --quick    # smaller runs, skips the slow conv autoencoder (~5–10 min)
    python scripts/review3.py --skab     # also re-run the full SKAB benchmark (needs SKAB/data, +20 min)

Steps (each one prints what it did; results land in app/services/anomaly/results/):
  1. forklift dataset     data/forklift/…              40 labelled forklift runs in SKAB format
  2. fault classifier     fault_classifier.pkl.gz      ARK Predict v2's supervised model (+ accuracy on unseen runs)
  3. live v1 vs v2        v2_live_eval.json            same simulated fleet, with and without the classifier
  4. forklift benchmark   forklift_results.json        fixed limit vs 10 models on the forklift runs (ML Lab page)
  5. SKAB benchmark       skab_results.json            (only with --skab) real pump data
  6. charts               docs/img/review3/*.png       (if matplotlib is installed)
  7. tests                tests/*.py
  8. report               FINAL_REPORT.md              every number above, in one place
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "app", "services", "anomaly", "results")
IMG = os.path.join(ROOT, "docs", "img", "review3")


def step(n, title):
    print(f"\n[{n}/8] {title}", flush=True)
    return time.time()


def done(t0, what=""):
    print(f"      done in {time.time() - t0:.0f} s {what}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller runs, skip the conv autoencoder")
    ap.add_argument("--skab", action="store_true", help="also re-run the SKAB benchmark (needs SKAB/data)")
    ap.add_argument("--keep-classifier", action="store_true", help="reuse the saved classifier instead of retraining")
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    os.makedirs(RESULTS, exist_ok=True)
    from app.services.anomaly import classifier as C
    from app.services.anomaly import evaluate as E
    from app.services.anomaly import forklift_dataset as FD
    from app.services.anomaly import skab as S

    t0 = step(1, "Forklift dataset (labelled runs in SKAB format)")
    fk_dir = os.path.join(ROOT, "data", "forklift")
    made = FD.generate(fk_dir, runs_per_scenario=5)
    done(t0)

    t0 = step(2, "ARK Predict v2 fault classifier (supervised, gradient boosting)")
    if a.keep_classifier and os.path.exists(C.MODEL_PATH):
        clf = C.FaultClassifier.load()
        ev = clf.info.get("evaluation") or json.load(open(os.path.join(RESULTS, "fault_classifier_eval.json")))
        print("      reusing the saved model")
    else:
        clf = C.build(1800 if a.quick else 5400)
        ev = C.evaluate_clf(clf, 1800 if a.quick else 3600)
        clf.info["evaluation"] = ev
        clf.save()
        json.dump(ev, open(os.path.join(RESULTS, "fault_classifier_eval.json"), "w"), indent=1)
    done(t0, f"· right fault type on unseen runs: {ev['faultTypeAccuracy']:.0%} of {ev['faultsTested']}")

    t0 = step(3, "Live stream: v1 (detectors + unsupervised ML) vs v2 (+ fault classifier)")
    live = E.compare_v2(1.0 if a.quick else 2.0, seed=11, rate=1.2)
    json.dump(live, open(os.path.join(RESULTS, "v2_live_eval.json"), "w"), indent=1)
    done(t0, f"· v2 names the right fault for {live['v2']['diagnosisAccuracy']:.0%} of alerted faults")

    t0 = step(4, "Forklift benchmark: fixed limit vs 10 models (same protocol as SKAB)")
    models = [m for m in S.DETECTORS if not (a.quick and m == "conv_ae")]
    fk = S.run(fk_dir, models, name="ARK forklift runs (simulated, labelled)", log=lambda m: print("     " + m))
    S.export_dashboard(fk, fk["_exps"], os.path.join(RESULTS, "forklift_results.json"))
    open(os.path.join(RESULTS, "forklift_results.md"), "w").write(S.markdown(fk))
    done(t0)

    skab_dir = os.path.join(ROOT, "SKAB", "data")
    if a.skab:
        t0 = step(5, "SKAB benchmark (real pump data)")
        if os.path.isdir(skab_dir):
            models = [m for m in S.DETECTORS if not (a.quick and m == "conv_ae")]
            sk = S.run(skab_dir, models, log=lambda m: print("     " + m))
            S.export_dashboard(sk, sk["_exps"], os.path.join(RESULTS, "skab_results.json"))
            open(os.path.join(RESULTS, "skab_results.md"), "w").write(S.markdown(sk))
            done(t0)
        else:
            print("      SKAB/data not found — git clone https://github.com/waico/SKAB  (skipped)")
    else:
        step(5, "SKAB benchmark — kept the saved results (add --skab to re-run)")

    t0 = step(6, "Charts")
    try:
        import charts_review3  # noqa: F401  (scripts/charts_review3.py)
        charts_review3.make_all(RESULTS, IMG)
        done(t0, f"→ {os.path.relpath(IMG, ROOT)}")
    except ImportError as e:
        print(f"      skipped ({e}) — pip install matplotlib to draw them")

    tests_ok = None
    if not a.no_tests:
        t0 = step(7, "Tests")
        tests_ok = True
        for t in ("tests/test_deepnp.py", "tests/test_skab.py", "tests/test_classifier.py", "tests/test_anomaly.py"):
            r = subprocess.run([sys.executable, t], cwd=ROOT, capture_output=True, text=True)
            last = (r.stdout.strip().splitlines() or ["?"])[-1]
            print(f"      {t:28s} {last}")
            tests_ok &= r.returncode == 0
        done(t0)
    else:
        step(7, "Tests — skipped")

    t0 = step(8, "FINAL_REPORT.md")
    import report_review3  # noqa: F401  (scripts/report_review3.py)
    path = report_review3.write(ROOT, tests_ok)
    done(t0, f"→ {os.path.relpath(path, ROOT)}")
    print("\nAll done. Start the app (uvicorn app.main:app --port 4000) → ARK Predict + ML Lab show the new results.")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
