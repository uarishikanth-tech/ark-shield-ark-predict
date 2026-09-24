"""
Demo-day smoke test for ARK Predict.

Start the server first:
    uvicorn app.main:app --port 4000
then, in a second terminal:
    python scripts/check_ark_predict.py            # or: python scripts/check_ark_predict.py http://localhost:4001

It calls every /api/anomaly endpoint on the *running* server, injects one fault,
and prints PASS / FAIL per check. Standard library only — nothing to install.
Exit code 0 = everything passed.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:4000").rstrip("/")
API = BASE + "/api/anomaly"
results: list[tuple[str, bool, str]] = []


def call(path: str, method: str = "GET", body: dict | None = None, raw: bool = False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        txt = r.read().decode()
        return txt if raw else json.loads(txt)


def check(name: str, fn):
    try:
        detail = fn()
        results.append((name, True, detail or ""))
    except AssertionError as e:
        results.append((name, False, str(e)))
    except urllib.error.HTTPError as e:
        results.append((name, False, f"HTTP {e.code}: {e.read().decode()[:200]}"))
    except Exception as e:  # noqa: BLE001
        results.append((name, False, repr(e)))


def main() -> int:
    print(f"Checking ARK Predict at {BASE} …\n")
    check("server is up (/health)", lambda: call(BASE + "/health")["status"] == "ok" and "ok")

    def status():
        st = call(API + "/status")
        assert st["ready"], "anomaly engine not ready (warm-up failed?)"
        assert st["running"], "stream is paused — POST /api/anomaly/start or press Resume"
        return f"sim time {st['simT']}s · ML: {st['mlModel']}"
    check("stream running", status)

    def config():
        cfg = call(API + "/config")
        assert len(cfg["machines"]) == 5 and len(cfg["sensors"]) == 5
        ml = cfg.get("ml") or {}
        assert ml.get("autoencoder"), "autoencoder missing — is scikit-learn installed?"
        return "5 trucks × 5 sensors · autoencoder " + str(ml["autoencoder"].get("layers"))
    check("config + ML models", config)

    check("machines", lambda: f"{len(call(API + '/machines'))} trucks")

    def series():
        s = call(API + "/series?machineId=ARK-F002&points=120")
        assert s["sensors"]["motor_temp"]["t"], "no sensor history yet"
        return f"{len(s['sensors']['motor_temp']['t'])} points"
    check("series (chart data)", series)

    for ep in ("alerts", "feed", "incidents?status=all", "feedback", "summary"):
        check(ep, lambda ep=ep: f"ok ({type(call(API + '/' + ep)).__name__})")
    check("report.md", lambda: (lambda md: (md.startswith("# ARK Predict") or (_ for _ in ()).throw(
        AssertionError("unexpected report"))) and f"{len(md)} chars")(call(API + "/report.md", raw=True)))

    def inject():
        before = call(API + "/status")["incidents"]
        r = call(API + "/inject", "POST", {"scenario": "stuck", "machineId": "ARK-F003", "sensor": "hydraulic_pressure",
                                           "durationS": 60})
        assert r["ok"]
        call(API + "/speed", "POST", {"speed": 10})
        for _ in range(20):
            time.sleep(1)
            if call(API + "/status")["incidents"] > before:
                break
        call(API + "/speed", "POST", {"speed": 2})
        inc = [i for i in call(API + "/incidents?status=all") if i["machine"] == "ARK-F003" and i["type"] == "stuck"]
        assert inc, "injected stuck sensor was not detected within ~20 s"
        return f"detected as #{inc[0]['id']} ({inc[0]['tier']}, severity {inc[0]['severity']})"
    check("inject + detect a fault", inject)

    def feedback():
        incs = call(API + "/incidents?status=all")
        assert incs, "no incidents to give feedback on"
        r = call(API + "/feedback", "POST", {"incidentId": incs[0]["id"], "label": "right_call"})
        return f"learned offset {r['learnedOffset']} for {r['type']} / {r['sensor']}"
    check("technician feedback", feedback)

    check("dashboard page (/)", lambda: "ARK Predict" in call(BASE + "/", raw=True) and "html ok")

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
