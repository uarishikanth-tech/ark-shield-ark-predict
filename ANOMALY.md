# ARK Predict: Severity-Aware Streaming Anomaly Detection (HTH-ML-10)

ARK Predict adds **machine-health monitoring** to ARK Shield. It watches a live stream of forklift sensor data, finds several different kinds of anomaly, gives each one a severity score, and sends it to the right place: **ignore** (log only), **monitor** (maintenance queue) or **urgent** (page someone). It is built so operators are not flooded with alerts.

It replaces the old "Predictive model not trained" placeholder with a working module: a backend service at `/api/anomaly/*` and a live dashboard on the **ARK Predict** page of the command center.

> Simulated sensor data. Severity is a decision-support aid for maintenance triage, not a certified safety system. Nothing here controls a vehicle.

---

## How this meets the problem statement

| HTH-ML-10 asks for | What ARK Predict does | Where |
|---|---|---|
| Monitor streaming sensor data | 5 forklifts × 5 sensors (motor temp, hydraulic pressure, mast vibration, battery voltage, motor current), 1 reading per second, processed one at a time | `services/anomaly/simulator.py`, `engine.py` |
| Detect **spike, drift, dropout, stuck** | A separate detector for each shape: short-excursion spike detector, spike-aware CUSUM drift detector, dropout counter, exact-repeat stuck detector | `detectors.py` |
| Statistical + ML (Isolation Forest / autoencoder) | **Both.** Statistical: context-aware per-sensor baselines + the detectors above. ML, three models: an **Isolation Forest** and a **multivariate Gaussian** (leave-one-out) on ~10 s-smoothed residuals catch *joint shifts* (several sensors a little off together, e.g. overload), and a **deep temporal autoencoder** (MLP 180→64→12→64→180) catches *shapes over time* (e.g. an oscillation inside every limit). Any of them can open an "abnormal pattern" incident when no single-sensor detector explains it, and each says which sensors it points at | `detectors.py`, `ml.py` |
| Estimate **severity** | Explainable 0–100 score = type + magnitude + persistence + trend (time until the hard limit) + spread on the truck + ML, scaled by how critical the sensor is | `severity.py` |
| Likely **root cause** | Checks which sensors move together (residual correlation) and matches known fault signatures, e.g. vibration↑ + temperature↑ → bearing wear; pressure↓ + current↑ → hydraulic leak; dropout while other sensors are fine → sensor/telemetry fault; same drift on several trucks → shared cause | `rootcause.py` |
| Route to **ignore / monitor / urgent** | IGNORE → log only · MONITOR → maintenance queue · URGENT → page on-call technician + supervisor | `router.py`, `config.py` |
| **No single fixed threshold** | Every threshold is per sensor, per anomaly type, and measured in that sensor's own learned noise (σ) around a load-aware expected value | `config.py` (`SensorSpec`) |
| **Minimise alert fatigue** | Confirmation delay, one alert per incident, re-alert only when it escalates, per-truck grouping, suppression of on/off flapping, spike bursts rolled into one incident. Every suppression is counted | `router.py` |
| Live dashboard with tiers + routing | ARK Predict page: fleet health cards, live sensor charts with detected incidents and injected ground truth, routing feed, incident drawer with severity breakdown | `app/static/index.html` |
| End-of-run summary report | `/api/anomaly/summary` (JSON) and `/api/anomaly/report.md` (Markdown download, "Run report" button) | `service.py` |
| Bonus: root-cause hints | Yes, e.g. "correlated with ARK-F002 mast vibration (r=+0.91)" | `rootcause.py` |
| *(extra)* Human in the loop | Technicians mark any alert Right call / Too urgent / Not urgent enough / False alarm; the system learns a bounded severity correction per (anomaly type, sensor) | `feedback.py` |

---

## Results (offline evaluation against ground truth)

```
python -m app.services.anomaly.evaluate --hours 6 --seed 11
```

These results come from 6 simulated hours, 5 trucks, 10 fault types, about 55 injected faults per hour (a deliberately harsh rate):

| Metric | Result |
|---|---|
| Faults detected | **97 %** (312 / 321) — the misses are mostly overloads on idle trucks |
| Anomaly type correct | **94 %** |
| Severity tier exactly right | **91 %** (within one tier: 97 %) |
| URGENT faults routed as URGENT | **100 %** |
| False alarms on a fault-free stream | **0** (`test_no_false_alarms_on_healthy_stream`, plus 6 fleet-hours of clean runs) |
| Alerts sent vs. naive fixed-threshold alerter | **303 vs 1,664** (82 % fewer). The baseline's precision is 36 % |
| Median detection latency | spike 1 s · dropout 2 s · stuck 10 s · oscillation 34 s · overload 43 s · leak / drift / bearing 56–64 s · battery fade 126 s |

A second seed (42) gives 99 % detection, 97 % type accuracy, 95 % exact tier and 82 % fewer alerts.

### What each model adds (ablation)

```
python -m app.services.anomaly.evaluate --compare --hours 6
```

Same 6-hour stream and faults, five model configurations:

| Models | Faults found | Right type | Right tier | Urgent caught | False alarms/h | Oscillations | Overloads |
|---|---:|---:|---:|---:|---:|---:|---:|
| Statistics only | 87 % | 84 % | 82 % | 100 % | 0 | 0 / 20 | 1 / 22 |
| + Isolation Forest only | 87 % | 84 % | 82 % | 100 % | 0 | 0 / 20 | 2 / 22 |
| + Gaussian only | 91 % | 88 % | 86 % | 100 % | 0 | 2 / 20 | 13 / 22 |
| + Autoencoder only | 94 % | 91 % | 88 % | 100 % | 0 | 20 / 20 | 3 / 22 |
| **+ all three (default)** | **97 %** | **94 %** | **91 %** | **100 %** | **0** | **20 / 20** | **13 / 22** |

Each ML model earns its place with a fault class the others miss: the **autoencoder** finds oscillations (a wobble inside every single-reading limit, e.g. a slack lift chain), and the **Gaussian** finds joint shifts (overload: pressure ↑, current ↑, voltage ↓, each under its own limit). The **Isolation Forest**, fed the same smoothed input, adds little on its own in this benchmark (2 overloads vs 1). In the full stack it is often the first model to raise an overload, and it makes no distribution assumptions, which matters on real data that is not Gaussian. Be honest about this if asked. Overload is the hardest fault: if the truck sits idle the extra load does not show, so 9 of 22 slip by.

**Caveat:** the detectors were tuned against this simulator, so the numbers show the design works end to end. They are not a claim about real forklifts. Real-data results (SKAB) are in the next section and in [SKAB_RESULTS.md](./SKAB_RESULTS.md).

---

## Run it

```bash
pip install -r requirements.txt           # adds numpy + scikit-learn
uvicorn app.main:app --reload --port 4000
```

Open `http://localhost:4000/`, sign in with any demo role that can see **ARK Predict** (Super Admin, Safety Manager, Fleet Manager), then click **ARK Predict** in the sidebar.

- **Postgres is not needed** for ARK Predict. The anomaly pipeline runs entirely in memory. If the database is not running, the server still boots: the forklift simulation logs a warning and ARK Predict runs normally.
- Without scikit-learn the ML layer falls back to numpy-only models (Mahalanobis distance instead of the Isolation Forest, a linear PCA autoencoder instead of the MLP) and everything else keeps working.
- Warm-up (fitting 25 baselines, the Isolation Forest, the Gaussian and the autoencoder) takes about 3 s at startup and on every Reset.

Other entry points:

```bash
python -m app.services.anomaly.evaluate --hours 4        # offline benchmark → Markdown report
python -m app.services.anomaly.evaluate --json           # same, full JSON incl. every fault
python -m app.services.anomaly.evaluate --compare        # ML ablation: none / IF / Gaussian / autoencoder / all
python tests/test_anomaly.py                             # 14 tests (or: python -m pytest tests -q)
python scripts/check_ark_predict.py                      # demo-day smoke test against the RUNNING server
```

### Run it on real data (SKAB) — done

**Full results: [SKAB_RESULTS.md](./SKAB_RESULTS.md)** (also live in the dashboard: **ML Lab** page).

On the real SKAB benchmark (34 water-pump experiments, 8 sensors, hand-labelled faults, official leaderboard
protocol, scoring code verified against the published leaderboard):

| | F1 | False-alarm rate | False-alarm episodes / h | Faults caught |
|---|---:|---:|---:|---:|
| Fixed 3σ limit (today's alarms) | 0.76 | 44 % | 95 | 32/34 |
| Conv-1D autoencoder (improved) | 0.75 | 18 % | 2.3 | 29/34 |
| **ARK Predict v2** (trained on past experiments only) | **0.81** | 24 % | 12 | **34/34** |
| ↳ its URGENT tier alone (supervised) | 0.75 | **5 %** | 5.9 | 27/34 |
| ARK Predict v2 on never-seen fault families | 0.81 | 26 % | 11 | 34/34 |
| Best published SKAB model (Conv-AE, unsupervised) | 0.78 | 13.6 % | — | — |

```bash
git clone https://github.com/waico/SKAB
python -m app.services.anomaly.skab SKAB/data --export   # 10 models + fixed limit + ARK v1/v2 (~20 min; --fast skips conv-AE)
SKAB_DIR=SKAB python tests/test_skab.py                   # incl. reproducing the published leaderboard numbers
```

`replay.py` still streams any SKAB-format CSV through the *live* v1 pipeline row by row
(`python -m app.services.anomaly.replay SKAB/data`).

### Settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `ANOMALY_AUTO_START` | `true` | start the stream when the server boots |
| `ANOMALY_PUBLIC_DEMO` | `true` | open `/api/anomaly/*` without a JWT (the demo UI has no real token). Set to `false` so reads need any logged-in user and controls need Safety/Fleet |
| `ANOMALY_SPEED` | `2` | simulated seconds per 0.5 s tick (2 = 4× real time) |
| `ANOMALY_FAULT_RATE_PER_MIN` | `0.8` | random faults injected per simulated minute across the fleet |
| `ANOMALY_SEED` | `7` | simulator seed (repeatable demos) |

Detector sensitivities, severity weights, tier cut-offs and routing windows live in `app/services/anomaly/config.py`.

---

## 3-minute demo script

1. **Open ARK Predict.** The fleet cards are green and the sensor charts follow the dashed "expected for this load" line. A heavy lift moves every signal, and nothing fires. *"We never compare against a fixed number. We compare against what this truck should read at this load."*
2. **Click "Single spike".** One sample jumps 12σ. It shows up as **LOGGED ONLY** under *All decisions* and nobody is paged. *"Noise is not an incident."*
3. **Click "Bearing wear" and set 20× speed.** Mast vibration starts drifting, then motor temperature follows. There is one **MONITOR** alert, then an **ESCALATED → URGENT**. The second sensor is **grouped** into the same alert instead of paging twice. The root cause reads *"Probable drive-motor bearing wear — vibration ↑ with temperature ↑"*. Click the alert to show the severity breakdown (type, magnitude, trend "hits hard limit in ~1m20s", spread, ML).
4. **Click "Stuck pressure sensor".** The reading flat-lines while the expected line keeps moving. It is typed **stuck**, and the root cause says *"Frozen transmitter… not a process fault"*. It escalates to URGENT after about 2 minutes, because the operator has no view of hydraulics.
5. **Click "Chain oscillation (DL only)".** The mast vibration wobbles inside its limits. In the ML chart, the purple **autoencoder** line climbs over the trigger while the orange **Isolation Forest** line stays flat. An *abnormal pattern (ML)* incident opens on the right sensor, and the root cause reads *"Oscillating mast vibration — slack lift chain, worn mast rollers or resonance"*.
6. **Click "Overload lift (joint pattern)".** Pressure, current and voltage all move a little, each inside its own band. The teal **Gaussian** line crosses the trigger and the incident reads *"Probable overload — lifting above rated capacity"*.
7. **Open any alert and click "False alarm".** A toast shows what the system learned (e.g. *"Future stuck alerts on battery voltage: −6 severity"*), and the *Learned from technicians* panel updates.
8. **Point at the tiles:** *"Kept from operators"* and *"X % fewer alerts than a fixed threshold"*.
9. **Click "Run report"** to download the end-of-run report: detection per type, severity calibration matrix and alert-fatigue funnel.

---

## Architecture

```
SensorFleetSimulator ──frame/s──▶ AnomalyEngine
 (duty cycle + faults,             ├─ SensorDetector ×25   context baseline → z → spike / drift / dropout / stuck
  ground-truth labels)             ├─ MachineModel ×2       Isolation Forest + leave-one-out Gaussian on ~10 s-smoothed [z₁..z₅, duty]
                                   ├─ SequenceAutoencoder   MLP autoencoder on the last 30 s of [z₁..z₅, duty] (shape only)
                                   └─ AlertRouter           incidents → severity → tier → route
                                        ├─ severity.py      explainable 0–100 (+ learned feedback term)
                                        ├─ rootcause.py     co-movement + fault signatures
                                        └─ feedback.py      technician verdicts → bounded per-fault-kind corrections
AnomalyService (asyncio task in the FastAPI loop) ──▶ /api/anomaly/* ──▶ ARK Predict page (polls 1 s)
                                                  └─▶ Socket.IO "anomaly:alert"
```

**Context-aware baseline.** Each sensor's expected value comes from a linear model on the truck's reported duty (instantaneous, fast-averaged and slow-averaged, so heating lag is captured). The model is fitted on 15 simulated minutes of healthy warm-up. Detectors work on `z = (reading − expected) / σ_sensor`. The noise level σ adapts slowly, and only on quiet samples, so it can follow normal sensor ageing without absorbing a fault. This is also why the naive baseline fails: heavy lifts push raw pressure and current past a fixed 3σ limit, while slow faults such as a leak stay inside the normal lift range.

**Drift detection.** A two-sided CUSUM on z first *arms* the alarm. The drift is only *confirmed* once the smoothed deviation also passes 2.5σ. Real residuals are autocorrelated, and CUSUM alone fires constantly on healthy wander (in testing: 740 false alarms in 75 sensor-hours without the confirmation step, 2 with it). Samples that turn out to be spikes are never fed to the CUSUM.

**Deep temporal autoencoder.** Input: the last 30 seconds of each truck's `[z of 5 sensors, duty]`, flattened to 180 numbers (z clipped to ±3σ, because anything bigger is the statistical detectors' job, and each sensor's mean over the window removed, so it models *shape*, not level). Network: 180 → 64 → 12 → 64 → 180, tanh, trained with Adam and early stopping (scikit-learn `MLPRegressor`) on healthy warm-up windows only (about 3,700 windows). Score: each sensor's reconstruction error, normalized so 1.0 = that sensor's 99.5th percentile on healthy data; the truck's score is the worst sensor. That gives attribution for free: the sensor it could not rebuild is the one it points at. Live scoring is a numpy forward pass (identical to `MLPRegressor.predict`, asserted in the tests). A pattern incident opens when max(IF, Gaussian, autoencoder) is at 1.6 or more for at least 20 of the last 30 s *and* no statistical detector fired in the last 35 s on the sensors the model points at (so a spike still inside the 30 s window is not reported twice). Pattern incidents wait 20 s before notifying, so all three models weigh in before a cause is named.

**Joint-shift Gaussian.** Input: each sensor's z smoothed over about 10 s (EWMA), plus duty. It measures the Mahalanobis distance from the healthy mean, scaled by the healthy covariance, **leave-one-out**: the single most deviant sensor is dropped before scoring. A lone drifting sensor is therefore left to the drift detector, and what remains is a *joint* shift. The Isolation Forest sees the same smoothed input.

**Technician feedback.** Each verdict turns into the severity correction that would have produced the right tier (e.g. "false alarm" → drop below 30). It is learned per (anomaly type, sensor) as a moving average (α = 0.35), so conflicting verdicts average out, and capped at ±30 points so feedback can tune the model but never silence it. The correction shows up as its own line in the severity explanation.

**Fast Isolation Forest.** scikit-learn's `score_samples` costs about 20 ms per call (per-tree dispatch), which is too slow for scoring every second. The fitted forest is compiled once into padded numpy arrays and all trees are traversed together. The scores are identical to sklearn's (asserted in the tests) at about 0.25 ms per call.

**Routing rules** (`router.py`):
1. Tier from severity: `< 30` IGNORE, `30–60` MONITOR, `≥ 60` URGENT.
2. A MONITOR incident must last 3 s before anyone is notified. URGENT is sent immediately.
3. One alert per incident. A new alert is only sent if the incident *escalates*.
4. A new incident on a truck that already has an open, notified incident joins that alert (as extra evidence) unless it raises the truck's tier.
5. An incident that recurs within 5 minutes of resolving is re-opened, not re-alerted.
6. Single spikes are logged. 3 or more spikes on one sensor within 10 minutes become one `spike_burst` incident (likely a loose connector).

---

## API: `/api/anomaly`

| Method | Path | Returns |
|---|---|---|
| GET | `/status` | running, simulated time, counters (readings, detector hits, incidents, notifications, suppressions, baseline alerts) |
| GET | `/config` | trucks, sensors (with per-sensor thresholds), scenarios, tiers, routes |
| GET | `/machines` | per-truck health: tier, severity, headline, root cause, latest reading/expected/z/state per sensor |
| GET | `/series?machineId=ARK-F002&points=240` | chart data: value, expected, z per sensor, incident bands, injected ground truth, ML score |
| GET | `/alerts?limit=50` | routed notifications only (what a human was told), newest first |
| GET | `/feed?limit=80` | every routing decision: alert, grouped, logged-only, resolved |
| GET | `/incidents?status=open\|all` · `/incidents/{id}` | incidents with severity components, explanation, root cause, tier history |
| GET | `/summary` · `/report.md` | end-of-run summary + evaluation vs ground truth (JSON / Markdown) |
| POST | `/start` `/pause` `/reset {seed?}` | stream control |
| POST | `/speed {speed}` · `/fault-rate {ratePerMin}` | demo pacing |
| POST | `/inject {scenario, machineId?, sensor?, durationS?, magnitude?}` | DEMO: inject a labelled fault. `scenario` ∈ spike, spike_burst, drift, dropout, stuck, bearing_wear, hydraulic_leak, battery_fade, oscillation, overload |
| POST | `/feedback {incidentId, label}` | technician verdict: `right_call` · `too_high` · `too_low` · `false_alarm` → returns the learned correction |
| GET | `/feedback` | everything learned so far, per (anomaly type, sensor), plus recent verdicts |

Socket.IO: every routed alert is also emitted as `anomaly:alert` (same payload as `/alerts`).

---

## Files

```
app/services/anomaly/
  config.py       sensors, per-sensor thresholds, severity weights, tiers, routing windows
  simulator.py    streaming fleet simulator + labelled fault injection (ground truth)
  detectors.py    context-aware baseline + spike / drift / dropout / stuck detectors
  ml.py           Isolation Forest (fast numpy scorer) · leave-one-out Gaussian · temporal MLP autoencoder · fallbacks
  feedback.py     technician verdicts → bounded severity corrections
  severity.py     explainable 0–100 severity + tier mapping
  rootcause.py    co-movement + fault-signature root-cause hints
  router.py       incidents, tiered routing, alert-fatigue controls
  engine.py       wires the layers together; naive fixed-threshold baseline for comparison
  service.py      live asyncio service, API views, Markdown report
  evaluate.py     offline benchmark vs ground truth (CLI)
  replay.py       replay real labelled CSVs (SKAB format) through the same pipeline + score them
  skab.py         SKAB benchmark: fixed limit vs 10 models (PCA, IF, LOF, OCSVM, 3 autoencoders, ensemble, supervised, hybrid)
  deepnp.py       numpy deep learning: Conv-1D + LSTM autoencoders with hand-written backprop (gradient-checked)
  results/skab_results.json   precomputed benchmark results + replays for the ML Lab page
app/routers/anomaly.py   REST API
tests/test_anomaly.py    14 tests
scripts/check_ark_predict.py   demo-day smoke test (run against the live server)
```
