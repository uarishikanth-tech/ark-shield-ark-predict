"""
Severity scoring (0–100) and tier mapping.

Severity is an explainable sum of named parts, so every alert can say
*why* it is urgent:

    score = crit x ( 35*type_prior + 35*magnitude + 15*persistence + 15*trend )
            + 20*spread + 10*ml

  type_prior   how worrying this *kind* of anomaly is (spike << drift)
  magnitude    how far from expected (sigma), or how long blind / frozen
  persistence  how long it has lasted
  trend        drift: how soon it reaches the hard limit at the current rate;
               dropout / stuck: how long the operator has been blind
  spread       other sensors on the same forklift misbehaving at the same time
               (process faults count process faults; sensor faults count the
               same fault on other sensors, e.g. several dropouts = gateway down)
  ml           the multivariate model's normalized anomaly score (process
               anomalies only — a dead sensor says nothing about the machine)
  crit         sensor criticality multiplier (hydraulics > battery, etc.)

Tiers: IGNORE < monitor_at <= MONITOR < urgent_at <= URGENT.
"""
from __future__ import annotations

from .config import SENSOR_BY_KEY, SeverityConfig

W_TYPE, W_MAG, W_PERS, W_TREND, W_SPREAD, W_ML = 35.0, 35.0, 15.0, 15.0, 20.0, 10.0

PROCESS_TYPES = {"drift", "multivariate"}             # the machine itself is changing
TYPE_LABEL = {"spike_burst": "spike burst", "multivariate": "ML pattern"}
SENSOR_FAULT_TYPES = {"spike", "spike_burst", "dropout", "stuck"}  # the measurement is broken


def tier_for(score: float, cfg: SeverityConfig) -> str:
    if score >= cfg.urgent_at:
        return "URGENT"
    if score >= cfg.monitor_at:
        return "MONITOR"
    return "IGNORE"


def _fmt_dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def score_incident(inc, cfg: SeverityConfig, spread_n: int, ml_peak: float,
                   feedback_offset: float = 0.0) -> tuple[int, dict, str]:
    """Returns (score, components, human explanation)."""
    typ = inc.type
    duration = max(0.0, inc.last_active_t - inc.opened_t + 1)
    crit_val = SENSOR_BY_KEY[inc.sensor].criticality if inc.sensor in SENSOR_BY_KEY else 0.9
    crit = 0.75 + 0.25 * crit_val

    prior = cfg.type_prior.get(typ, 0.4)
    trend = 0.0
    if typ == "spike":
        mag = min(1.0, abs(inc.peak_z) / (cfg.magnitude_full_z * 3))
        mag_txt = f"{abs(inc.peak_z):.1f}σ single-sample jump"
        pers = 0.0
    elif typ == "spike_burst":
        mag = min(1.0, inc.spike_count / 10)
        mag_txt = f"{inc.spike_count} spikes in {_fmt_dur(duration)}"
        pers = min(1.0, duration / (2 * cfg.persistence_full_s))
    elif typ == "drift":
        mag = min(1.0, inc.peak_z_abs / cfg.magnitude_full_z)
        mag_txt = f"{inc.peak_z_abs:.1f}σ from expected"
        pers = min(1.0, duration / 600.0)
        ttl = inc.info.get("time_to_limit_s")
        if inc.peak_z_abs >= cfg.critical_z:
            trend = 1.0
        elif ttl is not None:
            trend = max(0.0, 1.0 - ttl / 300.0)
    elif typ == "dropout":
        duration = float(inc.info.get("missing_s", duration))
        mag = min(1.0, duration / 90.0)
        mag_txt = f"no data for {_fmt_dur(duration)}"
        pers = min(1.0, duration / cfg.persistence_full_s)
    elif typ == "stuck":
        duration = float(inc.info.get("stuck_s", duration))
        mag = min(1.0, duration / 140.0)
        mag_txt = f"frozen at one value for {_fmt_dur(duration)}"
        pers = min(1.0, duration / cfg.persistence_full_s)
        trend = min(1.0, duration / 240.0)
    else:  # multivariate / abnormal pattern (raised by the ML models)
        mag = min(1.0, ml_peak / 12.0)
        mag_txt = f"pattern {ml_peak:.1f}x beyond anything seen while healthy"
        pers = min(1.0, duration / cfg.persistence_full_s)

    spread = min(1.0, spread_n / 2.0)
    # the multivariate model describes the *process*; a dead or frozen sensor
    # says nothing about the machine itself, so ML evidence only counts for
    # process anomalies (drift / multivariate)
    # (for a pattern incident the ML score already *is* the magnitude — don't count it twice)
    ml = min(1.0, ml_peak / 8.0) if typ == "drift" else 0.0

    parts = {
        "type": crit * W_TYPE * prior,
        "magnitude": crit * W_MAG * mag,
        "persistence": crit * W_PERS * pers,
        "trend": crit * W_TREND * trend,
        "spread": W_SPREAD * spread,
        "ml": W_ML * ml,
    }
    if feedback_offset:
        parts["feedback"] = feedback_offset          # learned from technician labels (feedback.py)
    score = int(round(max(0.0, min(100.0, sum(parts.values())))))

    bits = [f"{TYPE_LABEL.get(typ, typ)} ({parts['type']:.0f})", f"{mag_txt} ({parts['magnitude']:.0f})"]
    if parts["persistence"] >= 1:
        bits.append(f"lasting {_fmt_dur(duration)} ({parts['persistence']:.0f})")
    if parts["trend"] >= 1:
        if typ == "drift":
            ttl = inc.info.get("time_to_limit_s")
            t_txt = "already past hard limit" if inc.peak_z_abs >= cfg.critical_z or ttl in (None, 0) \
                else f"hits hard limit in ~{_fmt_dur(ttl)}"
        else:
            t_txt = "signal unusable — operator blind to it"
        bits.append(f"{t_txt} ({parts['trend']:.0f})")
    if parts["spread"] >= 1:
        bits.append(f"{spread_n} other sensor(s) abnormal on this truck ({parts['spread']:.0f})")
    if parts["ml"] >= 1:
        bits.append(f"ML score {ml_peak:.1f} ({parts['ml']:.0f})")
    if feedback_offset:
        bits.append(f"technician feedback ({feedback_offset:+.0f})")
    explanation = f"Severity {score} = " + " + ".join(bits)
    return score, {k: round(v, 1) for k, v in parts.items()}, explanation
