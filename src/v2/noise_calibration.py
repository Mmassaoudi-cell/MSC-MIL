"""Part 1: estimate real EVSE meter jitter and telemetry-gap rate from raw ACN 4-s telemetry.

Method (see results/v2/NOISE_CALIBRATION.md):
  plateau = maximal run of consecutive samples with Charging State in {ADAPTIVE, CHARGING}, Actual Pilot constant
            (or, if pilot missing, Charging Current within +-0.5 A of the run median), Power > 1 kW, and every Power
            sample within +-5% of the run median; run length >= 30 samples (~2 min).
  jitter  = 1.4826 * MAD(diff(P)) / sqrt(2) / mean(P)   (sample-to-sample, drift-robust)
  jitter_level = 1.4826 * MAD(P) / mean(P)              (level-based, includes slow drift)
Same for Charging Current (jitter_I), for sites/sessions lacking Power.
Missing: (i) NaN Power fraction among within-session samples of sessions whose Power column is populated;
         (ii) fraction of 5-min slots inside the charging span [first, last charging sample] with zero samples;
         (iii) fraction of inter-sample gaps > 12 s (3x nominal 4 s).
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/acn_static/time series data"
OUT = ROOT / "results/v2"
CHG = {"ADAPTIVE", "CHARGING"}
MIN_LEN = 30
rng = np.random.default_rng(0)


def robust_rel(x, diff=True):
    x = np.asarray(x, float)
    m = x.mean()
    if m <= 0: return np.nan
    if diff:
        d = np.diff(x)
        return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) / m
    return 1.4826 * np.median(np.abs(x - np.median(x))) / m


def plateaus(vals, ok, min_len=MIN_LEN, band=0.05, absband=None):
    """maximal runs where ok & vals within band of run median (greedy on ok-runs, then trim by median band)."""
    n = len(vals); out = []
    i = 0
    while i < n:
        if not ok[i]: i += 1; continue
        j = i
        while j < n and ok[j]: j += 1
        seg = vals[i:j]
        if len(seg) >= min_len:
            # split further into sub-runs within band of a rolling reference (median of the ok-run in windows)
            k = 0
            while k < len(seg):
                ref = np.median(seg[k:k + min_len])
                tol = band * ref if absband is None else absband
                e = k
                while e < len(seg) and abs(seg[e] - ref) <= tol: e += 1
                if e - k >= min_len: out.append((i + k, i + e))
                k = max(e, k + 1)
        i = j
    return out


def process(f):
    try:
        d = pd.read_csv(f, index_col=0)
    except Exception:
        return None
    if len(d) < 60: return None
    t = pd.to_datetime(d.index, utc=True, errors="coerce")
    st = d["Charging State"].astype(str).values
    P = d["Power (kW)"].values.astype(float)
    I = d["Charging Current (A)"].values.astype(float)
    pil = d["Actual Pilot (A)"].values.astype(float)
    chg = np.isin(st, list(CHG))
    res = {"n": len(d), "jit_P": [], "jit_P_lvl": [], "jit_I": [], "jit_I_lvl": [], "meanP": [], "meanI": []}
    hasP = np.isfinite(P).mean() > 0.5
    if chg.sum() < MIN_LEN: return res | {"hasP": hasP, "charging_samples": int(chg.sum())}
    pil_ok = np.isfinite(pil).mean() > 0.5
    pil_const = np.zeros(len(d), bool)
    if pil_ok:
        pil_const[1:] = (pil[1:] == pil[:-1]) & np.isfinite(pil[1:])
        pil_const[:-1] |= pil_const[1:]
    okI = chg & np.isfinite(I) & (I > 3) & (pil_const if pil_ok else True)
    for a, b in plateaus(I, okI, absband=0.5 if not pil_ok else None, band=0.05):
        s = I[a:b]; res["jit_I"].append(robust_rel(s)); res["jit_I_lvl"].append(robust_rel(s, False)); res["meanI"].append(s.mean())
    if hasP:
        okP = chg & np.isfinite(P) & (P > 1.0) & (pil_const if pil_ok else True)
        for a, b in plateaus(P, okP, band=0.05):
            s = P[a:b]
            res["jit_P"].append(robust_rel(s)); res["jit_P_lvl"].append(robust_rel(s, False)); res["meanP"].append(s.mean())
    # missing / gaps within session (from first to last charging sample)
    idx = np.where(chg)[0]
    res["hasP"] = hasP; res["charging_samples"] = int(chg.sum())
    if len(idx) >= 2:
        a, b = idx[0], idx[-1]
        tt = t[a:b + 1]
        sec = (tt - tt[0]).total_seconds().values
        res["span_s"] = float(sec[-1])
        # NaN power within span among sessions with populated Power
        if hasP:
            res["nan_P_frac"] = float(np.mean(~np.isfinite(P[a:b + 1])))
        gaps = np.diff(sec)
        res["gap_frac"] = float(np.mean(gaps > 12.0)); res["gap_time_frac"] = float(gaps[gaps > 12.0].sum() / max(sec[-1], 1))
        nb = int(sec[-1] // 300)
        if nb >= 1:
            occ = np.unique((sec // 300).astype(int)); occ = occ[occ < nb]
            res["empty_slot_frac"] = 1 - len(occ) / nb
            res["n_slots"] = nb
    return res


def summarize(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if len(v) == 0: return None
    return dict(n=int(len(v)), median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                mean=float(v.mean()))


def main():
    sites = {
        "caltech": sorted(glob.glob(str(RAW / "caltech/*/*.csv.gz"))),
        "office_01": sorted(glob.glob(str(RAW / "office_01/*/*.csv.gz"))),
        "jpl": sorted(glob.glob(str(RAW / "jpl/*/*.csv.gz"))),
    }
    n_sample = {"caltech": 3000, "office_01": 1474, "jpl": 1500}
    out = {}
    for site, fs in sites.items():
        sel = rng.choice(len(fs), size=min(n_sample[site], len(fs)), replace=False)
        R = [process(fs[i]) for i in sel]
        R = [r for r in R if r]
        pj = [np.median(r["jit_P"]) for r in R if r["jit_P"]]
        pjl = [np.median(r["jit_P_lvl"]) for r in R if r["jit_P_lvl"]]
        ij = [np.median(r["jit_I"]) for r in R if r["jit_I"]]
        seg_pj = [x for r in R for x in r["jit_P"]]
        nan_w = [(r["nan_P_frac"], r["charging_samples"]) for r in R if "nan_P_frac" in r]
        d = dict(files_in_site=len(fs), files_sampled=len(sel), files_read=len(R),
                 sessions_with_power_populated=int(sum(r["hasP"] for r in R)),
                 sessions_with_power_plateau=len(pj), n_power_plateaus=len(seg_pj),
                 rel_jitter_power_diff_per_session=summarize(pj),
                 rel_jitter_power_diff_per_plateau=summarize(seg_pj),
                 rel_jitter_power_level_per_session=summarize(pjl),
                 rel_jitter_current_diff_per_session=summarize(ij),
                 nan_power_frac_within_powered_sessions=(float(np.average([a for a, _ in nan_w], weights=[max(w, 1) for _, w in nan_w])) if nan_w else None),
                 nan_power_frac_per_session=summarize([a for a, _ in nan_w]),
                 gap_gt12s_frac_of_intervals=summarize([r["gap_frac"] for r in R if "gap_frac" in r]),
                 empty_5min_slot_frac_per_session=summarize([r["empty_slot_frac"] for r in R if "empty_slot_frac" in r]),
                 empty_5min_slot_frac_pooled=(float(sum(r["empty_slot_frac"] * r["n_slots"] for r in R if "n_slots" in r) /
                                                     max(sum(r["n_slots"] for r in R if "n_slots" in r), 1))))
        out[site] = d
        print(site, json.dumps(d, indent=1)[:1800], flush=True)
    json.dump(out, open(OUT / "noise_calibration_raw_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
