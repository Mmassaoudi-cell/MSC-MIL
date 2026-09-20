"""CIC-IDS-2017 -> windowed MIL arrays. python -m src.external.prep_cic [--W 64]"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from src.external import prep_common as pc

import os
DIR = os.environ.get("CIC_IDS_DIR", "data/external/CIC-IDS-2017")
DROP_DUP_COL = ["Fwd Header Length.1"]  # exact duplicate of 'Fwd Header Length'


def load_day(f):
    d = pd.read_csv(f, low_memory=False, encoding="latin1")
    d.columns = d.columns.str.strip()
    n0 = len(d)
    d = d[d["Label"].astype(str).str.strip() != "Label"]  # duplicate header rows (none found, guarded)
    lab = d["Label"].astype(str).str.strip()
    y = (lab.str.upper() != "BENIGN").astype(np.int8).values
    X = d.drop(columns=["Label"] + [c for c in DROP_DUP_COL if c in d.columns])
    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan)
    info = {"rows_raw": n0, "dup_header_rows": n0 - len(d), "inf_or_nan_cells": int(X.isna().sum().sum()),
            "labels": lab.value_counts().to_dict()}
    return X, y, info


def main(W=64, cap=60000):
    files = sorted(glob.glob(os.path.join(DIR, "*.csv")))
    segs, infos = [], {}
    for f in files:
        X, y, info = load_day(f)
        name = os.path.basename(f).split(".pcap")[0]
        feat = list(X.columns); infos[name] = info
        segs.append((name, X.values.astype(np.float32), y))
        print(name, X.shape, y.mean(), flush=True)
    out, stats = pc.build_splits(segs, W)
    rem = pc.dedup_across(out)
    capinfo = pc.cap_train(out, cap)
    T, meta = pc.fit_transform(out, feat)
    meta.update({"dataset": "cic2017", "W": W, "segments": [s[0] for s in segs], "per_segment": stats, "file_info": infos,
                 "dedup_removed": rem, "cap": capinfo,
                 "n_bags": {s: int(len(out[s]["y"])) for s in pc.SPLITS},
                 "prevalence": {s: float(out[s]["y"].mean()) for s in pc.SPLITS}})
    pc.save(f"cic2017_W{W}", T, out, meta)
    print(json.dumps({k: meta[k] for k in ("n_bags", "prevalence", "dedup_removed", "cap")}, indent=1))
    print("d =", len(meta["feature_names"]), "dropped const:", meta["dropped_constant"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--W", type=int, default=64); a = ap.parse_args(); main(a.W)
