"""Edge-IIoTset (DNN-EdgeIIoT-dataset.csv, packet-level, capture order) -> windowed MIL arrays.
python -m src.external.prep_edge [--W 64]
NOTE: ML-EdgeIIoT-dataset.csv is shuffled/de-duplicated with a corrupted frame.time -> unusable for ordering.
The DNN file keeps the original per-capture order (contiguous blocks per Attack_type). Each contiguous
Attack_type block is one 'source segment' (analogue of one CIC day-file)."""
import argparse, json
import numpy as np, pandas as pd
from src.external import prep_common as pc

import os
CSV = os.environ.get("EDGE_IIOT_CSV", "data/external/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv")
DROP = {  # column -> reason
    "frame.time": "ordering/verification only; corrupted (IPs/NaN) in DDoS_UDP and MITM blocks; absolute time is leaky",
    "ip.src_host": "raw IP identifier", "ip.dst_host": "raw IP identifier",
    "arp.src.proto_ipv4": "raw IP identifier", "arp.dst.proto_ipv4": "raw IP identifier",
    "http.file_data": "raw payload string", "http.request.uri.query": "raw URI string", "http.request.full_uri": "raw URI string",
    "tcp.payload": "raw payload string", "tcp.options": "raw option bytes string", "mqtt.msg": "raw message payload string",
    "dns.qry.name": "raw query-name identifier", "icmp.checksum": "pseudo-random checksum", "tcp.checksum": "pseudo-random checksum",
    "tcp.ack_raw": "raw (absolute) sequence number, connection-identifying", "udp.stream": "stream index identifier",
    "icmp.transmit_timestamp": "timestamp-like value",
}
LABELS = ["Attack_label", "Attack_type"]


def main(W=64, cap=60000):
    d = pd.read_csv(CSV, low_memory=False)
    ty = d["Attack_type"].astype(str).values
    y = d["Attack_label"].astype(np.int8).values
    ts = d["frame.time"].astype(str).str.strip()
    valid = ts.str.len() == 23
    sec = pd.Series(np.nan, index=d.index)
    t = ts[valid].str.slice(5)
    sec[valid] = t.str.slice(0, 2).astype(float) * 3600 + t.str.slice(3, 5).astype(float) * 60 + t.str.slice(6).astype(float)
    feats = d.drop(columns=list(DROP) + LABELS)
    conv = {}
    for c in feats.columns:
        if feats[c].dtype == object:
            num = pd.to_numeric(feats[c], errors="coerce")
            if num.isna().mean() < 0.02:
                feats[c] = num; conv[c] = "to_numeric"
            else:
                feats[c] = pd.factorize(feats[c])[0].astype(np.float64); conv[c] = "factorize_codes"
    feats = feats.astype(np.float32)
    run = (pd.Series(ty) != pd.Series(ty).shift()).cumsum().values
    segs, seg_time = [], {}
    for r in np.unique(run):
        m = run == r; name = ty[m][0]
        segs.append((name, feats.values[m], y[m]))
        s = sec[m].dropna().values
        seg_time[name] = {"rows": int(m.sum()), "valid_time_frac": float(len(s) / m.sum()),
                          "time_resets_gt60s_backwards": int((np.diff(s) < -60).sum()) if len(s) > 1 else None}
        print(name, m.sum(), flush=True)
    out, stats = pc.build_splits(segs, W)
    rem = pc.dedup_across(out)
    capinfo = pc.cap_train(out, cap)
    T, meta = pc.fit_transform(out, list(feats.columns))
    meta.update({"dataset": "edge_iiot", "W": W, "segments": [s[0] for s in segs], "per_segment": stats,
                 "segment_time_check": seg_time, "dropped_columns": DROP, "converted_columns": conv,
                 "dedup_removed": rem, "cap": capinfo,
                 "n_bags": {s: int(len(out[s]["y"])) for s in pc.SPLITS},
                 "prevalence": {s: float(out[s]["y"].mean()) for s in pc.SPLITS}})
    pc.save(f"edge_iiot_W{W}", T, out, meta)
    print(json.dumps({k: meta[k] for k in ("n_bags", "prevalence", "dedup_removed", "cap")}, indent=1))
    print("d =", len(meta["feature_names"]), "dropped const:", meta["dropped_constant"])
    print(json.dumps(seg_time, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--W", type=int, default=64); a = ap.parse_args(); main(a.W)
