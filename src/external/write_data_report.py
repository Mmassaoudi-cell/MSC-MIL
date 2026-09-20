"""Assemble tables for results/external/EXTERNAL_DATA_REPORT.md from cached meta + tuning JSONs (validation only)."""
import json
from src.external.prep_common import PROC, RES
from src.external.models import ALL_MODELS


def split_table(meta):
    ps = meta["per_segment"]
    rows = ["| source segment | rows | bags | train bags (pos) | val bags (pos) | test bags (pos) |", "|---|---|---|---|---|---|"]
    for k, v in ps.items():
        rows.append(f"| {k} | {v['n_rows']} | {v['n_bags']} | {v['train_bags']} ({v['train_pos']}) | {v['val_bags']} ({v['val_pos']}) | {v['test_bags']} ({v['test_pos']}) |")
    return "\n".join(rows)


def main():
    parts = []
    for ds, dn in (("cic2017", "cic2017_W64"), ("edge_iiot", "edge_iiot_W64")):
        m = json.loads((PROC / dn / "meta.json").read_text())
        parts.append(f"### {ds}: bags (after dedup) {m['n_bags']}, prevalence {{k: round(v,4) for k,v in m['prevalence'].items()}}".replace(
            "{k: round(v,4) for k,v in m['prevalence'].items()}", str({k: round(v, 4) for k, v in m['prevalence'].items()})))
        parts.append(f"dedup removed: {m['dedup_removed']}; cap: {m['cap']}; d = {len(m['feature_names'])} features; "
                     f"dropped constant columns: {m['dropped_constant']}\n")
        parts.append(split_table(m) + "\n")
    parts.append("### Validation macro-F1@0.5 (mean +- std over seeds 0,1,2 of the tuned configs)\n")
    rows = ["| model | cic2017 | edge_iiot |", "|---|---|---|"]
    cfg = {ds: json.loads((RES / f"tuned_configs_{ds}.json").read_text()) for ds in ("cic2017", "edge_iiot")}
    for n in ALL_MODELS:
        c = [f"{cfg[ds][n]['val_macro_f1_mean_3seeds']:.4f} +- {cfg[ds][n]['val_macro_f1_std']:.4f}" if n in cfg[ds] else "n/a" for ds in cfg]
        rows.append(f"| {n} | {c[0]} | {c[1]} |")
    parts.append("\n".join(rows))
    (RES / "_tables_autogen.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
