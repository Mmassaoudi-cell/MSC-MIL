import hashlib, json, glob
from src.v2.final import run_eval as R
from src.v2.common import DATA_V2, PROJECT_ROOT
from src.v2 import dataset_v2 as ds


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    P = PROJECT_ROOT
    files = {"frozen_baseline_configs.json": P / "results/v2/frozen_baseline_configs.json", "FINAL_MODEL_CONFIG_v2.yaml": P / "FINAL_MODEL_CONFIG_v2.yaml",
             "src/v2/enhanced_model.py": P / "src/v2/enhanced_model.py", "src/v2/baselines_lib.py": P / "src/v2/baselines_lib.py",
             "src/v2/dataset_v2.py": P / "src/v2/dataset_v2.py", "src/v2/final/run_eval.py": P / "src/v2/final/run_eval.py",
             "src/v2/final/stats.py": P / "src/v2/final/stats.py", "src/v2/final/latency.py": P / "src/v2/final/latency.py"}
    stamp = json.loads((R.FIN / "TEST_FIRST_LOAD.json").read_text())
    seeds = {m: {f"SCMA-{t}": list(range(R.n_seeds(m, t))) for t in (1, 2, 4)} for m in R.TREES + R.DEEP_LOW + R.DEEP_HI + ["e4_net"]}
    man = dict(
        config_sha256=dict((k, sha(v)) for k, v in files.items()),
        per_model_config_entry_sha256={m: R.entry_hash(m) for m in R.TREES + R.DEEP_LOW + ["msc_mil", "source_bilstm_attn", "pi_fusion", "e4_net"]},
        data_cache_sha256={f"scma{t}_v2.npz": sha(DATA_V2 / f"scma{t}_v2.npz") for t in (1, 2, 4)},
        seeds_per_model_and_type=seeds,
        test_attack_sampling_seeds=dict(val=ds.VAL_ATTACK_SEED, test_n200=ds.N200_ATTACK_SEED, test_full=ds.FULL_ATTACK_SEED),
        train_attack_variant_seed=ds.TRAIN_C_SEED,
        bootstrap=dict(resamples=2000, seed_base=20260919, seed_used="20260919 + scma_type", stratified_by_label=True, statistic="macro-F1 at threshold 0.5 of the seed-averaged probability"),
        test_split_first_loaded=stamp,
        test_split_statement=("The v2 test splits (test_full, test_n200) were first deserialised by src/v2/final/run_eval.py::load_test at "
                              f"{stamp['first_test_load_local']} (local time), at the start of the single final evaluation. Before this no development step read them "
                              "(enh_data.py / baselines_lib.load_trainval read train/val/meta keys only). One diagnostic look at feature means of the test arrays "
                              "(benign vs attack summary statistics on SCMA-2, to check the loader was not at fault after unexpectedly low tree scores) was made "
                              "after the first tree results appeared; it changed no configuration, no model and no result."),
        notes=["e4_net (E4 network alone) is the network component of each pi_fusion run: same training run, same seed, so it is not trained twice.",
               "Every (model,type,seed) trained once; no config changed after seeing test numbers."])
    (R.FIN / "EVAL_MANIFEST.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
