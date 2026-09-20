import json
from src.v2.robust import lib
from src.v2 import enhanced_model as em
P = lib.PROJECT_ROOT
rel = lambda f: f.relative_to(P).as_posix()
ck = {}
for f in sorted(list(lib.FCK.glob('*_scma[14]_seed[0-4]*')) + list(lib.RCK.glob('*.pkl')) + list(lib.RCK.glob('*.pt'))):
    ck[rel(f)] = lib.sha_file(f)
cfg = {k: lib.sha_obj(v) for k, v in lib.FROZEN.items()}
cfg['pi_fusion'] = lib.sha_obj(em.FROZEN_CFG)
files = ['FINAL_MODEL_CONFIG_v2.yaml', 'results/v2/frozen_baseline_configs.json', 'results/v2/noise_calibration.json'] + [rel(p) for p in (P / 'src/v2/robust').glob('*.py')]
man = dict(seeds=dict(robustness_ablation='0-4 on SCMA-1,SCMA-4', lowdata='0-2 on SCMA-1 (10/25/50%); 100% = frozen final runs', multiclass='0-4'),
           sigma_r=lib.SIGMA_R, missing_real_rate=lib.MISSING_REAL, perturbation_rng='np.random.default_rng([20260920, cond_idx, type, seed])',
           bootstrap=dict(resamples=1000, seed='20260921+type', stratified=True),
           retrained=['*_aug baselines (5 models x SCMA-1/4 x seeds 0-4)', '7-feature aug XGBoost (ablation)', 'lowdata 10/25/50% runs', 'multiclass nets/trees'],
           checkpoint_sha256=ck, config_entry_sha256=cfg, file_sha256={f: lib.sha_file(P / f) for f in files},
           data_cache_sha256={'scma%d_v2.npz' % t: lib.sha_file(lib.DATA_V2 / ('scma%d_v2.npz' % t)) for t in range(1, 7)})
(lib.ROB / 'ROBUST_MANIFEST.json').write_text(json.dumps(man, indent=1))
print(len(ck))
