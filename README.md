# MSC-MIL / PI-Fusion

Source code for physics-informed fusion (PI-Fusion) for weakly supervised detection of stealthy EV charge-manipulation attacks: a noise-augmented multi-scale attention multiple-instance network (NA-MSC-MIL) fused with a gradient-boosted tree on physics-informed consistency features.

## Layout
- `src/data/` session dataset and chronological split construction (ACN-Data).
- `src/reproduction/` attacker environment (PPO) and BiLSTM-MIL baseline.
- `src/v2/` curve-preserving attack realization, benchmark, baselines, noise calibration, final evaluation, statistics.
- `src/candidates/` multi-scale MIL and other candidate architectures, screening and tuning.
- `src/benchmarks/` baseline models, ablations, robustness, paired statistics.
- `src/external/` CIC-IDS-2017 and Edge-IIoTset windowed MIL experiments.
- `FINAL_MODEL_CONFIG_v2.yaml` final selected configuration.

## Setup
```
pip install -r requirements.txt
```
Datasets are not included. Download ACN-Data (Caltech, JPL, office001 sessions) into `data/`, and set `CIC_IDS_DIR` and `EDGE_IIOT_CSV` for the external datasets.

## Usage
Run modules from the repository root, e.g. `python -m src.external.prep_cic --W 64`, `python -m src.v2.final.run_eval`.

## License
See `LICENSE`.
