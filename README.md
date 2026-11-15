# PI-Fusion

PI-Fusion is a detector for stealthy charge-manipulation attacks on electric vehicle charging sessions, where only whole-session labels are available. It fuses two branches that view the same session: a noise-augmented multi-scale attention multiple-instance network (NA-MSC-MIL) that reads the 288-slot power curve, and an XGBoost model on 15 session-level features, eight of which are physics-informed consistency checks between declared and measured energy and timing. The fused probability is `0.2 * p_tree + 0.8 * p_net`, and an alert is raised at 0.5.

The whole model (fitting, calibration, scoring) is in a single file, `pi_fusion.py`.

## Install

Python 3.10 or newer. The model depends on `numpy`, `torch`, `scikit-learn` and `xgboost`:

```
pip install numpy torch scikit-learn xgboost
```

Download `pi_fusion.py` next to your script; there is nothing else to install.

## Usage

Each session is a bag of 288 five-minute slots: `slots` (M, 288, 3) holds standardised power and the sine and cosine of the hour of day, `mask` (M, 288) marks the slots in which the vehicle is charging, and `static` (M, 4) holds standardised connection hour, end time, declared energy and available time.

```python
import numpy as np
from pi_fusion import Bags, Calibration, PIFusion

# Standardisation fitted on benign training sessions (kW, h, kWh)
calib = Calibration.fit(active_power_kw, benign_static_raw)

train = Bags(slots, mask, static, labels)   # arrays already standardised with calib
val = Bags(...)
test = Bags(...)

model = PIFusion(device="cuda").fit(train, val, calib, seed=0)
model.calibrate_weight(val)                 # optional: re-select the fusion weight on validation data
scores = model.predict_proba(test)          # fused probability of manipulation
alerts = model.predict(test)                # scores >= 0.5
print(model.score(test))                    # macro-F1 (needs test.labels)

model.save("pi_fusion.pt")
model = PIFusion.load("pi_fusion.pt")
```

`Bags.from_npz(path, split)` reads `<split>__slots`, `<split>__mask`, `<split>__static` and `<split>__labels` arrays from an `.npz` file. `Calibration.standardise` converts raw power and static fields to the standardised units the model expects. `predict_proba_parts` returns the network and tree probabilities separately.

## Result

On the curve-preserving SCMA-1 demand-manipulation benchmark with chronological splits, PI-Fusion reaches a test macro-F1 of 0.784 (10 seeds). Paired bootstrap intervals show it outperforms the original multi-scale network by 0.011 macro-F1 and the source BiLSTM-attention architecture by 0.026. Under 30% stress noise it keeps 0.736 macro-F1, while physics-feature trees fall from 0.809 to 0.563. Physics-feature trees are more accurate on clean data, and at the measured meter jitter of 0.226% no model changes appreciably.

## Citation

M. Massaoudi and M. Ez Eddin, "Physics-Informed Fusion and Shortcut-Free Benchmarking for Weakly Supervised Detection of Stealthy Charge Manipulation in EV Charging Systems," IEEE Internet of Things Journal.

## License

MIT. See `LICENSE`.

## Contact

Mohamed Massaoudi, mohamed.massaoudi@tamu.edu
