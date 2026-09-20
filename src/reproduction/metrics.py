"""Metrics per SOURCE_PAPER_AUDIT.md Sec. 9 (Eq. 12-13) plus standard extras
needed for the broader study (macro-F1, PR-AUC, CI) that the paper omits."""
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    fa_rate = fp / max(tn + fp, 1)  # Eq. 13, paper's FA definition
    out = {
        "accuracy": acc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_alarm_rate": fa_rate,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["pr_auc"] = average_precision_score(y_true, y_score)
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return out


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """(F1_positive + F1_negative) / 2 -- binary_metrics() above only returns
    the positive-class F1 (sklearn's default 'binary' average, Eq.12-13 style,
    matching the source paper). Stage 2 (Master Prompt Sec.9) needs macro-F1
    as its primary early-stopping/ranking metric across the 3 candidates, so
    both classes count equally regardless of the ~90/10 imbalance. One-line
    addition: sklearn's average='macro' for binary y IS exactly this mean."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def aggregate_seeds(metric_dicts: list[dict]) -> dict:
    """Mean/std/median/95% CI across seed-level metric dicts (Master Prompt Sec. 22)."""
    keys = [k for k in metric_dicts[0] if isinstance(metric_dicts[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_dicts], dtype=float)
        n = len(vals)
        mean, std = vals.mean(), vals.std(ddof=1) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 1 else 0.0
        agg[k] = {
            "mean": float(mean), "std": float(std), "median": float(np.median(vals)),
            "ci95_lo": float(mean - 1.96 * se), "ci95_hi": float(mean + 1.96 * se), "n": n,
        }
    return agg
