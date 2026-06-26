"""
Metrics and evaluation utilities.
Computes detection latency, F1 score, ROC curve data,
trust MAE, and overhead statistics used across all experiments.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, roc_curve, auc, confusion_matrix, precision_score, recall_score
)


def detection_latency(
    trust_history: list[float],
    malicious_start_round: int,
    theta: float,
    is_malicious: bool = True,
) -> int:
    """
    Rounds from malicious_start_round until trust drops below theta.
    Returns -1 if detection never happens within the simulation.
    """
    for r in range(malicious_start_round, len(trust_history)):
        if trust_history[r] < theta:
            return r - malicious_start_round
    return -1  # not detected


def detection_latency_all(
    trust_histories: dict,      # node_id -> list of trust score per round
    malicious_ids: set,
    malicious_start_round: int,
    theta: float,
) -> list[int]:
    """Compute detection latency for all malicious nodes."""
    latencies = []
    for nid in malicious_ids:
        if nid in trust_histories:
            lat = detection_latency(
                trust_histories[nid], malicious_start_round, theta
            )
            if lat >= 0:
                latencies.append(lat)
    return latencies


def classification_metrics(
    trust_scores: np.ndarray,   # shape (N,)
    true_malicious: np.ndarray, # shape (N,) bool
    theta: float,
) -> dict:
    """
    Threshold trust scores at theta to get binary predictions.
    Returns precision, recall, F1, confusion matrix entries.
    """
    predicted_malicious = trust_scores < theta
    y_true = true_malicious.astype(int)
    y_pred = predicted_malicious.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn + 1e-9)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": float(fpr),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def roc_data(
    trust_scores: np.ndarray,
    true_malicious: np.ndarray,
) -> dict:
    """Compute ROC curve data for a trust model."""
    y_true = true_malicious.astype(int)
    # Invert trust so higher score = more likely malicious for ROC
    scores = 1.0 - trust_scores
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(), "auc": float(roc_auc)}


def trust_mae(
    trust_scores: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Mean absolute error between trust scores and ground truth quality."""
    return float(np.mean(np.abs(trust_scores - ground_truth)))


def summarize_runs(results: list[dict], key: str) -> dict:
    """
    Aggregate a metric across multiple runs.
    Returns mean, std, 95% CI half-width.
    """
    values = [r[key] for r in results if key in r and r[key] is not None]
    if not values:
        return {"mean": None, "std": None, "ci95": None}
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    ci95 = 1.96 * std / np.sqrt(len(arr))
    return {"mean": mean, "std": std, "ci95": float(ci95), "n": len(arr)}


def trust_divergence(T1: np.ndarray, T2: np.ndarray) -> float:
    """
    Mean absolute difference between two trust matrices.
    Used to measure divergence during partition (Exp 2).
    """
    return float(np.mean(np.abs(T1 - T2)))
