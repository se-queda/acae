import numpy as np
from sklearn.metrics import roc_auc_score


def _segments_from_labels(labels):
    """
    Extract contiguous anomalous segments from binary labels.
    Returns a list of (start_idx, end_idx) inclusive.
    """
    segments = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            start = i
            while i + 1 < n and labels[i + 1] == 1:
                i += 1
            end = i
            segments.append((start, end))
        i += 1
    return segments


def compute_auc(scores, labels):
    """Standard ROC-AUC on point-wise scores."""
    if np.all(labels == 0) or np.all(labels == 1):
        # AUC is undefined if only one class present; return 0.5 by convention
        return 0.5
    return roc_auc_score(labels, scores)


# ----------------------------------------------------------------------
# Fc1: Composite F-score (Garg et al., 2021)
# ----------------------------------------------------------------------
def compute_fc1(preds, labels):
    """
    Composite F-score (Fc1) implementation.

    - Event-wise recall: fraction of ground-truth anomalous segments
      that intersect with at least one predicted positive point.
    - Time-wise precision: point-wise precision over all time steps.

    Fc1 = 2 * (event_recall * time_precision) / (event_recall + time_precision)
    """

    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    assert labels.shape == preds.shape

    # ---- Event-wise recall ----
    segments = _segments_from_labels(labels)
    if len(segments) == 0:
        # No anomalies in ground truth; Fc1 defined as 0 here
        return 0.0

    detected_segments = 0
    for (s, e) in segments:
        # If any prediction in this segment is 1, count segment as detected
        if np.any(preds[s:e + 1] == 1):
            detected_segments += 1

    event_recall = detected_segments / len(segments)

    # ---- Time-wise precision ----
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))

    if tp + fp == 0:
        time_precision = 0.0
    else:
        time_precision = tp / (tp + fp)

    if event_recall + time_precision == 0:
        return 0.0

    fc1 = 2 * event_recall * time_precision / (event_recall + time_precision)
    return fc1


# ----------------------------------------------------------------------
# PA%K (Kim et al., 2022)
# ----------------------------------------------------------------------
def _point_adjustment_at_k(preds, labels, K):
    """
    Apply point adjustment with threshold K (0–1).
    Within each ground-truth anomalous segment, if the fraction of
    predicted positives exceeds K, mark the ENTIRE segment as predicted 1.
    Otherwise, set predictions in that segment to 0.

    Returns adjusted predictions (same length as preds).
    """
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    assert labels.shape == preds.shape

    adjusted = preds.copy()
    segments = _segments_from_labels(labels)

    for (s, e) in segments:
        seg_len = e - s + 1
        if seg_len <= 0:
            continue
        num_pred_pos = np.sum(preds[s:e + 1] == 1)
        frac = num_pred_pos / seg_len
        if frac >= K:
            # Mark entire segment as predicted anomalous
            adjusted[s:e + 1] = 1
        else:
            # Clear predictions in this segment
            adjusted[s:e + 1] = 0
    return adjusted


def _f1_from_preds(preds, labels):
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    if tp == 0 and fp == 0 and fn == 0:
        return 0.0

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_pa_k_auc(scores, labels, base_threshold=None, k_values=None):
    """
    PA%K as described by Kim et al. (2022):

    - For each K in k_values:
        1) Threshold scores into preds (if base_threshold is None, use
           median or another heuristic; you can also pass a pre-optimized threshold).
        2) Apply point adjustment at K: segments become +1 if fraction
           of positives in the segment >= K.
        3) Compute point-wise F1 between adjusted preds and labels.
    - Return the area under the K-F1 curve (simple trapezoidal integration).

    Args:
        scores: 1D array of anomaly scores.
        labels: 1D array of 0/1 labels.
        base_threshold: scalar thr for scores -> preds. If None, uses median.
        k_values: iterable of K in [0, 1]. If None, default 0.1..1.0.

    Returns:
        pa_auc: scalar, approximate area under PA%K curve.
        ks: K values used.
        f1s: F1 at each K.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)

    if base_threshold is None:
        # Heuristic: median score as base threshold
        base_threshold = float(np.median(scores))

    preds_raw = (scores > base_threshold).astype(int)

    if k_values is None:
        # Example grid: 0.1, 0.2, ..., 1.0
        k_values = np.linspace(0.1, 1.0, 10)

    f1s = []
    for K in k_values:
        adjusted = _point_adjustment_at_k(preds_raw, labels, K)
        f1 = _f1_from_preds(adjusted, labels)
        f1s.append(f1)

    ks = np.array(k_values, dtype=float)
    f1s = np.array(f1s, dtype=float)

    # Area under K-F1 curve using trapezoidal rule
    pa_auc = np.trapz(f1s, ks) / (ks[-1] - ks[0])  # normalize by K-range
    return pa_auc, ks, f1s
