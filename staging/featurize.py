"""Pose featurisation used for every corpus in the paper (verbatim logic from the production
scripts, collected in one dependency-free file).

Input:  MediaPipe Holistic keypoints, (T, 75, 3) as [x, y, z] with a (T, 75) confidence array
        (pose 0..32, left hand 33..53, right hand 54..74; model_complexity = 2).
Output: float32 (2, 32, 49): x/y only, 49 nodes (nose, L/R shoulder, L/R elbow, L/R wrist,
        21 left-hand, 21 right-hand landmarks), centred on the shoulder midpoint and scaled by
        shoulder width per frame, low-confidence points (< 0.1) set to NaN then forward/back-filled,
        resampled to 32 frames, remaining NaN written as 0.
z and the confidences are discarded; there is no rotation, no depth, no appearance.
"""
from __future__ import annotations

import numpy as np

T_FIX = 32
BODY = [0, 11, 12, 13, 14, 15, 16]              # nose, L/R shoulder, L/R elbow, L/R wrist
LEFT_HAND = list(range(33, 54))
RIGHT_HAND = list(range(54, 75))
INDEX = BODY + LEFT_HAND + RIGHT_HAND           # 49 nodes
P_LSH, P_RSH = 1, 2                             # shoulder positions inside the 49-node layout


def canon_frame(xy: np.ndarray) -> np.ndarray:
    """(T, 49, 2): centre on the shoulder midpoint and scale by shoulder width, per frame."""
    lsh, rsh = xy[:, P_LSH, :], xy[:, P_RSH, :]
    centre = (lsh + rsh) / 2.0
    width = np.linalg.norm(lsh - rsh, axis=1, keepdims=True)
    width = np.where(width > 1e-6, width, np.nan)
    return (xy - centre[:, None, :]) / width[:, None, :]


def impute(a: np.ndarray) -> np.ndarray:
    """(T, K): forward/back-fill NaN per column; an all-NaN column becomes zeros."""
    out = a.copy()
    for k in range(out.shape[1]):
        col = out[:, k]
        missing = np.isnan(col)
        if missing.all():
            col[:] = 0.0
            continue
        idx = np.where(~missing)[0]
        col[: idx[0]] = col[idx[0]]
        col[idx[-1] + 1 :] = col[idx[-1]]
        for i in range(idx[0], idx[-1] + 1):
            if missing[i]:
                col[i] = col[i - 1]
        out[:, k] = col
    return out


def resample(a: np.ndarray, t: int = T_FIX) -> np.ndarray:
    """(T, K) -> (t, K) by nearest-index resampling."""
    T = a.shape[0]
    if T == t:
        return a
    if T == 0:
        return np.zeros((t, a.shape[1]), a.dtype)
    return a[np.linspace(0, T - 1, t).round().astype(int)]


def featurize(keypoints: np.ndarray, confidences: np.ndarray | None = None, conf_thr: float = 0.1) -> np.ndarray:
    """(T, 75, 3) [+ (T, 75) confidences] -> (2, 32, 49) float32."""
    xy = keypoints[:, INDEX, :2].astype(np.float64)
    if confidences is not None:
        xy[confidences[:, INDEX] < conf_thr] = np.nan
    xy = canon_frame(xy)
    xy = impute(xy.reshape(xy.shape[0], -1)).reshape(-1, 49, 2)
    xy = resample(xy, T_FIX)
    return np.nan_to_num(xy).transpose(2, 0, 1).astype(np.float32)
