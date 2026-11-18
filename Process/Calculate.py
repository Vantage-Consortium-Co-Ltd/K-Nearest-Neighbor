"""KNN จากศูนย์ด้วย NumPy: ระยะทาง -> k เพื่อนบ้านใกล้สุด -> majority vote (ตรงกับ Section 3 ใน KNN.ipynb)"""

import numpy as np

METRICS = ("euclidean", "manhattan")


def calculate_distance(data_x, data_y, point_x, point_y, metric="euclidean"):
    """ระยะจาก (point_x, point_y) ไปยังทุกจุดใน training data

    euclidean = sqrt(dx^2 + dy^2), manhattan = |dx| + |dy|
    """
    dx = np.asarray(data_x, dtype=float) - point_x
    dy = np.asarray(data_y, dtype=float) - point_y
    if metric == "euclidean":
        return np.sqrt(dx**2 + dy**2)
    if metric == "manhattan":
        return np.abs(dx) + np.abs(dy)
    raise ValueError(f"unknown metric {metric!r}, choose from {METRICS}")


def get_k_nearest_neighbors(data_x, data_y, data_label, point_x, point_y, k, metric="euclidean"):
    """คืน index ของ k จุดที่ใกล้ที่สุด (เรียงจากใกล้ไปไกล)"""
    if not 1 <= k <= len(data_x):
        raise ValueError(f"k must be in [1, {len(data_x)}], got {k}")
    distances = calculate_distance(data_x, data_y, point_x, point_y, metric)
    return np.argsort(distances, kind="stable")[:k]


def predict_numeric_label(data_x, data_y, data_label, point_x, point_y, k, metric="euclidean"):
    """label ที่ชนะ majority vote ในกลุ่ม k เพื่อนบ้าน (เสมอ -> label เลขน้อยสุด)"""
    idx = get_k_nearest_neighbors(data_x, data_y, data_label, point_x, point_y, k, metric)
    labels, counts = np.unique(np.asarray(data_label)[idx], return_counts=True)
    return int(labels[np.argmax(counts)])
