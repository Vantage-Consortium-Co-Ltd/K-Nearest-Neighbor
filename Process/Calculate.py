"""KNN จากศูนย์ด้วย NumPy: ระยะทาง -> k เพื่อนบ้านใกล้สุด -> majority vote (ตรงกับ Section 3 ใน KNN.ipynb)"""

import numpy as np


def calculate_distance(data_x, data_y, point_x, point_y):
    """Euclidean distance จาก (point_x, point_y) ไปยังทุกจุดใน training data"""
    return np.sqrt((np.asarray(data_x) - point_x) ** 2 + (np.asarray(data_y) - point_y) ** 2)


def get_k_nearest_neighbors(data_x, data_y, data_label, point_x, point_y, k):
    """คืน index ของ k จุดที่ใกล้ที่สุด (เรียงจากใกล้ไปไกล)"""
    if not 1 <= k <= len(data_x):
        raise ValueError(f"k must be in [1, {len(data_x)}], got {k}")
    distances = calculate_distance(data_x, data_y, point_x, point_y)
    return np.argsort(distances, kind="stable")[:k]


def predict_numeric_label(data_x, data_y, data_label, point_x, point_y, k):
    """label ที่ชนะ majority vote ในกลุ่ม k เพื่อนบ้าน (เสมอ -> label เลขน้อยสุด)"""
    idx = get_k_nearest_neighbors(data_x, data_y, data_label, point_x, point_y, k)
    labels, counts = np.unique(np.asarray(data_label)[idx], return_counts=True)
    return int(labels[np.argmax(counts)])
