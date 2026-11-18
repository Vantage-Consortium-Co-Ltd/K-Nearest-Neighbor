"""KNN จากศูนย์ด้วย NumPy: ระยะทาง -> k เพื่อนบ้านใกล้สุด -> majority vote (ตรงกับ Section 3 ใน KNN.ipynb)"""

import numpy as np

METRICS = ("euclidean", "manhattan")
TIE_BREAKS = ("smallest", "nearest")


def standardize(data_x, data_y, point_x=None, point_y=None):
    """z-score แต่ละ feature ด้วย mean/std ของ training data (ใช้กับจุดทดสอบด้วยค่าเดียวกัน)

    กันไม่ให้ feature ที่ช่วงตัวเลขกว้างกว่าครอบงำระยะทาง feature ที่ค่าคงที่ (std = 0) จะถูกตั้งเป็น 0
    คืน (x_scaled, y_scaled, point_x_scaled, point_y_scaled)
    """
    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)
    mx, my = x.mean(), y.mean()
    sx, sy = x.std() or 1.0, y.std() or 1.0
    px = None if point_x is None else (point_x - mx) / sx
    py = None if point_y is None else (point_y - my) / sy
    return (x - mx) / sx, (y - my) / sy, px, py


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


def predict_numeric_label(data_x, data_y, data_label, point_x, point_y, k, metric="euclidean",
                          tie_break="smallest"):
    """label ที่ชนะ majority vote ในกลุ่ม k เพื่อนบ้าน

    ถ้าคะแนนเสมอ:
      - "smallest" (ค่าเริ่มต้น เหมือนใน notebook) -> label เลขน้อยสุด
      - "nearest" -> label ที่มีเพื่อนบ้านอยู่ใกล้จุดทดสอบที่สุด (ไม่ขึ้นกับการตั้งเลข label)
    """
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"unknown tie_break {tie_break!r}, choose from {TIE_BREAKS}")
    idx = get_k_nearest_neighbors(data_x, data_y, data_label, point_x, point_y, k, metric)
    neighbor_labels = np.asarray(data_label)[idx]
    labels, counts = np.unique(neighbor_labels, return_counts=True)
    tied = labels[counts == counts.max()]
    if len(tied) == 1 or tie_break == "smallest":
        return int(tied[0])
    # idx เรียงจากใกล้ไปไกลอยู่แล้ว -> label แรกที่อยู่ในกลุ่มเสมอคือ label ที่ใกล้ที่สุด
    return int(next(l for l in neighbor_labels if l in tied))


def loocv_accuracy(data_x, data_y, data_label, k, metric="euclidean", tie_break="smallest"):
    """Leave-one-out accuracy: เอาจุดออกทีละจุด ทำนายด้วยจุดที่เหลือ แล้วนับที่ทายถูก"""
    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)
    lab = np.asarray(data_label)
    n = len(x)
    if not 1 <= k <= n - 1:
        raise ValueError(f"for LOOCV k must be in [1, {n - 1}], got {k}")
    correct = 0
    for i in range(n):
        keep = np.arange(n) != i
        pred = predict_numeric_label(x[keep], y[keep], lab[keep], x[i], y[i], k, metric, tie_break)
        correct += pred == lab[i]
    return correct / n


def choose_k(data_x, data_y, data_label, ks=None, metric="euclidean", tie_break="smallest"):
    """ลองหลายค่า k ด้วย LOOCV คืน (k ที่ดีที่สุด, {k: accuracy})

    ค่าเริ่มต้นลอง k คี่ 1, 3, 5, ... (ลดโอกาสเสมอ) ถ้าคะแนนเท่ากันเลือก k ที่เล็กกว่า
    """
    n = len(data_x)
    ks = list(ks) if ks is not None else list(range(1, n, 2))
    scores = {k: loocv_accuracy(data_x, data_y, data_label, k, metric, tie_break) for k in ks}
    best = max(scores, key=lambda k: (scores[k], -k))
    return best, scores
