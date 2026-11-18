"""กราฟ: ข้อมูลดิบ, วงเพื่อนบ้านของจุดทดสอบ และ decision boundary — บันทึกเป็น PNG"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # รันบน server/CI ได้โดยไม่ต้องมีหน้าจอ
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from Process.Calculate import calculate_distance, get_k_nearest_neighbors, predict_numeric_label


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _scatter(ax, data):
    for lbl in np.unique(data.label):
        m = data.label == lbl
        ax.scatter(data.x[m], data.y[m], c=data.color_list[lbl], edgecolor="k", label=data.name(lbl))


def plot_data(data, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter(ax, data)
    ax.set(xlabel=xlabel, ylabel=ylabel, title="Training data")
    ax.legend()
    return _save(fig, path)


def plot_neighbors(data, point, k, xlabel, ylabel, path, metric="euclidean"):
    """วาดจุดทดสอบ เส้นไปหา k เพื่อนบ้าน และวงกลมรัศมีเท่ากับเพื่อนบ้านที่ไกลสุด"""
    px, py = point
    idx = get_k_nearest_neighbors(data.x, data.y, data.label, px, py, k, metric)
    radius = calculate_distance(data.x, data.y, px, py, metric)[idx].max()
    pred = predict_numeric_label(data.x, data.y, data.label, px, py, k, metric)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter(ax, data)
    for i in idx:
        ax.plot([px, data.x[i]], [py, data.y[i]], "k--", lw=0.8)
    if metric == "manhattan":  # "วงกลม" ของ manhattan คือสี่เหลี่ยมขนมเปียกปูน
        ax.add_patch(Polygon([(px + radius, py), (px, py + radius), (px - radius, py), (px, py - radius)],
                             fill=False, ls=":", color="gray"))
    else:
        ax.add_patch(Circle((px, py), radius, fill=False, ls=":", color="gray"))
    ax.scatter([px], [py], marker="*", s=300, c=data.color_list[pred], edgecolor="k", label="test point")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=f"k={k}, {metric} -> predicted {data.name(pred)}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    return _save(fig, path)


def plot_decision_boundary(data, k, xlabel, ylabel, path, resolution=150, metric="euclidean"):
    """ทาสีทุกพื้นที่ตาม class ที่ KNN จะทำนาย"""
    pad_x = (data.x.max() - data.x.min()) * 0.1 + 1
    pad_y = (data.y.max() - data.y.min()) * 0.1 + 1
    gx, gy = np.meshgrid(
        np.linspace(data.x.min() - pad_x, data.x.max() + pad_x, resolution),
        np.linspace(data.y.min() - pad_y, data.y.max() + pad_y, resolution),
    )
    z = np.array([
        predict_numeric_label(data.x, data.y, data.label, x, y, k, metric) for x, y in zip(gx.ravel(), gy.ravel())
    ]).reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(gx, gy, z, levels=np.arange(len(data.color_list) + 1) - 0.5, cmap=data.cmap, alpha=0.25)
    _scatter(ax, data)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=f"Decision boundary (k={k}, {metric})")
    ax.legend()
    return _save(fig, path)
