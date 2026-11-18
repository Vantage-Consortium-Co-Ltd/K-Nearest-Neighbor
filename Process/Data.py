"""โหลดและตรวจสอบชุดข้อมูลจาก config YAML"""

from pathlib import Path

import numpy as np
import yaml
from matplotlib.colors import ListedColormap


class Data:
    """เก็บพิกัด (x, y), label และ colormap สำหรับ plot — บทบาทเดียวกับ class Data ใน KNN.ipynb"""

    def __init__(self, data_x, data_y, data_label, colors=None, label_names=None):
        self.x = np.asarray(data_x, dtype=float)
        self.y = np.asarray(data_y, dtype=float)
        self.label = np.asarray(data_label, dtype=int)
        if not (len(self.x) == len(self.y) == len(self.label)):
            raise ValueError(
                f"x, y, label must have the same length (got {len(self.x)}, {len(self.y)}, {len(self.label)})"
            )
        if len(self.x) == 0:
            raise ValueError("dataset is empty")
        n_classes = int(self.label.max()) + 1
        self.color_list = list(colors) if colors else ["red", "blue", "green", "orange", "purple"][:n_classes]
        if len(self.color_list) < n_classes:
            raise ValueError(f"need at least {n_classes} colors, got {len(self.color_list)}")
        self.cmap = ListedColormap(self.color_list)
        self.label_names = {int(k): str(v) for k, v in (label_names or {}).items()}

    def __len__(self):
        return len(self.x)

    def name(self, label):
        return self.label_names.get(int(label), str(label))


def load_config(path):
    """อ่าน YAML แล้วคืน (config dict, Data) — ตรวจ key ที่จำเป็นและค่า k"""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for key in ("data", "test_point", "k"):
        if key not in cfg:
            raise KeyError(f"{path.name}: missing required key '{key}'")
    d = cfg["data"]
    for key in ("x", "y", "label"):
        if key not in d:
            raise KeyError(f"{path.name}: missing required key 'data.{key}'")

    data = Data(d["x"], d["y"], d["label"], d.get("colors"), d.get("label_names"))

    k = cfg["k"]
    if not isinstance(k, int) or k < 1 or k > len(data):
        raise ValueError(f"k must be an integer in [1, {len(data)}], got {k!r}")
    if len(cfg["test_point"]) != 2:
        raise ValueError("test_point must be [x, y]")

    cfg.setdefault("metric", "euclidean")
    if cfg["metric"] not in ("euclidean", "manhattan"):
        raise ValueError(f"metric must be 'euclidean' or 'manhattan', got {cfg['metric']!r}")
    cfg.setdefault("tie_break", "smallest")
    if cfg["tie_break"] not in ("smallest", "nearest"):
        raise ValueError(f"tie_break must be 'smallest' or 'nearest', got {cfg['tie_break']!r}")
    cfg.setdefault("standardize", False)
    if not isinstance(cfg["standardize"], bool):
        raise ValueError(f"standardize must be true/false, got {cfg['standardize']!r}")
    cfg.setdefault("output_dir", "output")
    return cfg, data
