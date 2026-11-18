"""รัน KNN แบบ config-driven

    python main.py                       # ใช้ config_dev.yaml
    python main.py --config my.yaml --k 5 --point 3 4
"""

import argparse
from pathlib import Path

import numpy as np

from Process.Calculate import (
    calculate_distance,
    choose_k,
    get_k_nearest_neighbors,
    predict_numeric_label,
    standardize,
)
from Process.Data import load_config
from Process.Plot import plot_data, plot_decision_boundary, plot_neighbors


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="From-scratch K-Nearest-Neighbor on 2D data")
    p.add_argument("--config", default="config_dev.yaml", help="path to YAML config")
    p.add_argument("--k", type=int, help="override k from config")
    p.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"), help="override test_point")
    p.add_argument("--metric", choices=["euclidean", "manhattan"], help="override distance metric")
    p.add_argument("--tie-break", choices=["smallest", "nearest"], help="override vote tie-breaking rule")
    p.add_argument("--standardize", action="store_true", help="z-score features before computing distances")
    p.add_argument("--tune-k", action="store_true", help="pick k by leave-one-out cross-validation")
    p.add_argument("--no-plot", action="store_true", help="skip saving PNG plots")
    return p.parse_args(argv)


def run(argv=None):
    args = parse_args(argv)
    cfg, data = load_config(args.config)
    k = args.k if args.k is not None else cfg["k"]
    if not 1 <= k <= len(data):
        raise SystemExit(f"k must be in [1, {len(data)}], got {k}")
    px, py = args.point if args.point else cfg["test_point"]
    metric = args.metric or cfg["metric"]
    tie_break = args.tie_break or cfg["tie_break"]
    scale = args.standardize or cfg["standardize"]
    # ระยะคำนวณบนพิกัดที่ scale แล้ว (ถ้าเปิด) แต่แสดงผล/กราฟด้วยพิกัดเดิม
    fx, fy, qx, qy = standardize(data.x, data.y, px, py) if scale else (data.x, data.y, px, py)

    if args.tune_k:
        k, scores = choose_k(fx, fy, data.label, metric=metric, tie_break=tie_break)
        print("LOOCV accuracy by k:")
        for kk, acc in scores.items():
            print(f"  k={kk:<3} {acc:.3f}{'  <- best' if kk == k else ''}")

    dist = calculate_distance(fx, fy, qx, qy, metric)
    idx = get_k_nearest_neighbors(fx, fy, data.label, qx, qy, k, metric)
    pred = predict_numeric_label(fx, fy, data.label, qx, qy, k, metric, tie_break)

    print(f"{len(data)} training points, test point = ({px}, {py}), k = {k}, metric = {metric}, standardize = {scale}")
    print("nearest neighbors:")
    for rank, i in enumerate(idx, 1):
        print(f"  {rank}. point {i:>2} ({data.x[i]:g}, {data.y[i]:g})  {data.name(data.label[i]):<10} d={dist[i]:.3f}")
    votes = {data.name(l): int(c) for l, c in zip(*np.unique(data.label[idx], return_counts=True))}
    print(f"votes: {votes}")
    print(f"prediction: {pred} -> {data.name(pred)}")

    if not args.no_plot and scale:
        print("note: plots use raw (unscaled) coordinates; skipped when --standardize is on")
    elif not args.no_plot:
        d = cfg.get("data", {})
        xl, yl = d.get("xlabel", "x"), d.get("ylabel", "y")
        out = Path(cfg["output_dir"])
        for path in (
            plot_data(data, xl, yl, out / "data.png"),
            plot_neighbors(data, (px, py), k, xl, yl, out / "neighbors.png", metric),
            plot_decision_boundary(data, k, xl, yl, out / "decision_boundary.png", metric=metric),
        ):
            print(f"saved {path}")
    return pred


if __name__ == "__main__":
    run()
