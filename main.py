"""รัน KNN แบบ config-driven

    python main.py                       # ใช้ config_dev.yaml
    python main.py --config my.yaml --k 5 --point 3 4
"""

import argparse
from pathlib import Path

import numpy as np

from Process.Calculate import calculate_distance, get_k_nearest_neighbors, predict_numeric_label
from Process.Data import load_config
from Process.Plot import plot_data, plot_decision_boundary, plot_neighbors


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="From-scratch K-Nearest-Neighbor on 2D data")
    p.add_argument("--config", default="config_dev.yaml", help="path to YAML config")
    p.add_argument("--k", type=int, help="override k from config")
    p.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"), help="override test_point")
    p.add_argument("--metric", choices=["euclidean", "manhattan"], help="override distance metric")
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

    dist = calculate_distance(data.x, data.y, px, py, metric)
    idx = get_k_nearest_neighbors(data.x, data.y, data.label, px, py, k, metric)
    pred = predict_numeric_label(data.x, data.y, data.label, px, py, k, metric)

    print(f"{len(data)} training points, test point = ({px}, {py}), k = {k}, metric = {metric}")
    print("nearest neighbors:")
    for rank, i in enumerate(idx, 1):
        print(f"  {rank}. point {i:>2} ({data.x[i]:g}, {data.y[i]:g})  {data.name(data.label[i]):<10} d={dist[i]:.3f}")
    votes = {data.name(l): int(c) for l, c in zip(*np.unique(data.label[idx], return_counts=True))}
    print(f"votes: {votes}")
    print(f"prediction: {pred} -> {data.name(pred)}")

    if not args.no_plot:
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
