"""เทสต์ฟีเจอร์ที่เพิ่มจาก notebook: manhattan, tie-break, standardize, LOOCV"""

import numpy as np
import pytest

from Process.Calculate import (
    calculate_distance,
    choose_k,
    get_k_nearest_neighbors,
    loocv_accuracy,
    predict_numeric_label,
    standardize,
)


def test_manhattan_distance():
    d = calculate_distance([0, 3], [0, -4], 0, 0, metric="manhattan")
    assert list(d) == [0, 7]


def test_metric_changes_ranking():
    # จุด A อยู่ห่างแนวทแยง (3,3), จุด B อยู่แนวตรง (4,0)
    x, y = np.array([3.0, 4.0]), np.array([3.0, 0.0])
    assert get_k_nearest_neighbors(x, y, [0, 1], 0, 0, 1, "euclidean")[0] == 1  # 4.24 vs 4
    assert get_k_nearest_neighbors(x, y, [0, 1], 0, 0, 1, "manhattan")[0] == 1  # 6 vs 4
    x2 = np.array([2.0, 3.5])
    y2 = np.array([2.0, 0.0])
    assert get_k_nearest_neighbors(x2, y2, [0, 1], 0, 0, 1, "euclidean")[0] == 0  # 2.83 vs 3.5
    assert get_k_nearest_neighbors(x2, y2, [0, 1], 0, 0, 1, "manhattan")[0] == 1  # 4 vs 3.5


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="metric"):
        calculate_distance([0], [0], 0, 0, metric="cosine")


def test_tie_break_nearest_vs_smallest():
    # label 1 อยู่ใกล้กว่า แต่คะแนนเสมอ 1:1
    x, y, lab = np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([1, 0])
    assert predict_numeric_label(x, y, lab, 0, 0, 2, tie_break="smallest") == 0
    assert predict_numeric_label(x, y, lab, 0, 0, 2, tie_break="nearest") == 1


def test_tie_break_ignored_when_clear_winner():
    x, y, lab = np.array([1.0, 2.0, 3.0]), np.zeros(3), np.array([1, 0, 0])
    assert predict_numeric_label(x, y, lab, 0, 0, 3, tie_break="nearest") == 0


def test_unknown_tie_break_raises():
    with pytest.raises(ValueError, match="tie_break"):
        predict_numeric_label([0], [0], [0], 0, 0, 1, tie_break="random")


def test_standardize_zero_mean_unit_std():
    xs, ys, px, py = standardize([0, 10, 20], [100, 200, 300], 10, 300)
    assert xs.mean() == pytest.approx(0) and xs.std() == pytest.approx(1)
    assert ys.std() == pytest.approx(1)
    assert px == pytest.approx(0) and py == pytest.approx(ys[2])


def test_standardize_constant_feature():
    xs, _, px, _ = standardize([5, 5, 5], [1, 2, 3], 5, 2)
    assert np.all(xs == 0) and px == 0


def test_standardize_fixes_scale_domination():
    # x แยกคลาสได้ (0 vs 1) แต่ y เป็นหลักร้อยหลักพัน -> ถ้าไม่ scale ระยะจะดูแค่ y
    x = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0.0, 1000.0, 500.0, 1500.0])
    lab = np.array([0, 0, 1, 1])
    assert predict_numeric_label(x, y, lab, 0.0, 500, 1) == 1          # ไม่ scale: ตามค่า y ไปเจอคลาส 1
    xs, ys, px, py = standardize(x, y, 0.0, 500)
    assert predict_numeric_label(xs, ys, lab, px, py, 1) == 0          # scale แล้ว: x มีน้ำหนักเท่ากัน -> คลาส 0


def test_loocv_perfectly_separable():
    x = np.array([0, 0.5, 1, 10, 10.5, 11], float)
    y = np.zeros(6)
    lab = np.array([0, 0, 0, 1, 1, 1])
    assert loocv_accuracy(x, y, lab, 1) == 1.0
    best, scores = choose_k(x, y, lab)
    assert best == 1 and set(scores) == {1, 3, 5}


def test_loocv_rejects_k_equal_n():
    with pytest.raises(ValueError):
        loocv_accuracy([0, 1], [0, 1], [0, 1], 2)
