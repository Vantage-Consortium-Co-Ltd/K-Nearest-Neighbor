import numpy as np
import pytest

from Process.Calculate import calculate_distance, get_k_nearest_neighbors, predict_numeric_label

X = np.array([0.0, 1.0, 5.0, 6.0])
Y = np.array([0.0, 0.0, 5.0, 5.0])
L = np.array([0, 0, 1, 1])


def test_distance_is_euclidean():
    d = calculate_distance(X, Y, 3.0, 4.0)
    assert d[0] == pytest.approx(5.0)
    assert d[2] == pytest.approx(np.hypot(2, 1))


def test_neighbors_sorted_nearest_first():
    assert list(get_k_nearest_neighbors(X, Y, L, 0.2, 0.0, 3)) == [0, 1, 2]


@pytest.mark.parametrize("point, expected", [((0.5, 0.5), 0), ((5.5, 4.5), 1)])
def test_majority_vote(point, expected):
    assert predict_numeric_label(X, Y, L, *point, k=3) == expected


def test_k_equals_n_predicts_majority_class():
    labels = np.array([0, 1, 1, 1])
    assert predict_numeric_label(X, Y, labels, 0.0, 0.0, k=4) == 1


def test_tie_goes_to_smallest_label():
    assert predict_numeric_label(X, Y, L, 3.0, 2.5, k=4) == 0


@pytest.mark.parametrize("k", [0, 5, -1])
def test_invalid_k_raises(k):
    with pytest.raises(ValueError):
        get_k_nearest_neighbors(X, Y, L, 0, 0, k)


def test_matches_notebook_example():
    """ตัวอย่างใน KNN.ipynb: จุด (9, 13), k=5 -> Gamer (1)"""
    x = np.array([28, 22, 30, 18, 25, 33, 6, 4, 9, 3, 8, 10, 10, 5, 12, 7, 14, 4, 8], float)
    y = np.array([4, 2.5, 6, 3, 1.5, 5, 7, 5, 9, 4.5, 6, 10, 22, 18, 26, 20, 28, 16.5, 32], float)
    lab = np.array([0] * 6 + [1] * 6 + [2] * 7)
    assert predict_numeric_label(x, y, lab, 9, 13, 5) == 1
