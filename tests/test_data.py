import pytest
import yaml

from Process.Data import Data, load_config


def _write(tmp_path, cfg):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


BASE = {"data": {"x": [0, 1, 2], "y": [0, 1, 2], "label": [0, 1, 1]}, "test_point": [1, 1], "k": 3}


def test_dev_config_loads():
    cfg, data = load_config("config_dev.yaml")
    assert len(data) == len(cfg["data"]["x"])
    assert 1 <= cfg["k"] <= len(data)
    assert data.name(0) == cfg["data"]["label_names"][0]


def test_defaults(tmp_path):
    cfg, data = load_config(_write(tmp_path, BASE))
    assert cfg["output_dir"] == "output"
    assert data.name(1) == "1"
    assert len(data.color_list) == 2


def test_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        Data([0, 1], [0], [0, 1])


@pytest.mark.parametrize("k", [0, 4, 1.5])
def test_bad_k(tmp_path, k):
    with pytest.raises(ValueError, match="k must be"):
        load_config(_write(tmp_path, {**BASE, "k": k}))


@pytest.mark.parametrize("missing", ["data", "test_point", "k"])
def test_missing_key(tmp_path, missing):
    cfg = {key: v for key, v in BASE.items() if key != missing}
    with pytest.raises(KeyError, match=missing):
        load_config(_write(tmp_path, cfg))


def test_too_few_colors():
    with pytest.raises(ValueError, match="colors"):
        Data([0, 1, 2], [0, 1, 2], [0, 1, 2], colors=["red"])
