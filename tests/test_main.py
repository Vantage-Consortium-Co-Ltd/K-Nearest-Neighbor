"""Smoke test ของ CLI main.py"""

import yaml

import main


def test_cli_default_config(capsys):
    pred = main.run(["--no-plot"])
    out = capsys.readouterr().out
    assert "prediction:" in out and isinstance(pred, int)


def test_cli_overrides(capsys):
    main.run(["--no-plot", "--k", "1", "--point", "7", "7", "--metric", "manhattan"])
    out = capsys.readouterr().out
    assert "k = 1, metric = manhattan" in out
    assert "point  4 (7, 7)" in out


def test_cli_tune_k(capsys):
    main.run(["--no-plot", "--tune-k"])
    assert "<- best" in capsys.readouterr().out


def test_cli_saves_plots(tmp_path, capsys):
    cfg = yaml.safe_load(open("config_dev.yaml", encoding="utf-8"))
    cfg["output_dir"] = str(tmp_path / "plots")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    main.run(["--config", str(path)])
    names = sorted(p.name for p in (tmp_path / "plots").iterdir())
    assert names == ["data.png", "decision_boundary.png", "neighbors.png"]
