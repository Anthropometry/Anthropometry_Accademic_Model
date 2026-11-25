"""End-to-end smoke test: train for one epoch on fake data, then reload and evaluate.

This proves the pipeline runs; it says nothing about accuracy (that needs real BodyM on a GPU).
"""

import pickle

import pytest
import torch

from anthro import checkpoint
from anthro import train as train_mod
from anthro.cli import main
from anthro.config import TrainConfig


@pytest.fixture
def trained(bodym, tmp_path):
    cfg = TrainConfig(img_h=64, img_w=32, epochs=1, batch_size=8, pretrained=False, num_workers=0,
                      val_fraction=0.25)
    return train_mod.run(str(bodym), str(tmp_path / "run"), cfg, device="cpu")


def test_train_saves_a_checkpoint_that_loads_safely(trained):
    net, cfg, xs, ys, ck = checkpoint.load(trained)        # uses weights_only=True inside
    assert cfg.img_h == 64 and len(ck["measurements"]) == 14
    assert (trained.parent / "train_log.json").exists()


def test_checkpoint_with_code_is_refused(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("this would have run",))
    p = tmp_path / "evil.pt"
    with open(p, "wb") as f:
        pickle.dump({"x": Evil()}, f)
    with pytest.raises(pickle.UnpicklingError):
        torch.load(p, weights_only=True)


def test_cli_evaluate_runs(bodym, trained, capsys):
    main(["evaluate", "--data", str(bodym), "--ckpt", str(trained), "--test", "testA"])
    out = capsys.readouterr().out
    assert "baseline_MAE_cm" in out and "photo_MAE_cm" in out
    assert (trained.parent / "eval_report.json").exists()
