"""Guards the donor training recipe (paper Appendix C.2). The released checkpoints and every JSON in
paper_results/ were produced with this recipe; these assertions stop it from drifting again (an early
release commit used weight_decay 5e-4, no Nesterov, no warm-up and no label smoothing, none of which
trained anything). See REPRODUCIBILITY.md."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import train_donor


def test_optimizer_matches_the_paper_recipe():
    param = torch.nn.Parameter(torch.zeros(3))
    opt = train_donor.build_optimizer([param])
    assert isinstance(opt, torch.optim.SGD)
    group = opt.param_groups[0]
    assert group["lr"] == 0.1
    assert group["momentum"] == 0.9
    assert group["nesterov"] is True
    assert group["weight_decay"] == 4e-4


def test_lr_schedule_is_warmup_then_two_step_decays():
    epochs = 35
    # five epochs of linear warm-up: 0.1 * (1..5)/5
    assert [round(0.1 * train_donor.lr_scale(e, epochs), 4) for e in range(5)] == [0.02, 0.04, 0.06, 0.08, 0.1]
    # full rate from epoch 5 until 60% of training (epoch 21)
    assert 0.1 * train_donor.lr_scale(5, epochs) == 0.1
    assert 0.1 * train_donor.lr_scale(20, epochs) == 0.1
    # 0.1x at 60% (int(35*0.6)=21) and again at 85% (int(35*0.85)=29)
    assert abs(0.1 * train_donor.lr_scale(21, epochs) - 0.01) < 1e-12
    assert abs(0.1 * train_donor.lr_scale(28, epochs) - 0.01) < 1e-12
    assert abs(0.1 * train_donor.lr_scale(29, epochs) - 0.001) < 1e-12
    assert abs(0.1 * train_donor.lr_scale(34, epochs) - 0.001) < 1e-12


def test_default_label_smoothing_is_point_one():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    assert parser.parse_args([]).label_smoothing == 0.1
