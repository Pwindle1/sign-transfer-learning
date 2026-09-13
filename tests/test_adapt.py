import numpy as np
import torch

from signcanon.adapt import full_finetune


class TinyNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.body = torch.nn.Linear(8, 4)
        self.fc = torch.nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(torch.relu(self.body(x)))


def test_empty_donor_state_trains_from_random_initialisation():
    torch.manual_seed(0)
    donor = TinyNet(3).state_dict()
    x = torch.randn(16, 8)
    y = np.array([0, 1, 2, 0] * 4)
    torch.manual_seed(1)
    net = full_finetune({}, 3, x, y, TinyNet, epochs=1)
    assert not torch.allclose(net.body.weight.cpu(), donor["body.weight"])
    torch.manual_seed(1)
    loaded = full_finetune(donor, 3, x, y, TinyNet, epochs=0)
    assert torch.allclose(loaded.body.weight.cpu(), donor["body.weight"])


def test_checkpointed_budgets_are_read_off_one_trajectory():
    torch.manual_seed(2)
    x = torch.randn(16, 8)
    y = np.array([0, 1] * 8)
    net, scores = full_finetune({}, 2, x, y, TinyNet, epochs=4, eval_at=(2, 4), eval_fn=lambda n: 0.5)
    assert sorted(scores) == [2, 4]
