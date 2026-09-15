"""build_ctrgcn must work against a clean upstream checkout: the 49-node graph module has to be
written into the checkout and bound so that CTR-GCN's `import_class("graph.mediapipe49.Graph")`
resolves it. This vendors the minimal upstream layout (a real `graph/tools.get_spatial_graph` and a
small `model.ctrgcn.Model` that resolves the graph exactly the way the official repo does), then
builds the model and runs a forward pass. It also builds twice to catch the stale-cache bug."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from signcanon.model import build_ctrgcn

_TOOLS = '''
import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    Dn = np.zeros_like(A)
    for i in range(A.shape[1]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    return np.stack((I, In, Out))
'''

# A minimal Model that resolves the graph string exactly as the official CTR-GCN does, so the graph
# binding done by build_ctrgcn is what is under test; the forward pass just consumes (N, C, T, V, M).
_MODEL = '''
import torch
import torch.nn as nn


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Model(nn.Module):
    def __init__(self, num_class, num_point, num_person, in_channels, graph, graph_args):
        super().__init__()
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32)
        self.num_point = num_point
        self.fc = nn.Linear(in_channels * num_point, num_class)

    def forward(self, x):
        n, c, t, v, m = x.size()
        pooled = x.mean(2).mean(-1).reshape(n, c * v)
        return self.fc(pooled)
'''


def _vendor_upstream(root: Path):
    (root / "graph").mkdir()
    (root / "graph" / "tools.py").write_text(_TOOLS)      # note: no graph/__init__.py, as upstream
    (root / "model").mkdir()
    (root / "model" / "__init__.py").write_text("")
    (root / "model" / "ctrgcn.py").write_text(_MODEL)


def test_builds_and_runs_a_forward_pass_on_a_clean_checkout(tmp_path):
    _vendor_upstream(tmp_path)
    net = build_ctrgcn(tmp_path, num_classes=226).eval()
    assert net.A.shape == (3, 49, 49)                     # the injected 49-node graph, three matrices
    x = torch.zeros(1, 3, 32, 49, 1)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (1, 226)


def test_a_second_build_in_the_same_process_reloads_the_graph(tmp_path):
    _vendor_upstream(tmp_path)
    build_ctrgcn(tmp_path, num_classes=226)
    net = build_ctrgcn(tmp_path, num_classes=10).eval()   # would reuse a stale module without the fix
    assert net.fc.out_features == 10
    with torch.no_grad():
        assert net(torch.zeros(1, 3, 32, 49, 1)).shape == (1, 10)
