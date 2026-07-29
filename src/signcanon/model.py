from __future__ import annotations

import os
import sys
from pathlib import Path

from .skeleton import INWARD_EDGES, NUM_NODES

_GRAPH_MODULE = f'''import numpy as np
from graph import tools

num_node = {NUM_NODES}
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = {[(child + 1, parent + 1) for child, parent in INWARD_EDGES]}
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode="spatial"):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            return tools.get_spatial_graph(num_node, self_link, inward, outward)
        raise ValueError(labeling_mode)
'''


def build_ctrgcn(repo_path: str | Path, num_classes: int):
    repo = Path(repo_path).resolve()
    (repo / "graph" / "mediapipe49.py").write_text(_GRAPH_MODULE)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        from model.ctrgcn import Model
    finally:
        os.chdir(cwd)
    return Model(
        num_class=num_classes,
        num_point=NUM_NODES,
        num_person=1,
        in_channels=3,
        graph="graph.mediapipe49.Graph",
        graph_args={"labeling_mode": "spatial"},
    )
