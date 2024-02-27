from itertools import permutations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import RGCNConv


class Model(nn.Module):
    def __init__(self, n_part_ids: int, n_family_ids: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forward_part_id = nn.Sequential(
            nn.Linear(n_part_ids, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.ReLU(),
        )
        self._forward_family_id = nn.Sequential(
            nn.Linear(n_family_ids, 10), nn.ReLU(), nn.Linear(10, 1), nn.ReLU()
        )

        self._conv1 = RGCNConv(2, 8)
        self._conv2 = RGCNConv(8, 16)

    def forward(self, x: tuple[Tensor, Tensor], edges: list[tuple[int, int]]):
        x_part_id = self._forward_part_id(x[0])
        x_family_id = self._forward_family_id(x[1])

        n_nodes = len(x[0].shape[0])
        perms = list(permutations(range(n_nodes), 2))
        edge_index = torch.tensor(perms).T
        edge_type = torch.zeros(n_nodes * (n_nodes - 1))
        edge_mask = [
            k for k, (i, j) in enumerate(perms) if (i, j) in edges or (j, i) in edges
        ]
        edge_type[edge_mask] = 1

        x = torch.cat([x_part_id, x_family_id], dim=-1)

        x = nn.ReLU(self._conv1(x, edge_index, edge_type=edge_type))
        x = nn.ReLU(self._conv2(x, edge_index, edge_type=edge_type))

        # TODO: for each edge, combine the feature of the 2 neighboring nodes

        return x
