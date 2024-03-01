import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import Tensor


class NodeEncoder(nn.Module):
    def __init__(self, n_part_ids: int, n_family_ids: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encode_part_id = nn.Sequential(
            nn.Linear(n_part_ids, 100), nn.ReLU(), nn.Linear(100, 8), nn.ReLU()
        )
        self._encode_family_id = nn.Sequential(
            nn.Linear(n_family_ids, 10), nn.ReLU(), nn.Linear(50, 8), nn.ReLU()
        )

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        x_part_id = self._encode_part_id(x[0])
        x_family_id = self._encode_family_id(x[1])
        out = torch.cat([x_part_id, x_family_id], dim=-1)
        return out


class EdgePredictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conv1 = gnn.GraphConv(2, 8)
        self._conv2 = gnn.GraphConv(8, 16)
        self._fc_unconnected = nn.Sequential(nn.Linear(8, 16))
        self._fc = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(
        self, x: Tensor, mask: Tensor, candidates: Tensor, edges: Tensor
    ) -> Tensor:
        x_unconnected = self._fc_unconnected(x[~mask])
        x_graph = F.relu(self._conv1(x[mask], edges))
        x_graph = F.relu(self._conv2(x_graph, edges))
        edge_candidates = torch.cat(
            (x_graph[candidates[:, 0]], x_unconnected[candidates[:, 1]])
        )
        out = torch.squeeze(self._fc(edge_candidates))
        return out
