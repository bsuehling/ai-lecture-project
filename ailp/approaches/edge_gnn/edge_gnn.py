import json
import os
import time
from itertools import product

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ailp.approaches.edge_gnn.dataset import EdgeGnnDataset
from ailp.approaches.edge_gnn.model import EdgePredictor, NodeEncoder
from ailp.approaches.edge_gnn.oracle import oracle_loss
from ailp.approaches.prediction_model import PredictionModel
from ailp.graph import Edge, Graph, Node, Part
from ailp.utils.path import TorchWrapper, save_model


class EdgeGnnModel(PredictionModel):

    def __init__(
        self,
        train_path: str | os.PathLike = None,
        test_path: str | os.PathLike = None,
        n_epochs: int = 50,
    ):
        super().__init__(train_path=train_path, test_path=test_path)
        torch.autograd.set_detect_anomaly(True)

        with open(os.path.join("data", "mappings", "part_map.json"), "r") as f:
            self._part_map: dict[str, int] = json.load(f)
        with open(os.path.join("data", "mappings", "family_map.json"), "r") as f:
            self._family_map: dict[str, int] = json.load(f)
        self._num_part_ids = len(self._part_map)
        self._num_family_ids = len(self._family_map)

        self._encoder = NodeEncoder(self._num_part_ids, self._num_family_ids)
        self._predictor = EdgePredictor()

        params = list(self._encoder.parameters()) + list(self._predictor.parameters())
        self._optimizer = torch.optim.Adam(params, lr=0.01)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer)

        self._n_epochs = n_epochs

    def predict_graph(self, parts: set[Part]) -> Graph:
        return Graph()

    def train(self, train_graphs: list[Graph], eval_graphs: list[Graph]):
        train_nodes = [sorted(graph.nodes) for graph in train_graphs]
        eval_nodes = [sorted(graph.nodes) for graph in eval_graphs]
        train_feat = self._prepare_node_features(train_nodes)
        eval_feat = self._prepare_node_features(eval_nodes)
        train_ids = [[n.part_id for n in nodes] for nodes in train_nodes]
        eval_ids = [[n.part_id for n in nodes] for nodes in eval_nodes]

        train_data = EdgeGnnDataset(train_feat, train_nodes, train_graphs, train_ids)
        eval_data = EdgeGnnDataset(eval_feat, eval_nodes, eval_graphs, eval_ids)
        train_loader = DataLoader(
            train_data, batch_size=10, shuffle=True, collate_fn=lambda x: x
        )
        eval_loader = DataLoader(
            eval_data, batch_size=10, shuffle=False, collate_fn=lambda x: x
        )

        for epoch in range(1, 1 + self._n_epochs):
            start_time = time.perf_counter()
            train_loss = self._train(train_loader) / len(train_data)
            eval_loss = self._eval(eval_loader) / len(eval_data)
            self._save_state(epoch)
            end_time = time.perf_counter() - start_time
            print(
                f"Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, "
                f"Eval Loss: {eval_loss:.4f}, Time: {end_time:.4f}"
            )

    def _train(self, data_loader: DataLoader) -> float:
        total_loss = 0.0
        self._encoder.train()
        self._predictor.train()
        for batch in data_loader:
            self._optimizer.zero_grad()
            loss = self._predict_multi(batch)
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item()
        self._scheduler.step(loss)
        return total_loss

    def _eval(self, data_loader: DataLoader):
        total_loss = 0.0
        self._encoder.eval()
        self._predictor.eval()
        for batch in data_loader:
            loss = self._predict_multi(batch)
            total_loss += loss.item()
        return total_loss

    def _predict_multi(
        self, batch: list[tuple[tuple[Tensor, Tensor], list[Node], Graph, list[int]]]
    ) -> Tensor:
        loss = 0
        for b in batch:
            loss += self._predict(*b)
        return loss

    def _predict(
        self,
        features: tuple[Tensor, Tensor],
        nodes: list[Node],
        graph: Graph,
        part_ids: Tensor,
    ) -> Tensor:
        loss = 0
        encoded: Tensor = self._encoder(features)
        mask: Tensor = torch.zeros(len(nodes), dtype=bool)
        mask[0] = True
        edges: list[tuple[int, int]] = []
        pools: list[set[Edge]] = []
        visited: list[set[Edge]] = []
        g_nodes: list[Node] = [nodes[0]]
        while not mask.all():
            ones = mask.nonzero().flatten().tolist()
            zeros = (~mask).nonzero().flatten().tolist()
            idx_cands = torch.tensor(list(product(ones, zeros)))
            p_ids = torch.tensor(part_ids)
            candidates = torch.tensor(list(product(p_ids[mask], p_ids[~mask])))
            x_u = encoded[~mask.clone()]
            x_g = encoded[mask.clone()]
            t_edges = torch.tensor(edges, dtype=int)
            if len(edges) > 0:
                t_edges = t_edges.T
            pred = self._predictor(x_u, x_g, idx_cands, t_edges)
            start_node = g_nodes[0] if not mask.any() else None
            l, new_edge, pools, visited = oracle_loss(
                graph, nodes, pred, candidates, idx_cands, pools, visited, start_node
            )
            loss += l
            new_node_idx = next(i for i, n in enumerate(nodes) if n == new_edge[1])
            mask[new_node_idx] = True
            g_nodes.append(new_edge[1])
            e0 = g_nodes.index(new_edge[0])
            e1 = len(g_nodes) - 1
            edges += [(e0, e1), (e1, e0)]
        return loss

    def _prepare_node_features(
        self, graph_nodes: list[list[Node]]
    ) -> list[tuple[Tensor, Tensor]]:

        def one_hot(parts: list[Part]):
            part_ids = [self._part_map[str(part.part_id)] for part in parts]
            family_ids = [self._family_map[str(part.family_id)] for part in parts]

            part_ids_encoded = F.one_hot(
                torch.tensor(part_ids), num_classes=len(self._part_map)
            ).float()
            family_ids_encoded = F.one_hot(
                torch.tensor(family_ids), num_classes=len(self._family_map)
            ).float()
            return part_ids_encoded, family_ids_encoded

        return [one_hot([n.part for n in sorted(nodes)]) for nodes in graph_nodes]

    def _save_state(self, epoch: int):
        state = {
            "encoder": self._encoder.state_dict(),
            "predictor": self._predictor.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
        }
        save_model(TorchWrapper(state), self._train_path, epoch=epoch)

    def _load_state(self):
        self._encoder.load_state_dict(self._model["encoder"])
        self._predictor.load_state_dict(self._model["predictor"])
        self._optimizer.load_state_dict(self._model["optimizer"])
        self._scheduler.load_state_dict(self._model["scheduler"])
