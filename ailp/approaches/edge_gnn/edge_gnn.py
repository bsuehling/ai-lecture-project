import json
import os
import time
from itertools import product

import torch
import torch.nn.functional as F
from torch import Tensor

from ailp.approaches.edge_gnn.model import EdgePredictor, NodeEncoder
from ailp.approaches.edge_gnn.oracle import oracle_loss
from ailp.approaches.prediction_model import PredictionModel
from ailp.graph import Edge, Graph, Node, Part


class EdgeGnnModel(PredictionModel):

    def __init__(
        self,
        train_path: str | os.PathLike = None,
        test_path: str | os.PathLike = None,
        n_epochs: int = 50,
    ):
        super().__init__(train_path=train_path, test_path=test_path)

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
        train_features = self._prepare_node_features(train_nodes)
        eval_features = self._prepare_node_features(eval_nodes)

        for epoch in range(1, 1 + self._n_epochs):
            start_time = time.perf_counter()
            train_loss = self._train(train_features, train_nodes, train_graphs)
            eval_loss = self._eval(eval_features, eval_nodes, eval_graphs)
            end_time = time.perf_counter() - start_time
            print(
                f"Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, "
                f"Eval Loss: {eval_loss:.4f}, Time: {end_time:.4f}"
            )

    def _train(
        self,
        graph_features: list[tuple[Tensor, Tensor]],
        graph_nodes: list[list[Node]],
        graphs: list[Graph],
    ):
        total_loss = torch.tensor(0)
        self._encoder.train()
        self._predictor.train()
        part_ids = torch.tensor(
            [[n.part.part_id for n in nodes] for nodes in graph_nodes]
        )
        for features, nodes, graph, p_ids in zip(
            graph_features, graph_nodes, graphs, part_ids
        ):
            self._optimizer.zero_grad()
            loss = self._predict(p_ids, features, nodes, graph)
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            total_loss += loss
        return float(total_loss / len(graphs))

    def _eval(
        self,
        graph_features: list[tuple[Tensor, Tensor]],
        graph_nodes: list[list[Node]],
        graphs: list[Graph],
    ):
        total_loss = torch.tensor(0)
        self._encoder.eval()
        self._predictor.eval()
        part_ids = torch.tensor(
            [[n.part.part_id for n in nodes] for nodes in graph_nodes]
        )
        for features, nodes, graph, p_ids in zip(
            graph_features, graph_nodes, graphs, part_ids
        ):
            total_loss += self._predict(p_ids, features, nodes, graph)
        return float(total_loss / len(graphs))

    def _predict(
        self,
        part_ids: Tensor,
        features: tuple[Tensor, Tensor],
        nodes: list[Node],
        graph: Graph,
    ) -> Tensor:
        loss = torch.tensor(0)
        encoded: Tensor = self._encoder(features)
        mask: Tensor = torch.zeros(len(nodes), dtype=bool)
        mask[0] = True
        edges: list[tuple[int, int]] = []
        target = [(n1, n2) for n1, v in graph.edges.items() for n2 in v]
        pools: list[set[Edge]] = []
        visited: list[set[Edge]] = []
        g_nodes: list[Node] = [nodes[0]]
        while not mask.all():
            idx_candidates = torch.tensor(
                list(product(mask.tolist(), (~mask).tolist()))
            )
            candidates = torch.stack([part_ids[mask], part_ids[~mask]]).T
            pred = self._predictor(encoded, mask, idx_candidates, edges)
            start_node = g_nodes[0] if not mask.any() else None
            l, new_edge, pools, visited = oracle_loss(
                graph, target, pred, candidates, pools, visited, start_node
            )
            loss += l
            new_node_idx = next(i for i, n in enumerate(nodes) if n == new_edge[1])
            mask[new_node_idx] = True
            g_nodes.append(new_edge[1])
            edges += [g_nodes.index(new_edge[0]), len(g_nodes) - 1]
        return loss

    def _prepare_node_features(
        self, graph_nodes: list[list[Node]]
    ) -> list[tuple[Tensor, Tensor]]:

        def one_hot(parts: list[Part]):
            part_ids = [self._part_map[str(part.part_id)] for part in parts]
            family_ids = [self._family_map[str(part.family_id)] for part in parts]

            part_ids_encoded = F.one_hot(
                torch.tensor(part_ids), num_classes=len(self._part_map)
            )
            family_ids_encoded = F.one_hot(
                torch.tensor(family_ids), num_classes=len(self._family_map)
            )
            return part_ids_encoded, family_ids_encoded

        return [one_hot([n.part for n in sorted(nodes)]) for nodes in graph_nodes]
