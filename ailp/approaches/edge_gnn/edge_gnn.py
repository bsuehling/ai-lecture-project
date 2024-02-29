import json
import os
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from ailp.approaches.edge_gnn.model import EdgePredictor, NodeEncoder
from ailp.approaches.prediction_model import PredictionModel
from ailp.graph import Graph, Part


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

        self._optimizer = torch.optim.Adam(self._encoder.parameters(), lr=0.01)

        self._n_epochs = n_epochs

    def predict_graph(self, parts: set[Part]) -> Graph:
        return Graph()

    def train(self, train_graphs: list[Graph], eval_graphs: list[Graph]):

        train_features: list[tuple[Tensor, Tensor]] = [
            self._prepare_node_features(set([node.part for node in graph.nodes]))
            for graph in train_graphs
        ]
        eval_features: list[tuple[Tensor, Tensor]] = [
            self._prepare_node_features(set([node.part for node in graph.nodes]))
            for graph in eval_graphs
        ]

        for epoch in range(1, 1 + self._n_epochs):
            start_time = time.perf_counter()
            loss = self._train(train_features, train_graphs)
            train_acc, test_acc = self._eval(eval_features, eval_graphs)
            end_time = time.perf_counter() - start_time
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} "
                f"Eval: {test_acc:.4f}, Time: {end_time:.4f}"
            )

    def _train(self, graph_features: list[tuple[Tensor, Tensor]], graphs: list[Graph]):
        self._encoder.train()
        self._optimizer.zero_grad()
        loss = 0
        for features in graph_features:
            encoded: Tensor = self._encoder(features, [])
            mask = torch.tensor(encoded.shape[0])
            edges = []
            while not mask.all():
                pred = self._predictor(encoded, mask, edges)
                loss += F.nll_loss(pred, torch.tensor())  # TODO
        loss.backward()
        self._optimizer.step()

    def _eval(self, features: list[Graph], graphs: list[Graph]):
        0.0, 0.0

    def _prepare_node_features(self, parts: set[Part]) -> tuple[Tensor, Tensor]:
        part_ids = [self._part_map[str(part.part_id)] for part in parts]
        family_ids = [self._family_map[str(part.family_id)] for part in parts]

        part_ids_encoded = F.one_hot(
            torch.tensor(part_ids), num_classes=len(self._part_map)
        )
        family_ids_encoded = F.one_hot(
            torch.tensor(family_ids), num_classes=len(self._family_map)
        )

        return part_ids_encoded, family_ids_encoded
