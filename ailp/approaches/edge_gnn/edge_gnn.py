import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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

        self._model = nn.Module()

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

        print(train_features)
        print(eval_features)

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
