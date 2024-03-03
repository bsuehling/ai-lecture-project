from torch import Tensor
from torch.utils.data import Dataset

from ailp.graph import Graph, Node


class EdgeGnnDataset(Dataset):
    def __init__(
        self,
        features: list[tuple[Tensor, Tensor]],
        nodes: list[list[Node]],
        graphs: list[Graph],
        part_ids: list[list[int]],
    ):
        self._data = list(zip(features, nodes, graphs, part_ids))

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[Tensor, Tensor], list[Node], Graph, list[int]]:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)
