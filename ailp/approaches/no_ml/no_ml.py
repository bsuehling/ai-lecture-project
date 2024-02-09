from ailp.approaches.prediction_model import PredictionModel
from ailp.graph import Graph, Part


class NoMLModel(PredictionModel):
    def predict_graph(self, parts: set[Part]) -> Graph:
        pass

    def train(self, train_graphs: list[Graph], eval_graphs: list[Graph]):
        connections: dict[str, dict[str, int]] = {}
        for graph in train_graphs:
            for node, neighbors in graph.edges.items():
                part_id = node.part.part_id
                if part_id not in connections:
                    connections[part_id] = {}
                for neighbor in neighbors:
                    neighbor_part_id = neighbor.part.part_id
                    if neighbor_part_id not in connections[part_id]:
                        connections[part_id][neighbor_part_id] = 0
                    connections[part_id][neighbor_part_id] += 1
