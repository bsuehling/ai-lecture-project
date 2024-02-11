from ailp.approaches.prediction_model import PredictionModel
from ailp.graph import Graph, Part
from ailp.utils import save_model


class NoMLModel(PredictionModel):
    def predict_graph(self, parts: set[Part]) -> Graph:
        model: dict[str, dict[str, int]] = self._model
        parts_ = parts.copy()
        graph = Graph()

        # Find the parts which to connect first
        max_conns = None, None, -1
        for part in parts_:
            for other in parts_ - set([part]):
                count = model.get(part.part_id, {}).get(other.part_id, None)
                if count and count > max_conns[2]:
                    max_conns = part, other, count
        if max_conns[0] and max_conns[1]:
            parts_.remove(max_conns[0])
            parts_.remove(max_conns[1])
            graph.add_undirected_edge(max_conns[0], max_conns[1])
        else:
            graph.add_undirected_edge(parts_.pop(), parts_.pop())

        # Iteratively add one unconnected part to the graph
        while parts_:
            max_conns = None, None, -1
            for part in graph.parts:
                for other in parts_:
                    count = model.get(part.part_id, {}).get(other.part_id, None)
                    if count and count > max_conns[2]:
                        max_conns = part, other, count
            if max_conns[1] is not None:
                parts_.remove(max_conns[1])
                graph.add_undirected_edge(max_conns[0], max_conns[1])
            else:
                graph.add_undirected_edge(list(graph.parts)[0], parts_.pop())

        return graph

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
        save_model(connections, self._train_path)
