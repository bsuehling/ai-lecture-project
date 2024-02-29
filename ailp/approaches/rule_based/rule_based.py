import dgl
import torch

from ailp.approaches.prediction_model import PredictionModel
from ailp.graph import Graph, Part
from ailp.graph.node import Node
from ailp.utils import save_model

# Scores:
#  92.3 (Always select part with highest open connectors as src)
#  97.8 (Rank and compare all possible edges from parts in tree and spare parts)
class RuleBasedModel(PredictionModel):

    def log(self, level: int, string: str):
        tgt_level = 0
        if level >= tgt_level: print(string)
    

    def predict_graph(self, parts: set[Part]) -> Graph:
        self.log(1, f"\n########### Parts {[p.part_id for p in parts]}")

        graph = Graph()
        # Parts which are currently (not) in the tree
        spare_parts, used_parts = parts.copy(), []
        # Number of used connections for any given part
        connection_counts = {}
        for part in parts: connection_counts[part] = 0

        # part_id -> (min_degree, max_degree, avg_degree, [common_neighbors])
        node_info_dict: dict[str, tuple[int, int, int, list[str]]] = self._model
        def degree(part): return node_info_dict[part.part_id][2] if part.part_id in node_info_dict else 1
        def max_degree(part): return node_info_dict[part.part_id][1] if part.part_id in node_info_dict else 1
        def neighbors(part): return node_info_dict[part.part_id][3] if part.part_id in node_info_dict else []
        def num_connections(part): return connection_counts[part]
        def open_connections(part): return int(degree(part)) - num_connections(part)
        def max_open_connections(part): return max_degree(part) - num_connections(part)
        
        def connect(src_part, tgt_part):
            # Remove from spare parts
            if src_part in spare_parts: spare_parts.remove(src_part)
            if tgt_part in spare_parts: spare_parts.remove(tgt_part)
            # Add to used parts
            if not src_part in used_parts: used_parts.append(src_part)
            if not tgt_part in used_parts: used_parts.append(tgt_part)
            # Add the actual edge
            graph.add_undirected_edge(src_part, tgt_part)
            # Take a note on the number of connections for that part
            connection_counts[src_part] += 1
            connection_counts[tgt_part] += 1
            # Logging
            self.log(1, f"Connect {src_part.part_id} #{connection_counts[src_part]} -> {tgt_part.part_id} (#{connection_counts[tgt_part]})")
            self.log(1, f"  Used {[p.part_id for p in used_parts]}, Spare {[p.part_id for p in spare_parts]})")


        # Add one initial part to the graph
        max_degree_num = max([degree(p) for p in spare_parts])
        first_part =  {p for p in spare_parts if degree(p) == max_degree_num}.pop()
        spare_parts.remove(first_part)
        used_parts.append(first_part)

        def generate_possible_edges():
            edges = []
            for src in used_parts:
                for tgt in spare_parts:
                    edges.append((src, tgt))
            return edges
        
        def rate_edge(edge: tuple[Part, Part]):
            src_part, tgt_part = edge
            wanted_neighbors = neighbors(src_part)

            def calc_wanted_score():
                if not tgt_part.part_id in wanted_neighbors: return -3
                if tgt_part.part_id == wanted_neighbors[0]: return 5
                if tgt_part.part_id in wanted_neighbors[:3]: return 3
                return 1

            wanted_score = calc_wanted_score()
            degree_score = open_connections(src_part)
            # print(f"SRC {src_part.part_id} wants: {wanted_neighbors}")
            # print(f"Edge {src_part.part_id}->{tgt_part.part_id} : wanted={wanted_score} degree={degree_score}")
            return wanted_score + degree_score

        # Build the graph by:
        #  - Select a part from the graph
        #  - Select a partner from the spare parts
        #  - Connect those two
        while spare_parts:
            edges = generate_possible_edges()
            rated = [ (e, rate_edge(e)) for e in edges]
            best_rating = max([ re[1] for re in rated ])
            print(best_rating)
            chosen_edge = next(filter(lambda re: re[1] == best_rating, rated), None)[0]
            connect(chosen_edge[0], chosen_edge[1])

        return graph

    def train(self, train_graphs: list[Graph], eval_graphs: list[Graph]):
        node_info_dict = self.calc_node_info(train_graphs)
        save_model(node_info_dict, self._train_path)

    def calc_node_info(self, train_graphs: list[Graph]) -> dict[str, tuple[int, int, int, list[str]]]:
        node_info_dict, node_num_neighbors_dict = {}, {}
        # Take note of the degree for any part id in all graphs
        for graph in train_graphs:
            for node, neighbors in graph.edges.items():
                key = node.part.part_id
                # Store the degree 
                if not key in node_info_dict: node_info_dict[key] = []
                node_info_dict[key].append(len(neighbors))
                # For each possible target count the number of connections
                if not key in node_num_neighbors_dict: node_num_neighbors_dict[key] = {}
                for tgt in neighbors:
                    tgt_key = tgt.part.part_id
                    if not tgt_key in node_num_neighbors_dict[key]: node_num_neighbors_dict[key][tgt_key] = 0
                    node_num_neighbors_dict[key][tgt_key] += 1

        # Calc the min, max and avg degree for each part id
        for part_id, array in node_info_dict.items():
            min_degree = min(array)
            max_degree = max(array)
            avg_degree = sum(array) / len(array) 

            # Generate a list of favorite parts connected to
            # this part based on the count. First part is most likely
            tuples = []
            for tgt_part_id, count in node_num_neighbors_dict[part_id].items():
                tuples.append((tgt_part_id, count))
            sorted_tuples = sorted(tuples, key=lambda tup: tup[1], reverse=True)
            fav_parts = [t[0] for t in sorted_tuples]

            # Store the information
            node_info_dict[part_id] = (min_degree, max_degree, avg_degree, fav_parts)
            self.log(1, f"{part_id} = {(min_degree, max_degree, avg_degree, fav_parts)}")

        return node_info_dict
    
    # def calc_edge_info(self, train_graphs: list[Graph]):
    #     # Calculate the set of all edges
    #     edges = set([])
    #     for graph in train_graphs:
    #         edges = edges.union(self.edges_for_graph(graph))
    #     self.log(1, f"There are {len(edges)} different edges")


    #     edge_info_dict: dict[(str, str), (int, int)] = {}
    #     for edge in edges:
    #         possible, found = 0, 0
    #         for graph in train_graphs:
    #             # Check whether it's possible to build this edge
    #             part_ids = [n.part.part_id for n in graph.nodes]
    #             src, tgt = edge
    #             if src in part_ids and tgt in part_ids:
    #                 possible += 1
    #             else: continue

    #             # Check whether this edge is actually found
    #             local_edges = self.edges_for_graph(graph)
    #             if edge in local_edges: found += 1

    #         score = found / possible
    #         edge_info_dict[edge] = (possible, found, score)

    #     return edge_info_dict


    # def edges_for_graph(self, graph: Graph):
    #     edges = set([])
    #     for src, neighbors in graph.edges.items():
    #         for tgt in neighbors:
    #             edges.add(frozenset([src.part.part_id, tgt.part.part_id]))
    #     return edges

    # def find_types(self, graphs: list[Graph]):
    #     types = []
    #     for graph in graphs:
    #         for node in graph.nodes:
    #             type = int(node.part.part_id)
    #             if not type in types:
    #                 types.append(type)
    #     types.sort()
    #     return types

    # def parts_to_vec(self, parts: set[Part]):
    
    