import multiprocessing
import os
import pickle
from os import PathLike

import dgl
import numpy as np
import torch
from gensim.models import Word2Vec
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk

from ailp.approaches.prediction_model import PredictionModel
from ailp.graph import Graph, Part
from ailp.graph.node import Node
from ailp.utils import save_model


# No divide degree and similarity (82.36451273062903)
# Do divide (88)
# With divide and *2: 90%
class RulesWithSimilarityModel(PredictionModel):

    def __init__(self, train_path: str | PathLike = None, test_path: str | PathLike = None) -> None:
        super().__init__(train_path, test_path)
        self._part_ids, self._family_ids = self.read_ids()

    def log(self, level: int, string: str):
        tgt_level = 0
        if level >= tgt_level: print(string)

    def predict_graph(self, parts: set[Part]) -> Graph:
        edge_score_matrix  = self._model["edge_score_matrix"]
        degree_dict: dict  = self._model["degree_dict"]

        graph = Graph()
        # Parts which are currently (not) in the tree
        spare_parts, used_parts = parts.copy(), []
        # Number of used connections for any given part
        connection_counts = {}
        for part in parts: connection_counts[part] = 0

        def degree(part): return degree_dict[part.part_id][2] if part.part_id in degree_dict else 1
        def num_connections(part): return connection_counts[part]
        def open_connections(part): return int(degree(part)) - num_connections(part)

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
            # print(f"Connect {src_part.part_id} #{connection_counts[src_part]} -> {tgt_part.part_id} (#{connection_counts[tgt_part]})")
            # print(f"  Used {[p.part_id for p in used_parts]}, Spare {[p.part_id for p in spare_parts]})")


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
        
        max_edge_score = np.max(edge_score_matrix)
        max_degree = max([d[2] for d in degree_dict.values()])
        def rate_edge(edge: tuple[Part, Part]):
            src_part, tgt_part = edge
            src_i, tgt_i = self._part_ids.index(src_part.part_id), self._part_ids.index(tgt_part.part_id)
            edge_score = edge_score_matrix[src_i][tgt_i] / max_edge_score
            open = open_connections(src_part) / max_degree
            return edge_score + open * 2

        # Iteratively build the graph
        while spare_parts:
            edges = generate_possible_edges()
            rated = [ (e, rate_edge(e)) for e in edges]
            best_rating = max([ re[1] for re in rated ])
            chosen_edge = next(filter(lambda re: re[1] == best_rating, rated), None)[0]
            connect(chosen_edge[0], chosen_edge[1])

        return graph
    
    ########################################################################################
    ## TRAIN
    ########################################################################################

    def train(self, train_graphs: list[Graph], eval_graphs: list[Graph]):
        part_ids, family_ids = self.read_ids()
        def extract_part_id(part: Part): return part.part_id
        def extract_family_id(part: Part): return part.family_id

        # We don't use the evaluation graphs for hyperparameter tuning anyways
        train_graphs = train_graphs + eval_graphs

        # Trim the data if wanted to speed up development
        if os.environ.get("TRIM_DATA", "no") == "yes":
            train_graphs = train_graphs[:10]
            part_ids = set()
            for graph in train_graphs:
                for node in graph.nodes:
                    part_ids.add(int(node.part.part_id))
            part_ids = list(part_ids)
            part_ids.sort()
            part_ids = [str(p) for p in part_ids]
            family_ids = set()
            for graph in train_graphs:
                for node in graph.nodes:
                    family_ids.add(int(node.part.family_id))
            family_ids = list(family_ids)
            family_ids.sort()
            family_ids = [str(p) for p in family_ids]
        
        # Create a matrix for family-id -> number of different parts in that family
        family_num_parts_lookup = self.calc_num_parts_per_family_lookup(family_ids, train_graphs)
        part_to_family_lookup   = self.calc_family_for_part_lookup(part_ids, family_ids, train_graphs)

        # Calculate the expected degrees for each part
        pid_degree_dict = self.calc_degree_lookup(train_graphs, part_ids, extract=extract_part_id)
        fid_degree_dict = self.calc_degree_lookup(train_graphs, family_ids, extract=extract_family_id)
        # print(pid_degree_dict)
        degree_dict: dict = {}
        for pid, pid_degrees in pid_degree_dict.items():
            f_index = part_to_family_lookup[part_ids.index(pid)]
            fid = family_ids[f_index]
            fid_degrees = fid_degree_dict[fid]
            degree_dict[pid] = ((pid_degrees[0] + fid_degrees[0] /2), (pid_degrees[1] + fid_degrees[1] /2), (pid_degrees[2] + fid_degrees[2] /2))

        # Create matrices for id x id -> likelihood of id x id
        pid_edge_likelihood = self.calc_edge_likelihood_lookup(train_graphs, part_ids,   extract=extract_part_id)
        fid_edge_likelihood = self.calc_edge_likelihood_lookup(train_graphs, family_ids, extract=extract_family_id)

        # Create Word2Vec models to calculate similarities between parts
        pid_wtv = self.create_word_2_vec_model(train_graphs, extract=extract_part_id)
        fid_wtv = self.create_word_2_vec_model(train_graphs, extract=extract_family_id)

        # Use the Word2Vec models to create matrices: id x id -> similarity between id and id with range [0,1]
        pid_sim =  self.create_similarity_matrix(part_ids, pid_wtv)
        fid_sim =  self.create_similarity_matrix(family_ids, fid_wtv)

        # Combine all of the matrices
        edge_score_matrix = self.create_edge_score_matrix(
            part_ids, 
            family_ids, 
            part_to_family_lookup, 
            family_num_parts_lookup, 
            pid_sim, 
            fid_sim, 
            pid_edge_likelihood, 
            fid_edge_likelihood
        )

        model = {
            "edge_score_matrix": edge_score_matrix,
            "degree_dict": degree_dict
        }
        save_model(model, self._train_path)
        print(f"Training completed")

    # Calculates a matrix for each possible edge between
    def calc_edge_likelihood_lookup(self, train_graphs: list[Graph], all_ids: list[str], extract=None):
        # Create matrices for all possible edges
        dim = (len(all_ids), len(all_ids))

        # Count the raw number of edges
        edge_count_matrix = np.zeros(dim)
        for graph in train_graphs:
            for src, neighbors in graph.edges.items():
                src_i = all_ids.index(extract(src.part))
                for tgt in neighbors:
                     tgt_i = all_ids.index(extract(tgt.part))
                     edge_count_matrix[src_i, tgt_i] = edge_count_matrix[src_i, tgt_i] + 1

        # Count the number of times edges are possible to build
        possibility_count_matrix = np.zeros(dim)
        for graph in train_graphs:
            for src in graph.nodes:
                for tgt in graph.nodes:
                    if src == tgt: continue
                    src_i = all_ids.index(extract(src.part))
                    tgt_i = all_ids.index(extract(tgt.part))
                    possibility_count_matrix[src_i, tgt_i] = possibility_count_matrix[src_i, tgt_i] + 1

        element_wise_division = np.vectorize(lambda x, y: x / y if y > 0 else 0)
        edge_likelihood = element_wise_division(edge_count_matrix, possibility_count_matrix)

        return edge_likelihood

    # Creates a Word2Vec model
    def create_word_2_vec_model(self, train_graphs: list[Graph],
                                extract=lambda part: part.part_id,
                                walk_length=10, # maximum length of a random walk
                                walk_number=25, # number of random walks per root node
                                w2v_window=2,   # Maximum distance between the current and predicted word within a sentence.
                                                # We choose a small value as we are only interested in the direct neighbors for our small graphs
                                w2v_vec_size=300
                                ):
        walks = []
        for graph in train_graphs:
            stellar_graph = StellarGraph.from_networkx(graph.to_nx())
            new_walks = BiasedRandomWalk(stellar_graph).run(
                nodes=list(stellar_graph.nodes()),  # root nodes
                length=walk_length, n=walk_number,  
                p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
                q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
            )
            walks.extend(new_walks)

        # Convert node traces to string traces
        str_walks = [[f"{extract(n.part)}" for n in walk] for walk in walks]
        cores = multiprocessing.cpu_count()
        print(f"Training Word2Vec on {len(str_walks)} using sentences {cores} cores...")
        model = Word2Vec(
            vector_size=w2v_vec_size,
            window=w2v_window, 
            min_count=1, 
            workers=cores,
        )
        model.build_vocab(str_walks)

        for epoch in range(5):
            model.train(str_walks, total_examples=model.corpus_count, epochs=5, report_delay=1,  compute_loss=True)
            print(f"Current loss {model.get_latest_training_loss()}")

        print(f"Created Word2Vec model with error {model.get_latest_training_loss()}")

        return model
    
    def calc_num_parts_per_family_lookup(self, family_ids: list[str], train_graphs: list[Graph]):
        data_dict: dict[str, list[str]] = {}
        for graph in train_graphs:
            for node in graph.nodes:
                pid, fid = node.part.part_id, node.part.family_id
                if not fid in data_dict: data_dict[fid] = []
                if not pid in data_dict[fid]: data_dict[fid].append(pid)

        matrix = np.ones(len(family_ids))

        for index in range(len(family_ids)):
            fid = family_ids[index]
            matrix[index] = len(data_dict[fid]) if fid in data_dict else 0

        print(f"Calculated num parts for family lookup (values between {min(matrix)} and {max(matrix)})")
        return matrix

    def calc_family_for_part_lookup(self, part_ids: list[str], family_ids: list[str], train_graphs: list[Graph]):
        matrix = np.full(len(part_ids), -1)
        for graph in train_graphs:
            for node in graph.nodes:
                pid, fid = node.part.part_id, node.part.family_id
                pid_index = part_ids.index(pid)
                fid_index = family_ids.index(fid)
                matrix[pid_index] = fid_index
        
        print(f"Calculated part-id-index -> family-id-index lookup with {np.sum(matrix == -1)} misses")
        return matrix

    def create_similarity_matrix(self, all_ids: list[str], wtv: Word2Vec):
        size = len(all_ids)
        matrix = np.zeros((size, size))

        misses, max_misses = 0, size*size
        for src_i in range(len(all_ids)):
            for tgt_i in range(len(all_ids)):
                src_id, tgt_id = all_ids[src_i], all_ids[tgt_i]
                if not (src_id in wtv.wv and tgt_id in wtv.wv): misses+=1 ; continue
                matrix[src_i][tgt_i] = abs(wtv.wv.similarity(src_id, tgt_id))

        print(f"Produced ({size}x{size}) similarity matrix with {(misses/max_misses *100)}% misses")
        return matrix
    
    def create_edge_score_matrix(self, part_ids, family_ids, part_to_family, num_parts_in_fam, pid_similarity, fid_similarity, pid_likelihood, fid_likelihood):
        matrix = np.zeros((len(part_ids), len(part_ids)))
        # For any fixed edge A->B
        for src_pi in range(len(part_ids)):
            print(f"Generating edge score matrix: {int(src_pi / len(part_ids) * 100)}%")
            src_fi = part_to_family[src_pi]
            for tgt_pi in range(len(part_ids)):
                # Resolve the family index for the part index
                tgt_fi = part_to_family[tgt_pi]

                # For any replacement of B: B'
                likelihood_sum = 0
                for tgt_replacement_pi in range(len(part_ids)):
                    # Resolve the family index for the part index
                    tgt_replacement_fi = part_to_family[tgt_replacement_pi]

                    # Check if we know the family for all concerned part ids
                    families_are_known = src_fi != -1 and tgt_fi != -1 and tgt_replacement_fi != -1

                    # Get the similarity between B and B'
                    part_similarity  = pid_similarity[tgt_pi][tgt_replacement_pi]
                    # Get the likelihood of an edge A->B' 
                    part_likelihood  = pid_likelihood[src_pi][tgt_replacement_pi]
                    # Multiply them together: If an edge A->B' is very likely 
                    # and B-B` are very similar, this increases the likelihood
                    # for an edge A->B
                    part_score = part_similarity * part_likelihood

                    # Repeat for the families if we know them
                    if families_are_known:
                        family_similarity = fid_similarity[tgt_fi][tgt_replacement_fi]
                        family_likelihood = fid_likelihood[src_fi][tgt_replacement_fi]
                        family_score = family_similarity * family_likelihood
                        # Combine the scores based on part and family id
                        # Families, which have a lot of different parts in them are not valued as much
                        # as families which only have a few parts, as this contains more information
                        num_parts_in_family = num_parts_in_fam[tgt_replacement_fi]
                        score = part_score + (family_score / num_parts_in_family)
                    else:
                        # If the family isn't known fall back to just using the part score
                        score = part_score

                    likelihood_sum += score
                matrix[src_pi][tgt_pi] = likelihood_sum

        # def softmax(x):
        #     e_x = np.exp(x - np.max(x))
        #     return e_x / e_x.sum(axis=0)
        # for index in range(len(matrix)): matrix[index] = softmax(matrix[index])

        print(f"Created edge score matrix {len(matrix)}x{len(matrix[0])} in range [{np.min(matrix)}, {np.max(matrix)}]")
        return matrix
    
    def calc_degree_lookup(self, train_graphs: list[Graph], all_ids, extract=None):
        degree_dict = {}
        # Take note of the degree for any part id in all graphs
        for graph in train_graphs:
            for node, neighbors in graph.edges.items():
                key = extract(node.part)
                # Store the degree 
                if not key in degree_dict: degree_dict[key] = []
                degree_dict[key].append(len(neighbors))

        # Calc the min, max and avg degree for each part id
        for id, array in degree_dict.items():
            min_degree = min(array)
            max_degree = max(array)
            avg_degree = sum(array) / len(array) 

            # Store the information
            degree_dict[id] = (min_degree, max_degree, avg_degree)
            # print(f"{part_id} = {(min_degree, max_degree, avg_degree)}")

        return degree_dict

    ########################################################################################
    ## UTILITY
    ########################################################################################

    def read_ids(self) -> tuple[list[str], list[str]]:
        with open("data/part_ids.dat", "rb") as f:
            part_ids = pickle.load(f)
        with open("data/family_ids.dat", "rb") as f:
            family_ids = pickle.load(f)
        return (part_ids, family_ids)
