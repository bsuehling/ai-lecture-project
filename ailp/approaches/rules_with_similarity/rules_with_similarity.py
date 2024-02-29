from os import PathLike
from ailp.approaches.prediction_model import PredictionModel
from ailp.graph import Graph, Part
from ailp.utils import save_model
from ailp.graph.node import Node
import numpy as np
import pickle
import dgl
import torch
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec

class RulesWithSimilarityModel(PredictionModel):

    def __init__(self, train_path: str | PathLike = None, test_path: str | PathLike = None) -> None:
        super().__init__(train_path, test_path)
        self._part_ids, self._family_ids = self.read_ids()

    def log(self, level: int, string: str):
        tgt_level = 0
        if level >= tgt_level: print(string)
    

    def predict_graph(self, parts: set[Part]) -> Graph:
        print(f"\n########### Parts {[p.part_id for p in parts]}")

        print(self._part_ids)

        pid_ls_matrix   = self._model["pid_ls_matrix"]
        fid_ls_matrix   = self._model["fid_ls_matrix"]
        pid_degree_dict = self._model["pid_degree_dict"]
        fid_degree_dict = self._model["fid_degree_dict"]

        graph = Graph()
        # Parts which are currently (not) in the tree
        spare_parts, used_parts = parts.copy(), []
        # Number of used connections for any given part
        connection_counts = {}
        for part in parts: connection_counts[part] = 0

        # part_id -> (min_degree, max_degree, avg_degree, [common_neighbors])
        
        degree_dict = pid_degree_dict

        def degree(part): return degree_dict[part.part_id][2] if part.part_id in degree_dict else 1
        def max_degree(part): return degree_dict[part.part_id][1] if part.part_id in degree_dict else 1
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
        
        def rate_edge(edge: tuple[Part, Part]):
            src_part, tgt_part = edge

            src_pid = self._part_ids.index(src_part.part_id)
            tgt_pid = self._part_ids.index(tgt_part.part_id)
            src_fid = self._family_ids.index(src_part.family_id)
            tgt_fid = self._family_ids.index(tgt_part.family_id)
            pid_likelihood = pid_ls_matrix[src_pid][tgt_pid]
            fid_likelihood = fid_ls_matrix[src_fid][tgt_fid]
            likelihood = pid_likelihood + fid_likelihood
            open = open_connections(src_part)
            return open + likelihood

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
    
    ########################################################################################
    ## TRAIN
    ########################################################################################

    def train(self, train_graphs: list[Graph], eval_graphs: list[Graph]):
        self.log(1, "Training like it means something")

        part_ids, family_ids = self.read_ids()
        def extract_part_id(part: Part): return part.part_id
        def extract_family_id(part: Part): return part.family_id

        if False:
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

        # node_info_dict = self.calc_node_info(train_graphs)

        # Create matrices for id x id -> likelihood of id x id
        pid_edge_likelihood = self.calc_edge_likelihood_lookup(train_graphs, part_ids,   extract=extract_part_id)
        fid_edge_likelihood = self.calc_edge_likelihood_lookup(train_graphs, family_ids, extract=extract_family_id)

        # Create Word2Vec models to calculate similarities between parts
        pid_wtv = self.create_word_2_vec_model(train_graphs, extract=extract_part_id)
        fid_wtv = self.create_word_2_vec_model(train_graphs, extract=extract_family_id)

        # Use the Word2Vec models to create matrices: id x id -> similarity between id and id with range [0,1]
        pid_sim =  self.create_similarity_matrix(part_ids, pid_wtv)
        fid_sim =  self.create_similarity_matrix(family_ids, fid_wtv)

        # Combine the likelihood and similarity matrices
        pid_ls_matrix = self.create_likelihood_similarity_matrix(part_ids, pid_edge_likelihood, pid_sim)
        fid_ls_matrix = self.create_likelihood_similarity_matrix(family_ids, fid_edge_likelihood, fid_sim)

        pid_degree_dict = self.calc_degree_lookup(train_graphs, part_ids, extract=extract_part_id)
        fid_degree_dict = self.calc_degree_lookup(train_graphs, family_ids, extract=extract_family_id)

        model = {
            "pid_ls_matrix": pid_ls_matrix,
            "fid_ls_matrix": fid_ls_matrix,
            "pid_degree_dict": pid_degree_dict,
            "fid_degree_dict": fid_degree_dict,
        }
        save_model(model, self._train_path)

        # For each part to each other part: How similar are these parts

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
    # See: https://radimrehurek.com/gensim/models/word2vec.html
    def create_word_2_vec_model(self, train_graphs: list[Graph],
                                extract=lambda part: part.part_id,
                                walk_length=10, # maximum length of a random walk
                                walk_number=10, # number of random walks per root node
                                w2v_window=3,   # Maximum distance between the current and predicted word within a sentence.
                                w2v_epochs=5,   # Number of iterations (epochs) over the corpus
                                w2v_vec_size=64
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
        print(f"Generated {len(walks)} walks based on {len(train_graphs)} graphs")

        # Convert node traces to string traces
        str_walks = [[f"{extract(n.part)}" for n in walk] for walk in walks]
        model = Word2Vec(
            sentences=str_walks,
            vector_size=w2v_vec_size,
            epochs=w2v_epochs,
            window=w2v_window, 
            min_count=1, 
            workers=4,
            compute_loss=True
        )
        print(f"Created Word2Vec model with error {model.get_latest_training_loss()}")

        return model
    
    def create_similarity_matrix(self, all_ids: list[str], wtv: Word2Vec):
        size = len(all_ids)
        matrix = np.zeros((size, size))

        misses, max_misses = 0, size*size
        for src_i in range(len(all_ids)):
            for tgt_i in range(len(all_ids)):
                src_id, tgt_id = all_ids[src_i], all_ids[tgt_i]
                if not (src_id in wtv.wv and tgt_id in wtv.wv): misses+=1 ; continue
                matrix[src_i][tgt_i] = wtv.wv.similarity(src_id, tgt_id)

        print(f"Produced ({size}x{size}) similarity matrix with {(misses/max_misses *100)}% misses")
        return matrix
    
    def create_likelihood_similarity_matrix(self, all_ids: list[str], likelihood, similarity):
        matrix = np.zeros((len(all_ids), len(all_ids)))
        # For any fixed edge A->B
        for src_i in range(len(all_ids)):
            # print(f"{src_i / len(all_ids)}%")
            for tgt_i in range(len(all_ids)):
                # For any replacement of B: B'
                likelihood_sum = 0
                for tgt_replacement_i in range(len(all_ids)):
                    # Get the similarity for B and B'
                    similarity_score = similarity[tgt_i][tgt_replacement_i]
                    # Get the likelihood of an edge A->B' 
                    likelihood_score = likelihood[src_i][tgt_replacement_i]
                    # Multiply them together: If an edge A->B' is very likely 
                    # and B-B` are very similar, this increases the likelihood
                    # for an edge A->B
                    likelihood_sum += similarity_score * likelihood_score
                matrix[src_i][tgt_i] = likelihood_sum

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        for index in range(len(matrix)): matrix[index] = softmax(matrix[index])

        print(f"Created likelihood-similarity matrix {len(all_ids)}x{len(all_ids)}")
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
