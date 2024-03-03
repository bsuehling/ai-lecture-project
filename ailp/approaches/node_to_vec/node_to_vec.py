import pickle

import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk

from ailp.approaches.prediction_model import PredictionModel
from ailp.graph import Graph, Part
from ailp.utils import save_model


class NodeToVecModel(PredictionModel):
    def log(self, level: int, string: str):
        tgt_level = 0
        if level >= tgt_level:
            print(string)

    def predict_graph(self, parts: set[Part]) -> Graph:
        print(f"########### Parts {[p.part_id for p in parts]}")

        part_id_lookup = self._model["part_id_lookup"]
        family_id_lookup = self._model["family_id_lookup"]
        degree_lookup = self._model["degree_lookup"]

        graph = Graph()
        # Parts which are currently (not) in the tree
        spare_parts, used_parts = parts.copy(), []
        # Number of used connections for any given part
        connection_counts = {}
        for part in parts:
            connection_counts[part] = 0

        def lookup_part_id(id, fallback=None):
            return part_id_lookup[id] if id in part_id_lookup else fallback

        def lookup_family_id(id, fallback=None):
            return family_id_lookup[id] if id in family_id_lookup else fallback

        def degree(part):
            return (
                degree_lookup[part.part_id][2] if part.part_id in degree_lookup else 1
            )

        def max_degree(part):
            return (
                degree_lookup[part.part_id][1] if part.part_id in degree_lookup else 1
            )

        def num_connections(part):
            return connection_counts[part]

        def open_connections(part):
            return int(degree(part)) - num_connections(part)

        def max_open_connections(part):
            return max_degree(part) - num_connections(part)

        def connect(src_part, tgt_part):
            # Remove from spare parts
            if src_part in spare_parts:
                spare_parts.remove(src_part)
            if tgt_part in spare_parts:
                spare_parts.remove(tgt_part)
            # Add to used parts
            if src_part not in used_parts:
                used_parts.append(src_part)
            if tgt_part not in used_parts:
                used_parts.append(tgt_part)
            # Add the actual edge
            graph.add_undirected_edge(src_part, tgt_part)
            # Take a note on the number of connections for that part
            connection_counts[src_part] += 1
            connection_counts[tgt_part] += 1
            # Logging
            self.log(
                1,
                f"Connect {src_part.part_id} #{connection_counts[src_part]} -> "
                + f"{tgt_part.part_id} (#{connection_counts[tgt_part]})",
            )
            self.log(
                1,
                f"  Used {[p.part_id for p in used_parts]}, Spare {[p.part_id for p in spare_parts]})",
            )

        # Add one initial part to the graph
        max_degree_num = max([degree(p) for p in spare_parts])
        first_part = {p for p in spare_parts if degree(p) == max_degree_num}.pop()
        spare_parts.remove(first_part)
        used_parts.append(first_part)

        # Selects the best part which is already in use based on:
        #  - Which has the most open connection slots
        def select_source_part():
            # Find the used part with the most open connections
            max_num_open_connectors = max([open_connections(p) for p in used_parts])
            # Of there is at least one part with an open connection choose that
            if max_num_open_connectors > 0:
                return next(
                    filter(
                        lambda p: open_connections(p) == max_num_open_connectors,
                        used_parts,
                    ),
                    None,
                )
            # Otherwise go off the maximum number of degrees
            max_num_open_connectors = max([max_open_connections(p) for p in used_parts])
            return next(
                filter(
                    lambda p: max_open_connections(p) == max_num_open_connectors,
                    used_parts,
                ),
                None,
            )

        # Selects the best spare part to connect to a given used part based on:
        #  - The most likely candidate (based on training data)
        #  - If there is no likely candidate: Use the one with the most degrees
        def select_target_part(src_part: Part):
            wanted_neighbors = lookup_part_id(src_part.part_id)
            if wanted_neighbors is not None:
                wanted_neighbors = wanted_neighbors[:3]
                for tgt in wanted_neighbors:
                    if tgt in [p.part_id for p in spare_parts]:
                        return next(
                            filter(lambda p: p.part_id == tgt, spare_parts), None
                        )

            wanted_neighbors = lookup_family_id(src_part.family_id)
            if wanted_neighbors is not None:
                wanted_neighbors = wanted_neighbors
                for tgt in wanted_neighbors:
                    if tgt in [p.family_id for p in spare_parts]:
                        return next(
                            filter(lambda p: p.family_id == tgt, spare_parts), None
                        )

            return spare_parts.copy().pop()

        # Build the graph by:
        #  - Select a part from the graph
        #  - Select a partner from the spare parts
        #  - Connect those two
        while spare_parts:
            # Select the used part with most open connections
            src_part = select_source_part()
            self.log(
                1,
                f"Selected {src_part.part_id} with {open_connections(src_part)} open slots",
            )
            tgt_part = select_target_part(src_part)
            self.log(1, f"Selected {tgt_part.part_id} as the partner")
            connect(src_part, tgt_part)

        return graph

    def train(self, train_graphs: list[Graph], eval_graphs: list[Graph]):
        # train_graphs = train_graphs[:10]

        def extract_part_id(part: Part):
            return part.part_id

        def extract_family_id(part: Part):
            return part.family_id

        wtv_part_ids = self.create_word_2_vec_model(
            train_graphs, extract=extract_part_id
        )
        wtv_family_ids = self.create_word_2_vec_model(
            train_graphs, extract=extract_family_id
        )

        part_ids, family_ids = self.read_ids()

        predictor_part_ids = self.create_predictor_model(
            train_graphs, part_ids, wtv_part_ids, extract=extract_part_id
        )
        predictor_family_ids = self.create_predictor_model(
            train_graphs, family_ids, wtv_family_ids, extract=extract_family_id
        )

        part_id_lookup = self.calc_lookup_table(
            part_ids, wtv_part_ids, predictor_part_ids
        )
        family_id_lookup = self.calc_lookup_table(
            family_ids, wtv_family_ids, predictor_family_ids
        )
        degree_lookup = self.calc_degree_lookup(train_graphs)

        model = {
            "part_id_lookup": part_id_lookup,
            "family_id_lookup": family_id_lookup,
            "degree_lookup": degree_lookup,
        }
        save_model(model, self._train_path)

    # Creates a Word2Vec model
    # See: https://radimrehurek.com/gensim/models/word2vec.html
    def create_word_2_vec_model(
        self,
        train_graphs: list[Graph],
        extract=lambda part: part.part_id,
        walk_length=10,  # maximum length of a random walk
        walk_number=10,  # number of random walks per root node
        w2v_window=2,  # Maximum distance between the current and predicted word within a sentence.
        w2v_epochs=5,  # Number of iterations (epochs) over the corpus
        w2v_vec_size=64,
    ):
        walks = []
        for graph in train_graphs:
            stellar_graph = StellarGraph.from_networkx(graph.to_nx())
            new_walks = BiasedRandomWalk(stellar_graph).run(
                nodes=list(stellar_graph.nodes()),  # root nodes
                length=walk_length,
                n=walk_number,
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
            compute_loss=True,
        )
        print(f"Created Word2Vec model with error {model.get_latest_training_loss()}")

        return model

    def create_predictor_model(
        self,
        train_graphs: list[Graph],
        all_ids: list[str],
        word_to_vec,
        extract=lambda part: part.part_id,
    ):
        examples = self.calc_training_data(train_graphs, all_ids, extract=extract)

        def embedded(id):
            return word_to_vec.wv[id] if id in word_to_vec.wv else None

        # A single input is the embedding for a part
        # The expected output is the probability distribution for an edge with any given other part
        X, Y = [], []
        for index in range(len(all_ids)):
            # The embedding might be None for parts which are not in the training set
            embedding = embedded(all_ids[index])
            if embedding is None:
                continue
            X.append(embedding)
            Y.append(examples[index])

        input_shape = (len(X[0]),)
        output_shape = len(all_ids)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(output_shape, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        model.fit(
            np.array(X), np.array(Y), epochs=10, batch_size=32, validation_split=0.2
        )

        return model

    def calc_training_data(
        self,
        train_graphs: list[Graph],
        all_ids: list[str],
        extract=lambda part: part.part_id,
    ):
        # Create a matrix for any combination of ids
        dim = (len(all_ids), len(all_ids))
        tensor = np.zeros(dim)

        for graph in train_graphs:
            for src, neighbors in graph.edges.items():
                src_i = all_ids.index(extract(src.part))
                for tgt in neighbors:
                    tgt_i = all_ids.index(extract(tgt.part))
                    tensor[src_i, tgt_i] = tensor[src_i, tgt_i] + 1

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        for index in range(len(tensor)):
            tensor[index] = softmax(tensor[index])

        return tensor

    def calc_degree_lookup(
        self, train_graphs: list[Graph]
    ) -> dict[str, tuple[int, int, int]]:
        degree_dict = {}
        # Take note of the degree for any part id in all graphs
        for graph in train_graphs:
            for node, neighbors in graph.edges.items():
                key = node.part.part_id
                # Store the degree
                if key not in degree_dict:
                    degree_dict[key] = []
                degree_dict[key].append(len(neighbors))

        # Calc the min, max and avg degree for each part id
        for part_id, array in degree_dict.items():
            min_degree = min(array)
            max_degree = max(array)
            avg_degree = sum(array) / len(array)

            # Store the information
            degree_dict[part_id] = (min_degree, max_degree, avg_degree)
            # print(f"{part_id} = {(min_degree, max_degree, avg_degree)}")

        return degree_dict

    def calc_lookup_table(self, all_ids, word_to_vec, predictor):
        table = {}
        misses = 0
        for id in all_ids:
            ordered = self.predict_parts(id, word_to_vec, predictor, all_ids)
            if ordered is None:
                misses += 1
            table[id] = ordered

        print(
            f"Calculated lookup table for {len(all_ids)} Ids. Misses: {misses} ({(misses / len(all_ids)) * 100}%)"
        )
        return table

    def predict_parts(self, id, word_to_vec, predictor, id_list):
        # For this ID grab the embedding
        embedding = word_to_vec.wv[id] if id in word_to_vec.wv else None
        # The embedding can be missing if the ID wasn't encountered in the training set
        if embedding is None:
            return None
        raw = predictor.predict(np.array([embedding]))[0]
        # Interpret the output
        # Create tuples of form: (id -> score)
        tuples = [(id_list[i], raw[i]) for i in range(len(raw))]
        # Sort by score (first is best)
        tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
        # Return the sorted array of ids
        return [t[0] for t in tuples]

    def read_ids(self) -> tuple[list[str], list[str]]:
        with open("data/part_ids.dat", "rb") as f:
            part_ids = pickle.load(f)
        with open("data/family_ids.dat", "rb") as f:
            family_ids = pickle.load(f)
        return (part_ids, family_ids)
