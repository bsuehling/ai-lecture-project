import unittest

from part import Part
from graph import Graph
from evaluation import edge_accuracy


class TestEvaluation(unittest.TestCase):

    def test_edge_accuracy_graphs_equivalent(self):
        part_a1 = Part(1, 2)
        part_a2 = Part(1, 2)
        part_b = Part(3, 4)
        part_c = Part(5, 6)
        part_d1 = Part(7, 8)
        part_d2 = Part(7, 8)

        """         Target Graph
              B 
            /   \
         A1       D1  -  A2  -  D2  
            \   /
              C
        """
        target_graph = Graph()
        target_graph.add_undirected_edge(part_a1, part_b)
        target_graph.add_undirected_edge(part_a1, part_c)
        target_graph.add_undirected_edge(part_b, part_d1)
        target_graph.add_undirected_edge(part_c, part_d1)
        target_graph.add_undirected_edge(part_d1, part_a2)
        target_graph.add_undirected_edge(part_a2, part_d2)

        """         Predicted Graph: A1 and A2 as well as D1 and D2 are interchanged
              B 
            /   \
         A2       D1  -  A1  - D2
            \   /
              C
        """
        predicted_graph = Graph()
        predicted_graph.add_undirected_edge(part_a2, part_b)
        predicted_graph.add_undirected_edge(part_a2, part_c)
        predicted_graph.add_undirected_edge(part_b, part_d1)
        predicted_graph.add_undirected_edge(part_c, part_d1)
        predicted_graph.add_undirected_edge(part_d1, part_a1)
        predicted_graph.add_undirected_edge(part_a1, part_d2)

        expected_edge_accuracy = pow(len(target_graph.get_nodes()), 2)   # all edges should be predicted correctly
        computed_edge_accuracy = edge_accuracy(predicted_graph, target_graph)

        self.assertEqual(expected_edge_accuracy, computed_edge_accuracy, 'Wrong value for edge accuracy.')

    def test_edge_accuracy_graphs_equivalent_all_parts_unique(self):
        part_a = Part(1, 2)
        part_b = Part(3, 4)
        part_c = Part(5, 6)
        part_d = Part(7, 8)
        part_e = Part(9, 10)

        """         Target Graph
              B 
            /   \
          A       D - E
            \   /
              C
        """
        target_graph = Graph()
        target_graph.add_undirected_edge(part_a, part_b)
        target_graph.add_undirected_edge(part_a, part_c)
        target_graph.add_undirected_edge(part_b, part_d)
        target_graph.add_undirected_edge(part_c, part_d)
        target_graph.add_undirected_edge(part_d, part_e)

        """         Predicted Graph: same as target graph
              B 
            /   \
          A       D  -  E
            \   /
              C
        """
        predicted_graph = Graph()
        predicted_graph.add_undirected_edge(part_a, part_b)
        predicted_graph.add_undirected_edge(part_a, part_c)
        predicted_graph.add_undirected_edge(part_b, part_d)
        predicted_graph.add_undirected_edge(part_c, part_d)
        predicted_graph.add_undirected_edge(part_d, part_e)

        expected_edge_accuracy = pow(len(target_graph.get_nodes()), 2)   # all edges should be predicted correctly
        computed_edge_accuracy = edge_accuracy(predicted_graph, target_graph)

        self.assertEqual(expected_edge_accuracy, computed_edge_accuracy, 'Wrong value for edge accuracy.')

    def test_edge_accuracy_all_parts_unique_wrong_prediction(self):
        part_a = Part(1, 2)
        part_b = Part(3, 4)
        part_c = Part(5, 6)
        part_d = Part(7, 8)
        part_e = Part(9, 10)

        """         Target Graph
              B 
            /   \
          A       D - E
            \   /
              C
        """
        target_graph = Graph()
        target_graph.add_undirected_edge(part_a, part_b)
        target_graph.add_undirected_edge(part_a, part_c)
        target_graph.add_undirected_edge(part_b, part_d)
        target_graph.add_undirected_edge(part_c, part_d)
        target_graph.add_undirected_edge(part_d, part_e)

        """         Predicted Graph: (B, D) is missing and new edge (B, C)            
              B 
            /   
          A   |   D - E
            \   /
              C
        """
        predicted_graph = Graph()
        predicted_graph.add_undirected_edge(part_a, part_b)
        predicted_graph.add_undirected_edge(part_a, part_c)
        predicted_graph.add_undirected_edge(part_b, part_c)
        predicted_graph.add_undirected_edge(part_c, part_d)
        predicted_graph.add_undirected_edge(part_d, part_e)

        expected_edge_accuracy = pow(len(target_graph.get_nodes()), 2) - 4  # two bidirectional edges are wrong
        computed_edge_accuracy = edge_accuracy(predicted_graph, target_graph)

        self.assertEqual(expected_edge_accuracy, computed_edge_accuracy, 'Wrong value for edge accuracy.')