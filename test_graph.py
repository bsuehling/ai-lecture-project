import unittest
import numpy as np

from graph import Graph
from node import Node
from part import Part


class TestGraph(unittest.TestCase):

    # -------------------- add_undirected_edge(part1, part2) --------------------
    def test_add_undirected_edge(self):
        graph = Graph()
        part1 = Part(1, 3)
        part2 = Part(2, 3)
        graph.add_undirected_edge(part1, part2)

        self.assertEqual(len(graph.get_nodes()), 2, 'Graph should contain two nodes')
        self.assertEqual(graph.get_nodes(), {Node(0, part1), Node(1, part2)},
                         'Graph should contain source and sink node.')

    def test_add_undirected_edge_duplicate_undirected_edge_ignored(self):
        part1 = Part(1, 11)
        part2 = Part(2, 12)
        part3 = Part(3, 13)

        graph = Graph()  # 1 = 3 - 2
        graph.add_undirected_edge(part1, part3)
        graph.add_undirected_edge(part2, part3)
        graph.add_undirected_edge(part1, part3)  # duplicate undirected edge

        graph_without_duplicated_edge = Graph()  # 1 - 3 - 2
        graph_without_duplicated_edge.add_undirected_edge(part1, part3)
        graph_without_duplicated_edge.add_undirected_edge(part2, part3)

        self.assertEqual(graph_without_duplicated_edge, graph, 'Graph should ignore duplicate reversed edges.')

    def test_add_undirected_edge_duplicate_reversed_undirected_edge_ignored(self):
        part1 = Part(1, 11)
        part2 = Part(2, 12)
        part3 = Part(3, 13)

        graph = Graph()  # 1 = 3 - 2
        graph.add_undirected_edge(part1, part3)
        graph.add_undirected_edge(part2, part3)
        graph.add_undirected_edge(part3, part1)   # duplicate *reversed* undirected edge

        graph_without_duplicated_edge = Graph()  # 1 - 3 - 2
        graph_without_duplicated_edge.add_undirected_edge(part1, part3)
        graph_without_duplicated_edge.add_undirected_edge(part2, part3)

        self.assertEqual(graph_without_duplicated_edge, graph, 'Graph should ignore duplicate reversed edges.')

    def test_add_undirected_edge_self_loops_of_nodes_ignored(self):
        part1 = Part(1, 11)
        part2 = Part(2, 12)

        graph = Graph()  # 1 - 2°
        graph.add_undirected_edge(part1, part2)
        graph.add_undirected_edge(part2, part2)  # self-loop

        graph_without_self_loop = Graph()  # 1 - 2
        graph_without_self_loop.add_undirected_edge(part1, part2)

        self.assertEqual(graph_without_self_loop, graph, 'Graph should ignore self-loops of nodes.')

    # -------------------- __eq__(other) --------------------

    def test_equality_returns_true(self):
        # Arrange
        # Two graphs with equal nodes but different order for adding edges
        g1 = Graph()
        part1 = Part(1, 11)
        part2 = Part(2, 11)
        part3 = Part(3, 13)
        part4 = Part(4, 14)
        g1.add_undirected_edge(part1, part2)
        g1.add_undirected_edge(part2, part3)
        g1.add_undirected_edge(part2, part4)

        g2 = Graph()
        part5 = Part(1, 11)
        part6 = Part(2, 11)
        part7 = Part(3, 13)
        part8 = Part(4, 14)
        # different order for adding the edges
        g2.add_undirected_edge(part6, part8)
        g2.add_undirected_edge(part5, part6)
        g2.add_undirected_edge(part6, part7)

        # Assert
        self.assertEqual(g1, g2, 'Two different graph instances with equal nodes and edges should be equal.')

    def test_equality_returns_false_for_different_nodes(self):
        # Arrange
        g1 = Graph()
        part1 = Part(1, 11)
        part2 = Part(2, 11)
        part3 = Part(3, 13)
        # part4 (as equivalent to part8 in graph2) is missing
        g1.add_undirected_edge(part1, part2)
        g1.add_undirected_edge(part2, part3)

        g2 = Graph()
        part5 = Part(1, 11)
        part6 = Part(2, 11)
        part7 = Part(3, 13)
        part8 = Part(1, 11)  # equivalent to part 5
        # different order for adding the edges
        g2.add_undirected_edge(part6, part8)
        g2.add_undirected_edge(part5, part6)
        g2.add_undirected_edge(part6, part7)

        # Assert
        self.assertNotEqual(g1, g2,
                            'Two different graph instances with different number of nodes should not be equal.')

    def test_equality_returns_false_for_different_edges(self):
        # Arrange
        g1 = Graph()
        part1 = Part(1, 11)
        part2 = Part(2, 11)
        part3 = Part(3, 13)
        part4 = Part(4, 14)
        g1.add_undirected_edge(part1, part2)
        g1.add_undirected_edge(part2, part3)
        g1.add_undirected_edge(part2, part4)

        g2 = Graph()
        part5 = Part(1, 11)
        part6 = Part(2, 11)
        part7 = Part(3, 13)
        part8 = Part(1, 14)
        g2.add_undirected_edge(part6, part8)
        g2.add_undirected_edge(part5, part6)
        g2.add_undirected_edge(part6, part7)
        g2.add_undirected_edge(part7, part8)   # additional edge in graph2

        # Assert equality but not identity
        self.assertNotEqual(g1, g2, 'Two different graph instances with different edges should not be equal.')

    # --------------------  --------------------

    def test_equivalent_parts_treated_as_different_nodes(self):
        part1 = Part(1, 11)
        part2 = Part(2, 12)
        part3_equivalent_to_part1 = Part(1, 11)
        g = Graph()
        g.add_undirected_edge(part1, part2)
        g.add_undirected_edge(part2, part3_equivalent_to_part1)

        self.assertEqual(3, len(g.get_nodes()), 'Graph should contain three nodes.')

    # -------------------- is_cyclic() --------------------

    def test_is_cyclic_raises_exception_for_empty_graph(self):
        graph = Graph()
        self.assertRaises(BaseException, graph.is_cyclic)

    def test_graph_with_trivial_cycle_is_noncyclic(self):
        part1 = Part(1, 11)
        part2 = Part(2, 12)

        graph = Graph()  # 1 - 2
        graph.add_undirected_edge(part1, part2)

        self.assertFalse(graph.is_cyclic(), 'Graph with trivial cycle (bidirectional edge) should not be cyclic.')

    def test_is_cyclic(self):
        part1 = Part(1, 11)
        part2 = Part(2, 12)
        part3 = Part(3, 13)

        graph = Graph()  # cyclic graph
        graph.add_undirected_edge(part1, part2)
        graph.add_undirected_edge(part2, part3)
        graph.add_undirected_edge(part3, part1)

        self.assertTrue(graph.is_cyclic(), 'Cyclic graph should be cyclic.')

    def test_is_cyclic_returns_false_two_equivalent_parts(self):
        part1 = Part(1, 11)
        part2 = Part(2, 12)
        part3 = Part(3, 13)
        part4 = Part(1, 11)  # same as part1

        graph = Graph()  # 1 - 2 - 3 - 4
        graph.add_undirected_edge(part1, part2)
        graph.add_undirected_edge(part2, part3)
        graph.add_undirected_edge(part3, part4)

        # check if part1 and part4 are distinguished
        self.assertFalse(graph.is_cyclic(), 'Graph should not be cyclic.')

    # -------------------- is_connected() --------------------

    def test_is_connected_raises_exception_for_empty_graph(self):
        graph = Graph()
        self.assertRaises(BaseException, graph.is_connected)

    def test_is_connected_returns_true_for_2node_graph(self):
        part1 = Part(1, 11)
        part2 = Part(2, 12)

        graph = Graph()  # 1 - 2
        graph.add_undirected_edge(part1, part2)

        self.assertTrue(graph.is_connected(), 'Graph composed of one bidirectional edge should be connected')

    def test_is_connected_returns_true_for_connected_graph(self):
        part1 = Part(1, 11)
        part2 = Part(2, 12)
        part3 = Part(3, 13)
        part4 = Part(4, 14)

        graph = Graph()
        graph.add_undirected_edge(part1, part2)
        graph.add_undirected_edge(part2, part3)
        graph.add_undirected_edge(part2, part4)

        self.assertTrue(graph.is_connected(), 'Connected graph should be connected')

    def test_is_connected_returns_false_for_separated_graph(self):
        part1 = Part(1, 11)
        part2 = Part(2, 12)
        part3 = Part(3, 13)
        part4 = Part(4, 14)

        graph = Graph()  # 1 - 2    3 - 4
        graph.add_undirected_edge(part1, part2)
        graph.add_undirected_edge(part3, part4)

        self.assertFalse(graph.is_connected(), 'Two unconnected graphs with each 2 nodes should not be connected.')

    # -------------------- get_adjacency_matrix(part_order) --------------------

    def test_get_adjacency_matrix(self):
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
        graph = Graph()
        graph.add_undirected_edge(part_d, part_e)
        graph.add_undirected_edge(part_b, part_d)
        graph.add_undirected_edge(part_c, part_d)
        graph.add_undirected_edge(part_a, part_c)
        graph.add_undirected_edge(part_a, part_b)

        part_order = (part_a, part_b, part_c, part_d, part_e)

        expected_adj_matrix = np.array([[0, 1, 1, 0, 0],
                                        [1, 0, 0, 1, 0],
                                        [1, 0, 0, 1, 0],
                                        [0, 1, 1, 0, 1],
                                        [0, 0, 0, 1, 0]], dtype=int)

        computed_adj_matrix = graph.get_adjacency_matrix(part_order)
        self.assertTrue(np.all(expected_adj_matrix == computed_adj_matrix), 'Adjacency matrices should be equal.')
