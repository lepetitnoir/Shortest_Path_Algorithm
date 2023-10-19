import pytest

# Import the necessary classes and functions from the main script
from path_discovery_solution import Node, UndirectedEdge, UndirectedGraph, UndirectedPath, compute_shortest_paths

# Create test nodes and graphs for different test cases
n5, n6, n7, n8 = Node(5), Node(6), Node(7), Node(8)
demo_graph2 = UndirectedGraph(
    [
        UndirectedEdge((n5, n6), 15),
        UndirectedEdge((n6, n7), 18),
        UndirectedEdge((n7, n8), 24),
    ]
)

n9, n10, n11, n12 = Node(9), Node(10), Node(11), Node(12)
demo_graph3 = UndirectedGraph(
    [
        UndirectedEdge((n9, n10), 15),
        UndirectedEdge((n10, n11), 18),
        UndirectedEdge((n11, n12), 24),
    ]
)

n13, n14, n15, n16 = Node(13), Node(14), Node(15), Node(16)
demo_graph4 = UndirectedGraph(
    [
        UndirectedEdge((n13, n14), 15),
        UndirectedEdge((n14, n15), 18),
        UndirectedEdge((n16, n16), 24),
    ]
)

# Test case: Check a path from n5 to n8 in demo_graph2
def test_alternate_nodes():
    assert compute_shortest_paths(demo_graph2, n5, n8, 1.0) == [UndirectedPath([n5, n6, n7, n8])]

# Test case: Check if the function raises an error with start and end nodes that are identical
def test_start_equals_end():
    with pytest.raises(ValueError): 
        compute_shortest_paths(demo_graph3, n9, n9, 1.0)

# Test case: Check if the function returns an empty list when there is no path from n13 to n16 in demo_graph4
def test_no_path_start_end():
    assert compute_shortest_paths(demo_graph4, n13, n16, 1.0) == []
