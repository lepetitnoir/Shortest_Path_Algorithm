"""
Algorithm Description: Shortest Path Discovery in an Undirected Weighted Graph

This algorithm is designed to find the N shortest paths from a starting node to an ending node within an undirected weighted graph. The graph consists of nodes and edges, where edges have strictly positive lengths. The primary objective of this algorithm is to identify the shortest path from the start node to the end node, and it can also find subsequent shorter paths based on a given tolerance factor.

Input:

    graph: The undirected graph in which the shortest paths are to be found.
    start: The starting node for the paths.
    end: The ending node for the paths.
    length_tolerance_factor: The maximum length ratio allowed for the discovered paths. This factor specifies how much longer the paths can be compared to the shortest path (minimum value: 1.0).

Output:

    A list of discovered paths. If no path from the starting node to the ending node exists, the result is an empty list.

Algorithm Details:

    1. Initialize an empty list goal_paths to store the discovered paths.

    2. Initialize shortest_path_length to positive infinity, as we have not found any paths yet.

    3. Create a priority queue candidate_paths for storing paths. Put the initial path with the starting node into the queue.

    4. Set a maximum number of iterations to avoid running indefinitely.

    5. Create a set visited_nodes to keep track of visited nodes to avoid loops.

    6. While the candidate_paths queue is not empty and we have not reached the maximum iterations:
        Get the path with the shortest length from the candidate_paths queue.
        If the length of the current path is greater than the shortest_path_length multiplied by the length_tolerance_factor, continue to the next iteration.
        If the current path's ending node is the same as the ending node we are searching for:
            Add the path to goal_paths if it is shorter or equal in length to the current shortest path.
            Update shortest_path_length if needed.
        If the current path does not reach the ending node:
            Explore adjacent edges of the ending node of the current path.
            For each neighbor node that has not been visited:
                Calculate the length of the new path by adding the edge's length.
                Create a new path by appending the edge to the current path.
                Put the new path with its length into the candidate_paths queue.
                Mark the neighbor node as visited in the visited_nodes set.

    7. Decrease the maximum iterations count in each iteration to prevent infinite loops.

    8. Return the goal_paths as the list of discovered paths.

This algorithm ensures that the discovered paths are the shortest possible within the defined tolerance factor, allowing for cyclic paths in the graph.

This implementation targets Python 3.9 and only uses the Python standard library, ensuring portability and compatibility.
"""

from functools import total_ordering
from typing import Any, List, Optional, List, Tuple, cast
from queue import PriorityQueue

class Node:
    """A node in a graph."""

    def __init__(self, id: int):
        self.id: int = id
        self.adjacent_edges: List["UndirectedEdge"] = []

    def edge_to(self, other: "Node") -> Optional["UndirectedEdge"]:
        """Returns the edge between the current node and the given one (if existing)."""
        matches = [edge for edge in self.adjacent_edges if edge.other_end(self) == other]
        return matches[0] if len(matches) > 0 else None

    def is_adjacent(self, other: "Node") -> bool:
        """Returns whether there is an edge between the current node and the given one."""
        return other in {edge.other_end(self) for edge in self.adjacent_edges}

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.id == other.id

    def __le__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.id <= other.id

    def __hash__(self) -> int:
        return self.id

    def __repr__(self) -> str:
        return f"Node({self.id})"


class UndirectedEdge:
    """An undirected edge in a graph."""
    def __init__(self, end_nodes: Tuple[Node, Node], length: float):
        self.end_nodes: Tuple[Node, Node] = end_nodes
        if 0 < length:
            self.length: float = length
        else:
            raise ValueError(
                f"Edge connecting {end_nodes[0].id} and {end_nodes[1].id}: "
                f"Non-positive length {length} not supported."
            )

        if any(e.other_end(end_nodes[0]) == end_nodes[1] for e in end_nodes[0].adjacent_edges):
            raise ValueError("Duplicate edges are not supported")

        self.end_nodes[0].adjacent_edges.append(self)
        if self.end_nodes[0] != self.end_nodes[1]:
            self.end_nodes[1].adjacent_edges.append(self)
        self.end_node_set = set(self.end_nodes)

    def other_end(self, start: Node) -> Node:
        """Returns the other end of the edge, given one of the end nodes."""
        return self.end_nodes[0] if self.end_nodes[1] == start else self.end_nodes[1]

    def is_adjacent(self, other_edge: "UndirectedEdge") -> bool:
        """Returns whether the current edge shares an end node with the given edge."""
        return len(self.end_node_set.intersection(other_edge.end_node_set)) > 0

    def __repr__(self) -> str:
        return (
            f"UndirectonalEdge(({self.end_nodes[0].__repr__()}, "
            f"{self.end_nodes[1].__repr__()}), {self.length})"
        )


class UndirectedGraph:
    """A simple undirected graph with edges attributed with their length."""

    def __init__(self, edges: List[UndirectedEdge]):
        self.edges: List[UndirectedEdge] = edges
        self.nodes_by_id = {node.id: node for edge in self.edges for node in edge.end_nodes}

@total_ordering
class UndirectedPath:
    """An undirected path through a given graph."""

    def __init__(self, nodes: List[Node]):
        assert all(
            node_1.is_adjacent(node_2) for node_1, node_2 in zip(nodes[:-1], nodes[1:])
        ), "Path edges must be a chain of adjacent nodes"
        self.nodes: List[Node] = nodes
        self.length = sum(
            cast(UndirectedEdge, node_1.edge_to(node_2)).length
            for node_1, node_2 in zip(nodes[:-1], nodes[1:])
        )

    @property
    def start(self) -> Node:
        return self.nodes[0]

    @property
    def end(self) -> Node:
        return self.nodes[-1]

    def prepend(self, edge: UndirectedEdge) -> "UndirectedPath":
        if self.start not in edge.end_nodes:
            raise ValueError("Edge is not adjacent")
        return UndirectedPath([edge.other_end(self.start)] + self.nodes)

    def append(self, edge: UndirectedEdge) -> "UndirectedPath":
        if self.end not in edge.end_nodes:
            raise ValueError("Edge is not adjacent")
        return UndirectedPath(self.nodes + [edge.other_end(self.end)])

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, UndirectedPath) and self.nodes == other.nodes

    def __le__(self, other: Any) -> bool:
        return isinstance(other, UndirectedPath) and self.length <= other.length

    def __hash__(self) -> int:
        return hash(n.id for n in self.nodes)

    def __repr__(self) -> str:
        nodestr: str = ", ".join([node.__repr__() for node in self.nodes])
        return f"UndirectedPath([{nodestr}])"


def compute_shortest_paths(
    graph: UndirectedGraph, start: Node, end: Node, length_tolerance_factor: float
) -> List[UndirectedPath]:
    """Computes and returns the N shortest paths between the given end nodes.

    The discovered paths always contain the shortest path between the two nodes. In addition, the
    second shortest, third shortest and following paths are also added (in ascending order by path
    length) up to (excluding) the path whose length is larger than the length of the shortest path
    multiplied with the given tolerance factor.

    We do not constrain this function to acyclic paths, i.e., cyclic paths should be found as well.

    For a given input of start node A, end node B and a tolerance factor of 2, the result has to
    contain all paths from A to B whose length is at most twice the length of the shortest path
    from A to B.

    Args:
        graph: The undirected graph in which the N shortest paths shall be found.
        start: The start node of the paths
        end: The end node of the paths
        length_tolerance_factor: The maximum length ratio which is allowed for the discovered paths
            (minimum: 1.0, maximum: infinite)

    Returns:
        The discovered paths. If no path from A to B exists, the result shall be empty.
    """
    if start == end:
        raise ValueError('Start and end nodes are identical!')
    goal_paths = []
    shortest_path_length = 100
    candidate_paths = PriorityQueue()
    candidate_paths.put((0, UndirectedPath([start])))
    while not candidate_paths.empty():
        _, current_path = candidate_paths.get()
        if current_path.length > shortest_path_length * length_tolerance_factor:
            continue
        if current_path.end == end:
            goal_paths.append(current_path)
            shortest_path_length = min(goal_paths).length
        for edge in current_path.end.adjacent_edges:
            neighbor = edge.other_end(current_path.end)
            new_path = current_path.append(edge)
            candidate_paths.put((new_path.length, new_path))
    return goal_paths


# Usage example
n1, n2, n3, n4 = Node(1), Node(2), Node(3), Node(4)
demo_graph = UndirectedGraph(
    [
        UndirectedEdge((n1, n2), 10),
        UndirectedEdge((n1, n3), 30),
        UndirectedEdge((n2, n4), 10),
        UndirectedEdge((n3, n4), 10),
    ]
)

# Should print the path [1, 2, 4]
print(compute_shortest_paths(demo_graph, n1, n4, 1.0))

# Should print the paths [1, 2, 4], [1, 2, 4, 2, 4], [1, 2, 1, 2, 4], [1, 2, 4, 3, 4], [1, 3, 4]
print(compute_shortest_paths(demo_graph, n1, n4, 2.0))
