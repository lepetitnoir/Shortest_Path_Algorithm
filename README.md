# Shortest path algorithm

An algorithm that finds the shortest path in a given network

# Summary 

Usage of railroad networks has to be efficiently scheduled in order for trains
to run on an optimum scale. This algorithm helps find to the shortest path between two destinations, including cycles ("back and forth") for a given
journey.

# How to run it

- Enter pip install -r requirements.txt

- Construct desired graph: Enter nodes and lenghts

- Call function: Enter constructed graph, chosen path (start/end node), tolerance factor (float)

- Alternatively enter python3 path_discovery_solution.py for demo run

- To run unit tests: Enter pytest test_path_discovery.py


# Steps taken

Firtstly the algorithm was implemented as the function compute_shortest_path in recourse to existing classes for nodes, edges graphs and paths of the network (Node, UndirectedEdge, UndirectedGraph, UndirectedPath). It is designed to ouput a list of nodes comprising said shortest path and second, third etc. shortest path while adhering to a given tolerance factor. Secondly a series of unit tests was created to test the performance and funcionality of the algorithm.

# Problems 

Since the length of the shortest path is not known in advance, it is initialized
with a fairly large number and then updated during execution. This means that 
running time can be exceedingly long in case there is no path meaning no adjacent edges to the final goal node. 

# Future improvements

One possible improvement might be to implement a classical dijkstra algorithm
which is guaranteed to find the shortest path. This way the shortest path length
can be computed in advance before computing others paths in the loop. This would
solve the runtime problem mentioned above.

Another perhaps more didactic point would be to combine the algorithm with a visualization library e.g. matplotlib to visualize the graph and make computed routes graphically available for other branches of the enterprise, customers 
or the general public.
