# https://medium.com/geekculture/graphs-or-networks-chapter-1-57aa9497be06
# https://medium.com/geekculture/graphs-or-networks-chapter-2-2af64596858e
# https://medium.com/geekculture/graph-or-networks-depth-first-search-algorithm-d3d4f6a66f01
# https://medium.com/geekculture/graph-or-networks-chapter-4-a04a34a2f084

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(42)

"""
Let us create a small graph
"""
# Initializing networkx graph
graph = nx.Graph()

# Adding nodes to graph
graph.add_node(1, name="A")
graph.add_node(2, name="B")
graph.add_node(3, name="C")
graph.add_node(4, name="D")

# Connecting nodes via links
graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph.add_edge(2, 4)

# Plotting graph
nx.draw(graph,
        node_size=1000,
        node_color='lightblue',
        with_labels=True)
plt.show()


"""
Let us see how a graph will look with edge weights
"""
data = [('Delhi', 'Kolkata', '1'),
        ("Delhi", "Mumbai", '0.7'),
        ("Ranchi", "Kolkata", '0.43'),
        ("Ranchi", "Amlapuram", '0.78'),
        ("Delhi", "Amlapuram", '0.66')
        ]
df = pd.DataFrame(data=data, columns=['city1', 'city2', 'distance'])
print(f'\n{df}\n')

graph = nx.from_pandas_edgelist(df=df,
                                source='city1',
                                target='city2',
                                edge_attr='distance',
                                create_using=nx.DiGraph()  # Показать направленность графа
                                )
pos = nx.spring_layout(graph)
nx.draw(graph,
        pos,
        node_size=1000,
        node_color='lightblue',
        with_labels=True)
labels = {edge_attribute: graph.edges[edge_attribute]['distance'] for edge_attribute in graph.edges}
nx.draw_networkx_edge_labels(graph,
                             pos,
                             edge_labels=labels)
plt.show()


"""
Heterogenous graph: 
A heterogeneous graph is a special kind of information network, 
which contains either multiple types of vertices or multiple types of edges. 
Types of vertices and edges are also referred as vertex label and edge labels.
"""
data = [('Delhi', 'Kolkata', '1'),
        ("Delhi", "Mumbai", '0.7'),
        ("Ranchi", "Kolkata", '0.43'),
        ("Ranchi", "Amlapuram", '0.78'),
        ("Delhi", "Amlapuram", '0.66'),
        ("Ram", "Delhi", '1'),
        ("Ram", "Mumbai", '0.5')
        ]
df = pd.DataFrame(data=data, columns=['city1', 'city2', 'distance'])
print(df)

graph = nx.from_pandas_edgelist(df=df,
                                source='city1',
                                target='city2',
                                edge_attr='distance',
                                create_using=nx.DiGraph())
pos = nx.spring_layout(graph, k=1)

nx.draw(graph,
        pos,
        node_size=1000,
        node_color='lightblue',
        with_labels=True)
labels = {edge_attribute: graph.edges[edge_attribute]['distance'] for edge_attribute in graph.edges}
nx.draw_networkx_edge_labels(graph,
                             pos,
                             edge_labels=labels)
plt.show()


"""
Adjacency Matrix: 
It is a matrix whose (i, j) position represents an edge running from i to j node 
in the graph and value at the position refers to edge weight.
Advantages:
- It is space efficient for dense graphs and complete graphs.
- To lookup edge weight time complexity O(1).
- Simplest Representation of graph.
"""
# Матрица смежности
# https://networkx.org/documentation/stable/reference/generated/networkx.linalg.graphmatrix.adjacency_matrix.html
A = nx.to_scipy_sparse_array(graph, nodelist=list(graph.nodes()))
print(f'\nAdjacency Matrix (матрица смежности):\n{A.todense()}\n')
# Диагональная матрица
# https://networkx.org/documentation/stable/reference/generated/networkx.linalg.graphmatrix.adjacency_matrix.html
D = A.setdiag(A.diagonal() * 2)
print(f'Graph Диагональная матрица:\n{D}')


"""
Adjacency List: It represents the graph as map from nodes to list of edges.
Advantages:
- Space efficient for sparse graph.
- Iterating over edges is efficient.
Disadvantages:
- Less space efficient for dense graphs
- Edge weight look up O(E)
Edge List: It is unordered list of edge triplets.
Advantages:
- efficient for sparse graphs
- all edge iteration is simple
Disadvantages:
- edge weight lookup is O(E)
- not efficient for dense graphs.
"""
print(f'\nGraph nodes:\t{graph.nodes()},\t{type(list(graph.nodes()))}'
      f'\nEdges:\t{graph.edges()}'
      f'\n{graph}\nGraph is directed: {graph.is_directed()}'
      )




data = [('A','B','1'),("A","C",'2'),("C","D",'6'),("C","E",'4'),
        ("B","C",'3'),("F","D",'7'),("D","B",'5')]
df = pd.DataFrame(data=data,columns=['node1','node2','edge'])
graph = nx.from_pandas_edgelist(df=df,
                                source='node1',
                                target='node2',
                                edge_attr='edge',
                                create_using=nx.Graph()
                                )
pos = nx.spring_layout(graph)
nx.draw(graph,
        pos,
        node_size=1000,
        node_color='lightblue',
        with_labels=True)
labels = {edge_attribute: graph.edges[edge_attribute]['edge'] for edge_attribute in graph.edges}
nx.draw_networkx_edge_labels(graph,
                             pos,
                             edge_labels=labels)
plt.show()


"""
Depth First Search Algorithm
Обход графов является одной из самых интересных тем в теории графов,
а глубина первого поиска является одним из самых фундаментальных и простых алгоритмов обхода
"""
data = [('A','B','1'),("A","C",'2'),("C","D",'6'),("C","E",'4'),
        ("B","C",'3'),("F","D",'7'),("D","B",'5')]
df = pd.DataFrame(data=data,columns=['node1','node2','edge'])
graph = nx.from_pandas_edgelist(df=df, source='node1', target='node2', edge_attr='edge',create_using=nx.Graph())

total_nodes = len(graph.nodes)
is_seen = [False] * total_nodes

def depth_first_algo(graph,start_node='A'):
    print(start_node,end='--->')
    start_node_index = list(graph.nodes).index(start_node)
    if is_seen[start_node_index]:
        print('retracting from',start_node)
        return 0
    is_seen[start_node_index] = True
    node_neighbours = graph.neighbors(start_node)
    for neighbour in node_neighbours:
        print('neighbour:',neighbour)
        depth_first_algo(graph,start_node=neighbour)
depth_first_algo(graph)