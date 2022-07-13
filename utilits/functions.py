import networkx as nx
from matplotlib import pyplot as plt


def show_graph(G, title='Graf'):

    print(f'\nGraph:\n{G}\n'
          f'\nGraph nodes:\t{G.nodes()},\t{type(list(G.nodes()))}'
          f'\nGraph edges:\t{G.edges()},\t{type(G.edges())}'
          f'\n{G}\nGraph is directed: {G.is_directed()}'
          )

    # выберем способ отрисоки графа
    # pos = nx.spring_layout(G, k=5)
    pos = nx.circular_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.planar_layout(G)
    # pos = nx.random_layout(G)
    # pos = nx.spectral_layout(G)
    # pos = nx.shell_layout(G)

    # установим метки и цвета для связей
    # edge_labels = {edge_attribute: G.edges[edge_attribute]['weight'] for edge_attribute in G.edges}
    edge_colors = {edge_attribute: G.edges[edge_attribute]['color'] for edge_attribute in G.edges}
    # отметим метки связей
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # print(f'edge_labels:\t{edge_labels}')
    print(f'edge_colors:\t{edge_colors}')

    # сделаем часть связей пунктиром
    # dotted_line = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == 2]
    # solid_line = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == 1]
    # edges
    # nx.draw_networkx_edges(G, pos, edgelist=solid_line, width=1)
    # nx.draw_networkx_edges( G, pos, edgelist=dotted_line, width=3, alpha=1, edge_color="gray", style="dotted")

    # nodes_colors = ['lightblue', 'white', 'lightblue', 'white', 'lightblue', 'lightblue', 'red', 'blue', 'olive', 'lightblue']

    plt.title(title)
    nx.draw(G,
            pos,
            node_size=1000,
            edge_color=edge_colors.values(),
            # node_color=nodes_colors,
            node_color='lightblue',
            with_labels=True)
    plt.show()
