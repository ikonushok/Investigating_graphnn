import sklearn
import sklearn.metrics
import scipy.sparse, scipy.sparse.linalg  # scipy.spatial.distance
import numpy as np
import networkx as nx
import graph


def grid_graph(grid_side,number_edges,metric):
    """Generate graph of a grid"""
    z = grid(grid_side)
    dist, idx = distance_sklearn_metrics(z, k=number_edges, metric=metric)
    A = adjacency(dist, idx)
    print("nb edges: ",A.nnz)
    return A


def grid(m, dtype=np.float32):
    """Return coordinates of grid points"""
    M = m**2
    x = np.linspace(0,1,m, dtype=dtype)
    y = np.linspace(0,1,m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M,2), dtype)
    z[:,0] = xx.reshape(M)
    z[:,1] = yy.reshape(M)
    return z

def draw_graph(A, m=28, ax=None, spring_layout=False, size_factor=10, title='graph'):
    '''
    Draw the graph given adjacency matrix(A),
    optionally with spring_layout.
    '''

    # assert m ** 2 == A.shape[0] == A.shape[1]
    # Create the nx.Graph object
    G = nx.from_scipy_sparse_array(A)
    print('Number of nodes: %d; Number of edges: %d' % \
          (G.number_of_nodes(), G.number_of_edges()))
    grid_coords = graph.grid(m)

    if spring_layout:
        # remove nodes without edges
        nodes_without_edges = [n for n, k in G.degree() if k == 0]
        G.remove_nodes_from(nodes_without_edges)
        print('After removing nodes without edges:')
        print('Number of nodes: %d; Number of edges: %d' % \
              (G.number_of_nodes(), G.number_of_edges()))

    z = graph.grid(m)

    # initial positions
    pos = {n: z[n] for n in G.nodes()}

    if spring_layout:
        pos = nx.spring_layout(G,
                               pos=pos,
                               iterations=200)

    ax.set_title(f'{title}', fontweight='bold')
    ax = nx.draw(G, pos,
                 node_size=[G.degree(n) * size_factor for n in G.nodes()],
                 ax=ax
                 )

    return ax

def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute pairwise distances"""
    #d = sklearn.metrics.pairwise.pairwise_distances(z, metric=metric, n_jobs=-2)
    d = sklearn.metrics.pairwise.pairwise_distances(z, metric=metric, n_jobs=1)
    # k-NN
    idx = np.argsort(d)[:,1:k+1]
    d.sort()
    d = d[:,1:k+1]
    return d, idx


def adjacency(dist, idx):
    """Return adjacency matrix of a kNN graph"""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0
    assert dist.max() <= 1

    # Pairwise distances
    sigma2 = np.mean(dist[:,-1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections
    W.setdiag(0)

    # Undirected graph
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W
