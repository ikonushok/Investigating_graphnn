import os
import random
import numpy as np
import tensorflow as tf
# import keras.backend as K
from keras.datasets import mnist
from sklearn import metrics

import graph
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt


# Устанавливаем случайный сид
SEED = 42
# os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
# tf.random.set_seed(SEED)

# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# K.set_session(sess)


# Load data
(X_train, Y_train), (X_test, y_test) = mnist.load_data()

X_train = np.array(X_train)
y_train = np.array(Y_train)

X_train = X_train.reshape(60000, 784).astype('float32') / 255

X_test = X_test.reshape(10000, 784).astype('float32') / 255

X_train, X_val = X_train[:-10000], X_train[-10000:]
y_train, y_val = Y_train[:-10000], Y_train[-10000:]

X_train, X_val, X_test = X_train[..., None], X_val[..., None], X_test[..., None]

N = X_train.shape[-2]      # Number of nodes in the graphs
F = X_train.shape[-1]      # Node features dimensionality
n_out = 10  # Dimension of the target



threshold = 0.25  # to reduce the noise for averaged signals

imgshape = 28
A = graph.grid_graph(imgshape, k=8)

for i in range(10):
    mask = y_train == i

    fig, axes = plt.subplots(figsize=(20, 5), ncols=4)

    x_train_i_avg = X_train[mask].mean(axis=0).flatten()
    axes[0].imshow(x_train_i_avg.reshape(28, 28))
    axes[0].set_title('original image', fontweight='bold')

    # threshold the averages of pixels
    x_train_i_avg[x_train_i_avg < threshold] = 0
    axes[1].imshow(x_train_i_avg.reshape(28, 28))
    axes[1].set_title('filtered image', fontweight='bold')

    # a sparse diag matrix with the intensities values on the diagnoal
    A_diag_i = sp.diags(x_train_i_avg, dtype=np.float32).tolil()

    # "prune" the adjacency of the grid graph to preserve the subgraph with the data
    A_i = A.dot(A_diag_i)

    axes[2] = graph.draw_graph(A_i, ax=axes[2], size_factor=1)

    axes[3] = graph.draw_graph(A_i, ax=axes[3], size_factor=1, spring_layout=True, title='Силовая визуализация Фрюхтермана-Рейнгольда')
    fig.tight_layout()


for i in range(10):
    mask = y_train == i

    dist = metrics.pairwise_distances(X_train[mask].reshape(-1, 784).T, metric='cosine', n_jobs=-2)

    W = sp.coo_matrix(1 - dist, dtype=np.float32)

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is sp.csr_matrix

    fig, axes = plt.subplots(figsize=(15, 5), ncols=3)

    x_train_i_avg = X_train[mask].mean(axis=0).flatten()
    axes[0].imshow(x_train_i_avg.reshape(28, 28))
    axes[0].set_title('original image', fontweight='bold')

    # thresholding
    W = W.multiply(W > 0.8)

    axes[1] = graph.draw_graph(W, ax=axes[1], size_factor=1, title='graph')

    axes[2] = graph.draw_graph(W, ax=axes[2], size_factor=1, spring_layout=True, title='Силовая визуализация Фрюхтермана-Рейнгольда')
    fig.tight_layout()
    plt.show()