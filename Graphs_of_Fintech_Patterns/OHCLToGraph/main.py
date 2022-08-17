# python_candlestick_chart.py
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import yfinance as yf
import pandas as pd
from PIL import Image
from sklearn import metrics
import networkx as nx

from Graphs_of_Fintech_Patterns.OHCLToGraph.functions import \
    draw_graph, grid_graph, get_pattern_from_img, create_pattern_imgs

if not os.path.isdir(f'../source_root/patterns'):
    os.mkdir(f'../source_root/patterns')
if not os.path.isdir(f'../source_root/graphs'):
    os.mkdir(f'../source_root/graphs')


destination_root = '../source_root'
# get historical market data
data = yf.download("VZ", start="2018-01-01")
data = data[['Open', 'High', 'Low', "Close"]]
print(data)


# Creating Images
quality = 70
create_pattern_imgs(data,
                    pattern_sizes=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    quality=quality,
                    address=f'{destination_root}/patterns'
                    )



d_digit_graphs = {}  # to collect feature graphs from each class
imgshape = quality
A = grid_graph(imgshape, k=8)
for idx, item in enumerate(os.listdir(f'{destination_root}/patterns')):

    pattern = get_pattern_from_img(f'{destination_root}/patterns/{item}')

    fig, axes = plt.subplots(figsize=(15, 5), ncols=3)
    # Выводим паттерн
    axes[0].imshow(pattern)

    # a sparse diag matrix with the intensities values on the diagonal
    A_diag_i = sp.diags(pattern.flatten(), dtype=np.float32).tolil()

    # "prune" the adjacency of the grid graph to preserve the subgraph with the data
    A_i = A.dot(A_diag_i)
    d_digit_graphs[idx] = A_i

    axes[1], _, _ = draw_graph(A_i, m=imgshape, ax=axes[1], size_factor=1)
    axes[2], G, pos = draw_graph(A_i, m=imgshape, ax=axes[2], size_factor=1, spring_layout=True)

    fig.tight_layout()
    plt.savefig(f'{destination_root}/graphs/graphs{idx}.png')
    plt.show()

    # make GIF
    # nx.draw(G, pos, node_size=2)
    # plt.savefig(f'../source_root/gif/graph{idx}.png')













exit()

imgshape = 25
d_digit_corr_graphs = {}  # build digit feature graph by correlation
for idx, item in enumerate(os.listdir(destination_root)):

    pattern_img = Image.open(f'{destination_root}/{item}')
    pattern_img = pattern_img.resize((imgshape, imgshape), Image.Resampling.LANCZOS)
    pattern_img = pattern_img.convert('L')
    pattern = np.array(pattern_img).reshape(imgshape * imgshape, 1) / 255
    pattern = np.abs(pattern - 1)  # меняем белый фон на черный - инвертируем паттерн
    # убираем шум
    pattern1 = np.where(pattern >= 0.5, pattern, 0)
    pattern = np.where(pattern1 == 0, pattern1, 1)
    assert len(np.unique(pattern)) == 2, "Из изображения не убраны шумы"

    dist = metrics.pairwise_distances(pattern.reshape(-1, imgshape*imgshape).T,
                                      metric='cosine',
                                      n_jobs=-2)
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

    x_train_i_avg = pattern
    axes[0].imshow(x_train_i_avg.reshape(imgshape, imgshape))

    # thresholding
    W = W.multiply(W > 0.8)
    d_digit_corr_graphs[idx] = W

    axes[1] = draw_graph(W, m=imgshape, ax=axes[1], size_factor=0.2)
    axes[2] = draw_graph(W, m=imgshape, ax=axes[2], size_factor=0.2, spring_layout=True)
    fig.tight_layout()
    plt.show()
