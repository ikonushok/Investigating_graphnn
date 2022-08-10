# python_candlestick_chart.py
import os

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import scipy.sparse as sp
import yfinance as yf
import pandas as pd
from PIL import Image
from sklearn import metrics
import networkx as nx

from Graphs_of_Fintech_Patterns.OHCLToGraph.functions import draw_graph, grid_graph

destination_root = '../source_root/patterns'
# get historical market data
data = yf.download("VZ", start="2018-01-01")
data = data[['Open', 'High', 'Low', "Close"]]
print(data)

# Creating Images
imgshape = 70
for pattern_size in [5, 7, 10]:
    mpf.plot(data[4:pattern_size+4], type='candle', figsize=(1, 1), axisoff=True,
             savefig=f'{destination_root}/pattern{pattern_size}')
    # plt.show()

    pattern_img = Image.open(f'{destination_root}/pattern{pattern_size}.png')
    pattern_img = pattern_img.crop((20, 13, 90, 83))  # подобрал вручную
    pattern_img = pattern_img.save(f'{destination_root}/pattern{pattern_size}.png', quality=imgshape)
    # pattern_img = Image.open(f'{destination_root}/pattern{pattern_size}.png')
    # pattern_img = pattern_img.convert('L')
    # pattern = np.array(pattern_img) / 255
    # pattern = np.abs(pattern - 1)  # меняем белый фон на черный - инвертируем паттерн
    # убираем шум
    # temp = np.where(pattern >= 0.5, pattern, 0)
    # pattern = np.where(temp == 0, temp, 1)
    # assert len(np.unique(pattern)) == 2, "Из изображения не убраны шумы"
    # выводим паттерн
    # plt.imshow(pattern)
    # plt.show()


d_digit_graphs = {}  # to collect feature graphs from each class
# imgshape = 70
A = grid_graph(imgshape, k=8)
for idx, item in enumerate(os.listdir(destination_root)):
    pattern_img = Image.open(f'{destination_root}/{item}')
    pattern_img = pattern_img.convert('L')
    pattern = np.array(pattern_img) / 255
    pattern = np.abs(pattern - 1)  # меняем белый фон на черный - инвертируем паттерн
    # убираем шум
    pattern1 = np.where(pattern >= 0.5, pattern, 0)
    pattern = np.where(pattern1 == 0, pattern1, 1)
    assert len(np.unique(pattern)) == 2, "Из изображения не убраны шумы"

    fig, axes = plt.subplots(figsize=(15, 5), ncols=3)
    # Выводим паттерн
    pattern = pattern.flatten()
    axes[0].imshow(pattern.reshape(imgshape, imgshape))

    # a sparse diag matrix with the intensities values on the diagnoal
    A_diag_i = sp.diags(pattern, dtype=np.float32).tolil()

    # "prune" the adjacency of the grid graph to preserve the subgraph with the data
    A_i = A.dot(A_diag_i)
    d_digit_graphs[idx] = A_i

    axes[1] = draw_graph(A_i, m=imgshape, ax=axes[1], size_factor=1)
    axes[2] = draw_graph(A_i, m=imgshape, ax=axes[2], size_factor=1, spring_layout=True)

    fig.tight_layout()
    plt.show()


imgshape = 70
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
