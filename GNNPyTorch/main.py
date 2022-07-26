# Импортируем библиотеки
import torch
from torch.autograd import Variable
import collections
import time
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# Импортируем функции
from grid_graph import grid_graph, draw_graph
from coarsening import coarsen
from coarsening import lmax_L
from coarsening import perm_data

# Импортируем модель
from GraphConvNetModel import Graph_ConvNet_LeNet5

# Проверяем подключена ли CUDA
if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

# Загружаем датасет MNIST
(X_train, Y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

X_train, X_val = X_train[:-10000], X_train[-10000:]
y_train, y_val = Y_train[:-10000], Y_train[-10000:]

# Параметры графа
t_start = time.time()
grid_side = 28
number_edges = 8
metric = 'euclidean'
A = grid_graph(grid_side, number_edges, metric)  # Создаем граф евклидовой формы

# fig, ax = plt.subplots(figsize=(8, 8))
# draw_graph(A, ax=ax, size_factor=1, title='Full graph')
# plt.show()

# fig, ax = plt.subplots(figsize=(8, 8))
# ax = draw_graph(A, ax=ax, size_factor=1, title='graph')
# plt.show()

# Делаем граф более грубым, уменьшая число связей (https://www.youtube.com/watch?v=o0mhbHdfgTA)
coarsening_levels = 4
L, perm = coarsen(A, coarsening_levels)

fig, ax = plt.subplots(figsize=(8*coarsening_levels+8, 8), ncols=coarsening_levels+1)
for i in range(coarsening_levels+1):

    if i == 0:
        ax[i] = draw_graph(A, ax=ax[i], size_factor=1, title='Full graph')
    else:
        ax[i] = draw_graph(L[i], ax=ax[i], size_factor=1, title=f'coarsened graph level {i}', spring_layout=False)

plt.show()

fig, ax = plt.subplots(figsize=(8*coarsening_levels+8, 8), ncols=coarsening_levels+1)
for i in range(coarsening_levels+1):

    if i == 0:
        ax[i] = draw_graph(A, ax=ax[i], size_factor=1, title='Full graph')
    else:
        ax[i] = draw_graph(L[i], ax=ax[i], size_factor=1, title=f'coarsened graph level {i}', spring_layout=True)

plt.show()

# Вычисляем собственный вектор матрицы Кирхгофа. (https://www.youtube.com/watch?v=IdsV0RaC9jM)
# Матрица Кирхгофа - одно из представлений конечного графа с помощью матрицы.
# Матрица Кирхгофа представляет дискретный оператор Лапласа для графа.
lmax = []
for i in range(coarsening_levels):
    lmax.append(lmax_L(L[i]))
print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

# Индексируем данные изображения для образования бинарного дерева.
# perm - Cписок индексов для переупорядочивания матриц смежности и данных таким образом,
#        чтобы объединение двух соседей от слоя к слою образовывало бинарное дерево.
train_data = perm_data(X_train, perm)
val_data = perm_data(X_val, perm)
test_data = perm_data(X_test, perm)

print('Execution time: {:.2f}s'.format(time.time() - t_start))
del perm

# Параметры нейронной сети
D = train_data.shape[1]
CL1_F = 32
CL1_K = 25
CL2_F = 64
CL2_K = 25
FC1_F = 512
FC2_F = 10
net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]

# Обьявляем нейронную сеть и задаем туда параметры
net = Graph_ConvNet_LeNet5(net_parameters)
if torch.cuda.is_available():
    net.cuda()
print(net)

# Веса
L_net = list(net.parameters())

# Параметры обучения
learning_rate = 0.05
dropout_value = 0.5
l2_regularization = 5e-4
batch_size = 100
num_epochs = 30
train_size = train_data.shape[0]
nb_iter = int(num_epochs * train_size) // batch_size
print('num_epochs=', num_epochs, ', train_size=', train_size, ', nb_iter=', nb_iter)

# Optimizer
global_lr = learning_rate
global_step = 0
decay = 0.95
decay_steps = train_size
lr = learning_rate
optimizer = net.update(lr)

acc = []
acc_valid = []
lss = []
lss_valid = []

# Обучаем нейронную сеть
indices = collections.deque()
indices_val = collections.deque()
for epoch in tqdm(range(num_epochs)):

    # Перемешиваем датасет
    indices.extend(np.random.permutation(train_size))
    indices_val.extend(np.random.permutation(val_data.shape[0]))

    # Устанавливаем таймер
    t_start = time.time()

    running_loss = 0.0
    running_accuracy = 0
    running_loss_val = 0
    running_accuracy_val = 0
    running_total = 0
    while len(indices) >= batch_size:

        # Получаем Batch
        batch_idx = [indices.popleft() for i in range(batch_size)]
        train_x, train_y = train_data[batch_idx, :], y_train[batch_idx]
        train_x = Variable(torch.FloatTensor(train_x).type(dtypeFloat), requires_grad=False)
        train_y = train_y.astype(np.int64)
        train_y = torch.LongTensor(train_y).type(dtypeLong)
        train_y = Variable(train_y, requires_grad=False)

        # Forward
        y = net.forward(train_x, dropout_value, L, lmax)
        loss = net.loss(y, train_y, l2_regularization)
        loss_train = loss.item()

        # Accuracy
        acc_train = net.evaluation(y, train_y.data)

        # backward
        loss.backward()

        # Update
        global_step += batch_size  # to update learning rate
        optimizer.step()
        optimizer.zero_grad()

        # loss, accuracy
        running_loss += loss_train
        running_accuracy += acc_train
        running_total += 1

        # validation
        with torch.no_grad():
            # формируем val
            mask = np.random.choice(indices_val, batch_size, replace=False)
            val_x, val_y = val_data[mask], y_val[mask]
            val_x = Variable(torch.FloatTensor(val_x).type(dtypeFloat), requires_grad=False)
            val_y = val_y.astype(np.int64)
            val_y = torch.LongTensor(val_y).type(dtypeLong)
            val_y = Variable(val_y, requires_grad=False)

            # получаем loss
            y = net.forward(val_x, dropout_value, L, lmax)
            loss = net.loss(y, val_y, l2_regularization)
            loss_val = loss.item()

            # получаем accuracy
            acc_val = net.evaluation(y, val_y.data)

            running_loss_val += loss_val
            running_accuracy_val += acc_val

        # print
        if not running_total % 100:  # Выводим результат каждый 100ый batch
            print('epoch= %d, i= %4d, loss(batch)= %.4f, accuracy(batch)= %.2f' % (
            epoch + 1, running_total, loss_train, acc_train))

    # print
    t_stop = time.time() - t_start
    print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
          (epoch + 1, running_loss / running_total, running_accuracy / running_total, t_stop, lr))



    # Обновляем learning rate
    lr = global_lr * pow(decay, float(global_step // decay_steps))
    optimizer = net.update_learning_rate(optimizer, lr)

    acc.append((running_accuracy / running_total).item())
    acc_valid.append((running_accuracy_val / running_total).item())
    lss.append(running_loss / running_total)
    lss_valid.append(running_loss_val / running_total)

# Тестируем нейронную сеть
running_accuracy_test = 0
running_total_test = 0
indices_test = collections.deque()
indices_test.extend(range(test_data.shape[0]))
t_start_test = time.time()
while len(indices_test) >= batch_size:
    batch_idx_test = [indices_test.popleft() for i in range(batch_size)]
    test_x, test_y = test_data[batch_idx_test, :], y_test[batch_idx_test]
    test_x = Variable(torch.FloatTensor(test_x).type(dtypeFloat), requires_grad=False)
    y = net.forward(test_x, 0.0, L, lmax)
    test_y = test_y.astype(np.int64)
    test_y = torch.LongTensor(test_y).type(dtypeLong)
    test_y = Variable(test_y, requires_grad=False)
    acc_test = net.evaluation(y, test_y.data)
    running_accuracy_test += acc_test
    running_total_test += 1
t_stop_test = time.time() - t_start_test
print('Accuracy(test) = %.3f %%, time= %.3f' % (running_accuracy_test / running_total_test, t_stop_test))

plt.figure(figsize=(10, 5))
plt.plot(acc, label='acc')
plt.plot(acc_valid, label='val_acc')
plt.legend()
plt.grid()

plt1 = plt.twinx()
plt1.set_ylabel('loss')
plt1.plot(lss, label='loss', color='aqua')
plt1.plot(lss_valid, label='val_loss', color='goldenrod')

plt.legend()
plt.show()
