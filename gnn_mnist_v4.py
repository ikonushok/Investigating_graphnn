# https://towardsdatascience.com/deep-learning-on-graphs-convolution-is-all-you-need-3c1cf8f1e715
# https://github.com/wangz10/gcn-playground/blob/master/MNIST_experiments.ipynb
# https://github.com/danielegrattarola/spektral
# https://github.com/danielegrattarola/spektral/blob/master/examples/other/graph_signal_classification_mnist.py

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Dense, BatchNormalization, Dropout
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.data import MixedLoader
from spektral.datasets.mnist import MNIST
from spektral.layers import GCNConv, GlobalSumPool
from spektral.utils.sparse import sp_matrix_to_sp_tensor


# Parameters
batch_size = 1028  # Batch size
epochs = 1000  # Number of training epochs
patience = 10  # Patience for early stopping
l2_reg = 5e-4  # Regularization rate for l2

# Load data
data = MNIST()

# The adjacency matrix is stored as an attribute of the dataset.
# Create filter for GCN and convert to sparse tensor.
data.a = GCNConv.preprocess(data.a)
data.a = sp_matrix_to_sp_tensor(data.a)

# Train/valid/test split
data_tr, data_te = data[:-10000], data[-10000:]
np.random.shuffle(data_tr)
data_tr, data_va = data_tr[:-10000], data_tr[-10000:]

# We use a MixedLoader since the dataset is in mixed mode
loader_tr = MixedLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = MixedLoader(data_va, batch_size=batch_size)
loader_te = MixedLoader(data_te, batch_size=batch_size)


# Build model
class Net(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(32, activation="relu", kernel_regularizer=l2(l2_reg))
        self.batch = BatchNormalization()
        self.conv1 = GCNConv(32, activation="relu", kernel_regularizer=l2(l2_reg))
        self.batch = BatchNormalization()
        self.conv2 = GCNConv(32, activation="relu", kernel_regularizer=l2(l2_reg))
        self.batch = BatchNormalization()
        self.flatten = GlobalSumPool()
        self.fc1 = Dense(512, activation="relu")
        # self.drop = Dropout(0.5)
        self.fc2 = Dense(10, activation="softmax")  # MNIST has 10 classes

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)

        return output


# Create model
model = Net()
optimizer = Adam()
loss_fn = SparseCategoricalCrossentropy()


# Training function
@tf.function
def train_on_batch(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
        acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc


# Evaluation function
def evaluate(loader):
    step = 0
    results = []
    for batch in loader:
        step += 1
        inputs, target = batch
        predictions = model(inputs, training=False)
        loss = loss_fn(target, predictions)
        acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
        results.append((loss, acc, len(target)))  # Keep track of batch size
        if step == loader.steps_per_epoch:
            results = np.array(results)
            return np.average(results[:, :-1], 0, weights=results[:, -1])


# Setup training
best_val_loss = 99999
current_patience = patience
step = 0
epoch = 0
# Training loop
results_tr = []
for batch in loader_tr:
    step += 1

    # Training step
    inputs, target = batch
    loss, acc = train_on_batch(inputs, target)
    results_tr.append((loss, acc, len(target)))

    if step == loader_tr.steps_per_epoch:
        results_va = evaluate(loader_va)
        if results_va[0] < best_val_loss:
            best_val_loss = results_va[0]
            current_patience = patience
            results_te = evaluate(loader_te)
        else:
            current_patience -= 1
            if current_patience == 0:
                print("Early stopping")
                break

        # Print results
        epoch += 1
        results_tr = np.array(results_tr)
        results_tr = np.average(results_tr[:, :-1], 0, weights=results_tr[:, -1])
        print(
            f"{epoch}:\t\t"
            f"Train loss: {round(results_tr[0], 4)}, acc: {round(results_tr[1], 4)}\t\t"
            f"Valid loss: {round(results_va[0], 4)}, acc: {round(results_va[1], 4)}\t\t"
            f"Test loss: {round(results_te[0], 4)}, acc: {round(results_te[1], 4)}"
        )

        # Reset epoch
        results_tr = []
        step = 0