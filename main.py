import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import pandas as pd

###### Loading Data ######
training_data = np.loadtxt("data/train.csv", delimiter=",", dtype=str)[1:]
x_test = np.loadtxt("data/test.csv", delimiter=",", dtype=str)[1:].astype('int32')
test = pd.read_csv('data/test.csv')
# print(training_data[:5])
y_train, x_train = training_data[:, 0].reshape((-1, 1)).astype('int32'), training_data[:, 1:].astype('int32')
x_train = x_train.reshape((len(x_train), 28, 28))
x_test = x_test.reshape((len(x_test), 28, 28))
# print(X_train.shape, X_test.shape)


###### Data Analysis ######
# plt.figure()
# plt.imshow(X_train[5])
# plt.colorbar()
# plt.grid(False)
# plt.show()

plt.figure(figsize=(10, 8))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.grid(False)
    plt.xlabel(y_train[i])
plt.show()

###### Models ######
model = Sequential(
    [
        Flatten(input_shape=(28, 28)),
        Dense(units=25, activation="relu", name="layer1"),
        Dense(units=18, activation="relu", name="layer2"),
        Dense(units=13, activation="relu", name="layer3"),
        Dense(units=10, activation="linear", name="layer4")
    ], name="my_model"
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

model.fit(x_train, y_train, epochs=50)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
y_pred = probability_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)


###### Submission ######
output = pd.DataFrame({'ImageId': test.index + 1, 'Label': y_pred})
output.to_csv('submission.csv', index=False)