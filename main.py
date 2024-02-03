import tensorflow as tf
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x = []
y = []
with open('data3.txt', 'r') as file:
    for line in file.readlines()[1:]:
        xi, yi = [float(i) for i in line.split(' ')]
        x.append(xi)
        y.append(yi)
x = np.array(x)
y = np.array(y)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

class Polynomial(keras.layers.Layer):
    def __init__(self, grade, **kwargs):
        super(Polynomial, self).__init__(**kwargs)
        self.grade = grade

    def build(self, input_shape):
        super(Polynomial, self).build(input_shape)

    def call(self, inputs):
        return tf.concat([inputs**i for i in range(1, self.grade + 1)], axis=-1)

poly = 3
model = keras.Sequential([keras.Input(shape=(1,)), Polynomial(grade=poly), keras.layers.Dense(1, bias_initializer=tf.constant_initializer(1))])
print(model.summary())

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(xtrain, ytrain, batch_size=64, epochs=100, validation_data=(xtest, ytest))

alphas = model.layers[-1].get_weights()
print("final weights: ", alphas)

loss, accuracy = model.evaluate(xtest, ytest)
print("Loss:", loss)
print("Accuracy:", accuracy)

ypred = model.predict(xtest)
plt.scatter(xtrain, ytrain)
plt.scatter(xtest, ypred, c='r')
plt.show()