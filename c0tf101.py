import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

print(diabetes.data.shape, diabetes.target.shape)

diabetes.data[0:3]

diabetes.target[:3]

plt.scatter(diabetes.data[:, 2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x = diabetes.data[:, 2]
y = diabetes.target

for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
print(w, b)

for i in range(1, 100):
    for x_i, y_i in zip(x, y):
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 * err

print(w, b)

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
hist = model.fit(x, y, epochs=100, verbose=0)

model.summary()
model.weights
plt.plot(hist.history['loss'])
plt.show()


xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

def  tf101fit(x, y, epochs):
  history={"loss":[], "w":[], "b":[]}
  hismse=[];   hisw=[];   hisb=[]
  w=1; b=1
  for i in range(1, epochs):
    for x_i, y_i in zip(x, y):
      y_hat = x_i *w + b
      err = y_i - y_hat
      w_rate = x_i
      w = w + w_rate * err
      b = b + 1 * err
    history["loss"].append(chkmse(x,y,w,b))
    history['w'].append(w)
    hisb.append(b)
  print(f'{w=}', f'{b=}')
  return history

history = tf101fit(xs, ys, 10)
plt.plot(history['loss'])
plt.show()

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
hist = model.fit(xs, ys, epochs=100)

model.weights
plt.plot(hist.history['loss'])
plt.show()

model.predict([10.0])

class mytf:
    def __init__(self):
        self.w = 1.0
        self.b = 1.0
        self.history={"loss":[], "w":[], "b":[]}

    def chkmse(x, y, w, b):
        pred = [x_i * w + b for x_i in x]
        err = (pred - y) ** 2
        return err.mean();

    def fit(self, x,y, epochs=100):
        for i in range(epochs):
            for x_i, y_i in zip(x, y):
                y_hat = x_i * self.w + self.b
                err = -(y_i - y_hat)
                self.w = self.w + x_i * err
                self.b = self.b + 1 * err
                self.history["loss"].append(self.chkmse(x, y, w, b))
                self.history['w'].append(w)
                self.history['b'].append(b)
        return self.history