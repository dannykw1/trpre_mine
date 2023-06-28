import tensorflow as tf
import numpy as np
from tensorflow import keras


def house_model():
    xs = np.array([1,2,3,4,5,6], dtype= float)
    ys = np.array([100000,150000,200000,250000,300000,350000],dtype =float)

    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=1000, verbose=1)
    return model


model = house_model()
new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)