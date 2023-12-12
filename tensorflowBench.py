import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

import numpy as np
import matplotlib.pyplot as plt
import timeit

# AMX Instructions
# https://blog.tensorflow.org/2023/01/optimizing-tensorflow-for-4th-gen-intel-xeon-processors.html
# remove for baseline
policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_global_policy(policy)

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# scaling image values between 0-1
X_train_scaled = X_train/255
X_test_scaled = X_test/255
# one hot encoding labels
y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'float32')
y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')

def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')    
    ])
    model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


# CPU
def bench():
    tf.device('/CPU:0')

    n = 2
    print('Tensorflow bench: ')
    print("tensorflow takes on average: {}s".format(timeit.timeit('model_cpu.fit(X_train_scaled, y_train_encoded, epochs = 1)', number=n, setup="model_cpu = get_model()", globals=globals())/n))

