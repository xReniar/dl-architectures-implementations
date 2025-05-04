import tensorflow as tf
from tensorflow import keras
from keras import layers, models

def residual_block(x, filters, stride=1):
    shortcut = x
    

    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)


    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)


    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Skip connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x
