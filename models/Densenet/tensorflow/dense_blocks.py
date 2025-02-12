import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input




def conv_block(x, growth_rate):
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(4 * growth_rate, (1, 1), padding='same')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
    
    return x


def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        conv_out = conv_block(x, growth_rate)
        x = layers.Concatenate()([x, conv_out])  # Concat. feature maps
    return x


def transition_block(x, reduction):
    filters = int(tf.keras.backend.int_shape(x)[-1] * reduction)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.AveragePooling2D((2, 2), strides=2, padding='same')(x)
    return x