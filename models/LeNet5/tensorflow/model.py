import tensorflow as tf
from tensorflow import keras
from keras import layers


def LeNet5(num_classes):
    model = keras.Sequential([
          
        # 1 
        layers.Conv2D(6, kernel_size=5, strides=1, input_shape=(32, 32, 1), use_bias=False),
        layers.Activation('tanh'),


        # 2
        layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'),


        # 3
        layers.Conv2D(16, kernel_size=5, strides=1, use_bias=False),
        layers.Activation('tanh'),


        # 4
        layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'),


        # 5
        layers.Conv2D(120, kernel_size=5, strides=1, use_bias=False),
        layers.Activation('tanh'),

    
        layers.Flatten(),

        
        # Dense layers
        
        layers.Dense(84, activation='tanh'),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Test
num_classes = 1000
model = LeNet5(num_classes)

model.summary()
