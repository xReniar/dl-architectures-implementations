import tensorflow as tf
from tensorflow import keras
from keras import layers


def AlexNet(num_classes):
    model = keras.Sequential([
          
        layers.Conv2D(96, kernel_size=11, strides=4, input_shape=(227, 227, 3), use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=3, strides=2),


        layers.ZeroPadding2D(padding=2),
        layers.Conv2D(256, kernel_size=5, strides=1, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=3, strides=2),


        layers.ZeroPadding2D(padding=1),
        layers.Conv2D(384, kernel_size=3, strides=1, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),


        layers.ZeroPadding2D(padding=1),
        layers.Conv2D(384, kernel_size=3, strides=1, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),


        layers.ZeroPadding2D(padding=1),
        layers.Conv2D(256, kernel_size=3, strides=1, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=3, strides=2),

    
        layers.Flatten(),

        
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),


        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        
        
        layers.Dense(4096, activation='relu'),

 
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Test
num_classes = 1000
model = AlexNet(num_classes)

model.summary()
