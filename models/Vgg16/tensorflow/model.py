import tensorflow as tf
from tensorflow import keras
from keras import layers


def vgg16(num_classes):
    model = keras.Sequential([
          
        # 1
        layers.Conv2D(64, kernel_size=3, strides=1, input_shape=(224, 224, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        # 2
        layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        
        layers.MaxPooling2D(pool_size=2, strides=2),
        
        
        # 3
        layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        # 4
        layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        
        layers.MaxPooling2D(pool_size=2, strides=2),
        
        
        # 5
        layers.Conv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        # 6
        layers.Conv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        # 7
        layers.Conv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        
        layers.MaxPooling2D(pool_size=2, strides=2),
        
        
        # 8
        layers.Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        # 9
        layers.Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        # 10
        layers.Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        
        layers.MaxPooling2D(pool_size=2, strides=2),
        
        
        # 11
        layers.Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        # 12
        layers.Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        # 13
        layers.Conv2D(512, kernel_size=3, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        
        layers.MaxPooling2D(pool_size=2, strides=2),
        

    
        layers.Flatten(),
        

        # 14
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),

        # 15
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),

        # 16
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Test
num_classes = 1000
model = vgg16(num_classes)

model.summary()
