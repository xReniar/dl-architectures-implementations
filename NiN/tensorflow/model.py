import tensorflow as tf
from tensorflow import keras
from keras import layers

def NiN(num_classes):
    model = keras.Sequential([
        
        # 1 NiN Block
        layers.Conv2D(96, kernel_size=5, strides=1, input_shape=(224, 224, 3), use_bias=False),
        layers.ReLU(),
        # 1X1 Convolutions
        layers.Conv2D(192, kernel_size=1, strides=1, padding='valid', activation='relu', use_bias=False),
        layers.Conv2D(96, kernel_size=1, strides=1, padding='valid', activation='relu', use_bias=False),
        # Max Pooling
        layers.ZeroPadding2D(padding=1),
        layers.MaxPooling2D(pool_size=3, strides=2),




        # 2 NiN Block
        layers.ZeroPadding2D(padding=2),
        layers.Conv2D(384, kernel_size=5, strides=1, use_bias=False),
        layers.ReLU(),
        # 1X1 Convolutions
        layers.Conv2D(192, kernel_size=1, strides=1, padding='valid', activation='relu', use_bias=False),
        layers.Conv2D(192, kernel_size=1, strides=1, padding='valid', activation='relu', use_bias=False),
        # Max Pooling
        layers.ZeroPadding2D(padding=1),
        layers.MaxPooling2D(pool_size=3, strides=2),




        # 3 NiN Block
        layers.ZeroPadding2D(padding=1),
        layers.Conv2D(192, kernel_size=3, strides=1, use_bias=False),
        layers.ReLU(),
        # 1X1 Convolutions
        layers.Conv2D(192, kernel_size=1, strides=1, padding='valid', activation='relu', use_bias=False),
        layers.Conv2D(num_classes, kernel_size=1, strides=1, padding='valid', activation='relu', use_bias=False),




        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # For classification
        layers.Dense(num_classes, activation='softmax')
    ])

    return model



# Test
num_classes = 1000
model = NiN(num_classes)

model.summary()
