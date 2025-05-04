import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from residual_block import residual_block


def build_resnet18(input_shape=(224, 224, 3), num_classes=1000):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    filters = [64, 128, 256, 512]
    for i, f in enumerate(filters):
        strides = 1 if i == 0 else 2  
        x = residual_block(x, f, stride=strides)
        x = residual_block(x, f, stride=1)  

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ResNet18")

    return model



model = build_resnet18(input_shape=(224, 224, 3), num_classes=1000)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()