import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, Input
from dense_blocks import dense_block, transition_block



def build_densenet(input_shape, num_blocks, num_layers_per_block, growth_rate, reduction, num_classes):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Costruzione dei blocchi densi e di transizione
    for i in range(num_blocks - 1):
        x = dense_block(x, num_layers_per_block, growth_rate)
        x = transition_block(x, reduction)
    
    # Ultimo blocco denso
    x = dense_block(x, num_layers_per_block, growth_rate)
    
    # Classificazione finale
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model



# Test
input_shape = (224, 224, 3)
num_blocks = 4
num_layers_per_block = 6
growth_rate = 32
reduction = 0.5
num_classes = 10

model = build_densenet(input_shape, num_blocks, num_layers_per_block, growth_rate, reduction, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()