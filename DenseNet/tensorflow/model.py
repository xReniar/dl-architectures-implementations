from tensorflow import keras
from keras import layers, Model, Input
from dense_blocks import dense_block, transition_block



def build_densenet121(input_shape, growth_rate, reduction, num_classes):
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Dense 1
    x = dense_block(x, 6, growth_rate)
    x = transition_block(x, reduction)


    # Dense 2
    x = dense_block(x, num_layers=12, growth_rate=growth_rate)
    x = transition_block(x, reduction)

    # Dense 3
    x = dense_block(x, 24, growth_rate)
    x = transition_block(x, reduction)

    # Dense 4
    x = dense_block(x, 16, growth_rate)


    x = layers.GlobalAveragePooling2D()(x)    
    x = layers.Dense(1000, activation='relu')(x)


    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model



# Test
input_shape = (224, 224, 3)
growth_rate = 32
reduction = 0.5
num_classes = 10

model = build_densenet121(input_shape, growth_rate, reduction, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()