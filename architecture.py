import tensorflow as tf
import keras_cv
    
def build_conv_cascade(filters: int, drop_proba: float, drop_size: int) -> tf.keras.Sequential:

    kernel_initializer = 'he_normal'

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, 3, kernel_initializer=kernel_initializer, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        keras_cv.layers.DropBlock2D(drop_proba, drop_size),
        tf.keras.layers.Conv2D(filters, 3, kernel_initializer=kernel_initializer, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])
    
def build_reduction_sequential(filters: int, drop_proba: float, drop_size: int, prev_node: tf.keras.layers.Layer):

    seq = build_conv_cascade(filters, drop_proba, drop_size)(prev_node)
    pooling = tf.keras.layers.MaxPool2D()(seq)

    return seq, pooling

def build_expansion_sequential(filters: int, drop_proba: float, drop_size: int, prev_node: tf.keras.layers.Layer, concat_with: tf.keras.Sequential) -> tf.keras.Sequential:

    kernel_initializer = 'he_normal'

    e = tf.keras.layers.Conv2DTranspose(filters, 2, 2, 'same', kernel_initializer=kernel_initializer)(prev_node)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.ReLU()(e)
    e = tf.keras.layers.Concatenate()([e, concat_with])

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, 3, 1, 'same', kernel_initializer=kernel_initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        keras_cv.layers.DropBlock2D(drop_proba, drop_size),
        tf.keras.layers.Conv2D(filters, 3, 1, 'same', kernel_initializer=kernel_initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])(e)

def build_model(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:

    input_layer = tf.keras.layers.Input(input_shape)

    r1, i1 = build_reduction_sequential(16, 0, 1, input_layer)

    r2, i2 = build_reduction_sequential(32, 0, 1, i1)

    r3, i2 = build_reduction_sequential(64, 0.1, 16, i2)

    r4, i3 = build_reduction_sequential(128, 0.15, 8, i2)

    r5 = build_conv_cascade(256, 0.2, 4)(i3)

    e = build_expansion_sequential(128, 0.15, 8, r5, r4)

    e = build_expansion_sequential(64, 0.1, 16, e, r3)

    e = build_expansion_sequential(32, 0, 1, e, r2)

    e = build_expansion_sequential(16, 0, 1, e, r1)

    output = tf.keras.layers.Conv2D(num_classes, 1, 1, 'same', activation='softmax')(e)

    return tf.keras.Model(inputs=input_layer, outputs=output)
