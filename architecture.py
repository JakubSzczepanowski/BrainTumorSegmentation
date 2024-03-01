import tensorflow as tf
import keras_cv

class InceptionModule(tf.keras.layers.Layer):

    def __init__(self, filters: int, strides: int, activation: str, kernel_initializer: str, **kwargs):
        super().__init__(**kwargs)
        self.filters_per_path = filters // 4
        self.strides = strides
        self.activation = activation
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):

        self.conv1x1 = tf.keras.layers.Conv2D(self.filters_per_path, 1, self.strides, activation=self.activation, kernel_initializer=self.kernel_initializer, padding='same')
        
        self.sec_conv1x1 = tf.keras.layers.Conv2D(self.filters_per_path, 1, self.strides, activation=self.activation, kernel_initializer=self.kernel_initializer, padding='same')
        self.conv3x3 = tf.keras.layers.Conv2D(self.filters_per_path, 3, 1, activation=self.activation, kernel_initializer=self.kernel_initializer, padding='same')

        self.third_conv1x1 = tf.keras.layers.Conv2D(self.filters_per_path, 1, self.strides, activation=self.activation, kernel_initializer=self.kernel_initializer, padding='same')
        self.conv5x5 = tf.keras.layers.Conv2D(self.filters_per_path, 5, 1, activation=self.activation, kernel_initializer=self.kernel_initializer, padding='same')

        self.max_pool = tf.keras.layers.MaxPooling2D(3, self.strides, 'same')
        self.last_conv1x1 = tf.keras.layers.Conv2D(self.filters_per_path, 1, 1, 'same', activation=self.activation, kernel_initializer=self.kernel_initializer)

        super().build(input_shape)
        

    def call(self, input):

        result1 = self.conv1x1(input)

        temp2 = self.sec_conv1x1(input)
        result2 = self.conv3x3(temp2)

        temp3 = self.third_conv1x1(input)
        result3 = self.conv5x5(temp3)

        temp4 = self.max_pool(input)
        result4 = self.last_conv1x1(temp4)

        return tf.keras.layers.Concatenate()([result1, result2, result3, result4])
    
def build_conv_cascade(filters: int, drop_proba: float, drop_size: int, activation: str, kernel_initializer: str) -> tf.keras.Sequential:

    return tf.keras.Sequential([
        tf.keras.layers.DepthwiseConv2D(3, 1, 'same', filters // 4, activation=activation, kernel_initializer=kernel_initializer),
        keras_cv.layers.DropBlock2D(drop_proba, drop_size),
        tf.keras.layers.Conv2D(filters, 3, 1, 'same', activation=activation, kernel_initializer=kernel_initializer)
    ])
    
def build_reduction_sequential(filters: int, drop_proba: float, drop_size: int, activation: str, kernel_initializer: str, prev_node: tf.keras.layers.Layer) -> tuple[tf.keras.Sequential, InceptionModule]:

    seq = build_conv_cascade(filters, drop_proba, drop_size, activation, kernel_initializer)(prev_node)
    inception = InceptionModule(filters, 2, activation, kernel_initializer)(seq)

    return seq, inception

def build_expansion_sequential(filters: int, drop_proba: float, drop_size: int, activation: str, kernel_initializer: str, prev_node: tf.keras.layers.Layer, concat_with: tf.keras.Sequential) -> tf.keras.Sequential:

    e = tf.keras.layers.Conv2DTranspose(filters, 2, 2, 'same', activation=activation, kernel_initializer=kernel_initializer)(prev_node)
    e = tf.keras.layers.Concatenate()([e, concat_with])

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, 3, 1, 'same', activation=activation, kernel_initializer=kernel_initializer),
        keras_cv.layers.DropBlock2D(drop_proba, drop_size),
        InceptionModule(filters, 1, activation, kernel_initializer)
    ])(e)

def build_model(input_shape: tuple[int, int, int], num_classes: int, activation: str, kernel_initializer: str) -> tf.keras.Model:

    input_layer = tf.keras.layers.Input(input_shape)

    r1, i1 = build_reduction_sequential(16, 0, 1, activation, kernel_initializer, input_layer)

    r2, i2 = build_reduction_sequential(32, 0, 1, activation, kernel_initializer, i1)

    r3, i2 = build_reduction_sequential(64, 0.1, 16, activation, kernel_initializer, i2)

    r4, i3 = build_reduction_sequential(128, 0.15, 8, activation, kernel_initializer, i2)

    r5 = build_conv_cascade(256, 0.2, 4, activation, kernel_initializer)(i3)

    e = build_expansion_sequential(128, 0.15, 8, activation, kernel_initializer, r5, r4)

    e = build_expansion_sequential(64, 0.1, 16, activation, kernel_initializer, e, r3)

    e = build_expansion_sequential(32, 0, 1, activation, kernel_initializer, e, r2)

    e = build_expansion_sequential(16, 0, 1, activation, kernel_initializer, e, r1)

    output = tf.keras.layers.Conv2D(num_classes, 1, 1, 'same', activation='softmax')(e)

    return tf.keras.Model(inputs=input_layer, outputs=output)
