import tensorflow as tf
import keras_cv

class InceptionModule(tf.keras.layers.Layer):

    def __init__(self, filters: int, strides: int, **kwargs):
        super().__init__(**kwargs)
        self.filters_per_path = filters // 4
        self.strides = strides

    def build(self, input_shape):

        activation = tf.keras.layers.LeakyReLU()
        kernel_initializer = 'he_normal'

        self.conv1x1 = tf.keras.layers.Conv2D(self.filters_per_path, 1, self.strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')
        
        self.sec_conv1x1 = tf.keras.layers.Conv2D(self.filters_per_path, 1, self.strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')
        self.conv3x3 = tf.keras.layers.Conv2D(self.filters_per_path, 3, 1, activation=activation, kernel_initializer=kernel_initializer, padding='same')

        self.third_conv1x1 = tf.keras.layers.Conv2D(self.filters_per_path, 1, self.strides, activation=activation, kernel_initializer=kernel_initializer, padding='same')
        self.conv5x5 = tf.keras.layers.Conv2D(self.filters_per_path, 5, 1, activation=activation, kernel_initializer=kernel_initializer, padding='same')

        self.max_pool = tf.keras.layers.MaxPooling2D(3, self.strides, 'same')
        self.last_conv1x1 = tf.keras.layers.Conv2D(self.filters_per_path, 1, 1, 'same', activation=activation, kernel_initializer=kernel_initializer)

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
    
def map_initializers(kernel_identifier: str) -> tf.keras.initializers.VarianceScaling:

    mapping = {'he_normal': tf.keras.initializers.HeNormal,
               'he_uniform': tf.keras.initializers.HeUniform,
               'glorot_normal': tf.keras.initializers.GlorotNormal,
               'glorot_uniform': tf.keras.initializers.GlorotUniform}
    
    return mapping[kernel_identifier]
    
class DepthwiseConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activations, kernel_initializers, **kwargs):

        super().__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activations = [tf.keras.activations.get(activation) if activation != 'prelu' else tf.keras.layers.PReLU(tf.initializers.constant(0.1)) for activation in activations]
        self.kernel_initializers = [map_initializers(kernel_initializer) for kernel_initializer in kernel_initializers]

    def build(self, input_shape):

        self.channels = input_shape[-1]

        depth_multiplier = self.filters // self.channels

        self.depthwise_kernels = tf.Variable(tf.concat([
            tf.Variable(
                initial_value=self.kernel_initializers[i % len(self.kernel_initializers)]()((self.kernel_size, self.kernel_size, 1, depth_multiplier)),
            )
            for i in range(self.channels)
        ], axis=-2),
            shape=(self.kernel_size, self.kernel_size, self.channels, depth_multiplier),
            trainable=True
        )

        self.bias = self.add_weight('bias', shape=(self.filters,), initializer='zeros', trainable=True)

        self.norms = [tf.keras.layers.BatchNormalization() for activation in self.activations if hasattr(activation, '__name__') and activation.__name__ in ('tanh', 'sigmoid')]
        
        super().build(input_shape)
        
    def call(self, inputs):

        x = tf.nn.depthwise_conv2d(
                inputs,
                self.depthwise_kernels,
                strides=self.strides,
                padding=self.padding
            )
        
        x = tf.nn.bias_add(x, self.bias)
        
        kernel_range = self.channels // len(self.activations)

        if self.activations is not None:
            normalization_index = 0
            output = []
            for i, activation in enumerate(self.activations):
                if (hasattr(activation, '__name__') and activation.__name__ in ('tanh', 'sigmoid')):
                    output.append(activation(self.norms[normalization_index](x[..., i*kernel_range : (i+1)*kernel_range])))
                    normalization_index += 1
                else:
                    output.append(activation(x[..., i*kernel_range : (i+1)*kernel_range]))

            x = tf.concat(output, axis=-1)
        
        return x
    
class SeparableConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activations, kernel_initializers, **kwargs):

        super().__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activations = [tf.keras.activations.get(activation) if activation != 'prelu' else tf.keras.layers.PReLU(tf.initializers.constant(0.1)) for activation in activations]
        self.kernel_initializers = [map_initializers(kernel_initializer) for kernel_initializer in kernel_initializers]

        self.norms = [tf.keras.layers.BatchNormalization() for activation in activations if activation in ('tanh', 'sigmoid')]

    def build(self, input_shape):

        self.channels = input_shape[-1]

        depth_multiplier = self.filters // self.channels

        self.depthwise_kernels = tf.Variable(tf.concat([
            tf.Variable(
                initial_value=self.kernel_initializers[i % len(self.kernel_initializers)]()((self.kernel_size, self.kernel_size, 1, depth_multiplier)),
            )
            for i in range(self.channels)
        ], axis=-2),
            shape=(self.kernel_size, self.kernel_size, self.channels, depth_multiplier),
            trainable=True
        )

        self.bias = self.add_weight('bias', shape=(self.filters,), initializer='zeros', trainable=True)
        
        self.pointwise_kernel = self.add_weight(
            name='pointwise_kernel',
            shape=(1, 1, self.channels, self.filters),
            initializer='glorot_uniform',
            trainable=True
        )

        self.norms = [tf.keras.layers.BatchNormalization() for activation in self.activations if hasattr(activation, '__name__') and activation.__name__ in ('tanh', 'sigmoid')]
        
        super().build(input_shape)
        
    def call(self, inputs):

        x = tf.nn.depthwise_conv2d(
                inputs,
                self.depthwise_kernels,
                strides=self.strides,
                padding=self.padding
            )
        
        x = tf.nn.bias_add(x, self.bias)
        
        kernel_range = self.channels // len(self.activations)

        if self.activations is not None:
            normalization_index = 0
            output = []
            for i, activation in enumerate(self.activations):
                if (hasattr(activation, '__name__') and activation.__name__ in ('tanh', 'sigmoid')):
                    output.append(activation(self.norms[normalization_index](x[..., i*kernel_range : (i+1)*kernel_range])))
                    normalization_index += 1
                else:
                    output.append(activation(x[..., i*kernel_range : (i+1)*kernel_range]))

            x = tf.concat(output, axis=-1)

        x = tf.nn.conv2d(x, self.pointwise_kernel, strides=self.strides, padding=self.padding)
        
        return x
    
def build_conv_cascade(filters: int, drop_proba: float, drop_size: int) -> tf.keras.Sequential:

    return tf.keras.Sequential([
        DepthwiseConv2D(filters, 3, (1, 1, 1, 1), 'SAME', ('elu', 'prelu', 'tanh', 'sigmoid'), ('he_normal', 'he_normal', 'glorot_normal', 'glorot_normal')),
        keras_cv.layers.DropBlock2D(drop_proba, drop_size),
        SeparableConv2D(filters, 3, (1, 1, 1, 1), 'SAME', ('elu', 'prelu', 'tanh', 'sigmoid'), ('he_normal', 'he_normal', 'glorot_normal', 'glorot_normal'))
    ])
    
def build_reduction_sequential(filters: int, drop_proba: float, drop_size: int, prev_node: tf.keras.layers.Layer) -> tuple[tf.keras.Sequential, InceptionModule]:

    seq = build_conv_cascade(filters, drop_proba, drop_size)(prev_node)
    inception = InceptionModule(filters, 2)(seq)

    return seq, tf.keras.layers.BatchNormalization()(inception)

def build_expansion_sequential(filters: int, drop_proba: float, drop_size: int, prev_node: tf.keras.layers.Layer, concat_with: tf.keras.Sequential) -> tf.keras.Sequential:

    activation = tf.keras.layers.LeakyReLU()
    kernel_initializer = 'he_normal'

    e = tf.keras.layers.Conv2DTranspose(filters, 2, 2, 'same', activation=activation, kernel_initializer=kernel_initializer)(prev_node)
    e = tf.keras.layers.Concatenate()([e, concat_with])

    return tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters, 3, 1, 'same', activation=activation, kernel_initializer=kernel_initializer),
        tf.keras.layers.BatchNormalization(),
        keras_cv.layers.DropBlock2D(drop_proba, drop_size),
        InceptionModule(filters, 1),
        tf.keras.layers.BatchNormalization()
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
