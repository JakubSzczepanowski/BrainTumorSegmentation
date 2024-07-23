import tensorflow as tf
import keras_cv
import numpy as np
import pywt

from dataloader import X_DTYPE

def residual_block(x, filters, drop_proba, drop_size):
    skip = x
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = keras_cv.layers.DropBlock2D(drop_proba, drop_size)(x)
    x = tf.keras.layers.LeakyReLU(0.01)(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = keras_cv.layers.DropBlock2D(drop_proba, drop_size)(x)
    
    skip = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(skip)
    skip = tf.keras.layers.BatchNormalization()(skip)
    
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(0.01)(x)
    return x

def encoder_block(x, filters, drop_proba, drop_size):
    x = residual_block(x, filters, drop_proba, drop_size)
    p = tf.keras.layers.MaxPooling2D()(x)
    return x, p

def decoder_block(x, skip, filters, drop_proba, drop_size):
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = residual_block(x, filters, drop_proba, drop_size)
    return x

def build_resunet(input_shape, num_classes):
    inputs = tf.keras.layers.Input(input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, 32, 0, 1)
    s2, p2 = encoder_block(p1, 64, 0.1, 3)
    s3, p3 = encoder_block(p2, 128, 0.2, 5)
    s4, p4 = encoder_block(p3, 256, 0.3, 7)
    
    # Bottleneck
    b = residual_block(p4, 512, 0.3, 8)
    
    # Decoder
    d4 = decoder_block(b, s4, 256, 0.3, 7)
    d3 = decoder_block(d4, s3, 128, 0.2, 5)
    d2 = decoder_block(d3, s2, 64, 0.1, 3)
    d1 = decoder_block(d2, s1, 32, 0, 1)
    
    outputs = tf.keras.layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(d1)
    
    model = tf.keras.models.Model(inputs, outputs)
    return model

def inception_res_block(x, filters):

    filters_per_path = filters // 3
    skip = x

    first_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    first_path = tf.keras.layers.BatchNormalization()(first_path)
    first_path = tf.keras.layers.LeakyReLU(0.01)(first_path)

    sec_path = tf.keras.layers.MaxPooling2D(pool_size=2, strides=1, padding='same')(x)
    sec_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(sec_path)
    sec_path = tf.keras.layers.BatchNormalization()(sec_path)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)

    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=(1, 3), padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.BatchNormalization()(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=(3, 1), padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.BatchNormalization()(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)

    last_layer = tf.keras.layers.Concatenate()([first_path, sec_path, third_path])
    last_layer = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(last_layer)
    last_layer = tf.keras.layers.BatchNormalization()(last_layer)

    skip = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(skip)
    skip = tf.keras.layers.BatchNormalization()(skip)
    
    last_layer = tf.keras.layers.Add()([last_layer, skip])
    last_layer = tf.keras.layers.LeakyReLU(0.01)(last_layer)

    return last_layer

def down_sampling_block(x, filters, drop_proba, drop_size):

    filters_per_path = filters // 3

    first_path = WaveletPooling2D()(x)

    sec_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)
    sec_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(sec_path)
    sec_path = keras_cv.layers.DropBlock2D(drop_proba, drop_size)(sec_path)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)

    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=3, padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(third_path)
    third_path = keras_cv.layers.DropBlock2D(drop_proba, drop_size)(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)

    last_layer = tf.keras.layers.Concatenate()([first_path, sec_path, third_path])
    last_layer = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(last_layer)
    last_layer = tf.keras.layers.BatchNormalization()(last_layer)
    last_layer = tf.keras.layers.LeakyReLU(0.01)(last_layer)

    return last_layer

def up_sampling_block(x, filters, drop_proba, drop_size):

    filters_per_path = filters // 3

    first_path = tf.keras.layers.UpSampling2D()(x)

    sec_path = tf.keras.layers.Conv2DTranspose(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)
    sec_path = tf.keras.layers.Conv2DTranspose(filters_per_path, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(sec_path)
    sec_path = keras_cv.layers.DropBlock2D(drop_proba, drop_size)(sec_path)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)

    third_path = tf.keras.layers.Conv2DTranspose(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2DTranspose(filters_per_path, kernel_size=3, padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2DTranspose(filters_per_path, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(third_path)
    third_path = keras_cv.layers.DropBlock2D(drop_proba, drop_size)(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)

    last_layer = tf.keras.layers.Concatenate()([first_path, sec_path, third_path])
    last_layer = tf.keras.layers.Conv2DTranspose(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(last_layer)
    last_layer = tf.keras.layers.BatchNormalization()(last_layer)
    last_layer = tf.keras.layers.LeakyReLU(0.01)(last_layer)

    return last_layer

def bottleneck_layer(x, filters):
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.01)(x)
    return x

def dense_inception_block(x, filters, num_layers=3):
    inputs = [x]

    for i in range(num_layers):
        x = inception_res_block(x, filters)
        x = bottleneck_layer(x, filters)
        inputs.append(x)
        x = tf.keras.layers.Concatenate()(inputs)
    
    return x

def diu_encoder_block(x, filters, drop_proba, drop_size, block):
    x = block(x, filters)
    x = CBAM()(x)
    p = down_sampling_block(x, filters, drop_proba, drop_size)
    p = CBAM()(p)
    return x, p

def diu_decoder_block(x, skip, filters, drop_proba, drop_size, block):
    x = up_sampling_block(x, filters, drop_proba, drop_size)
    x = CBAM()(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = block(x, filters)
    x = CBAM()(x)
    return x

def build_diunet(input_shape, num_classes):
    inputs = tf.keras.layers.Input(input_shape)
    
    # Encoder
    s1, p1 = diu_encoder_block(inputs, 32, 0, 1, inception_res_block)
    s2, p2 = diu_encoder_block(p1, 64, 0.1, 3, inception_res_block)
    s3, p3 = diu_encoder_block(p2, 128, 0.2, 5, inception_res_block)
    s4, p4 = diu_encoder_block(p3, 256, 0.3, 7, dense_inception_block)
    
    # Bottleneck
    b = dense_inception_block(p4, 512)
    
    # Decoder
    d4 = diu_decoder_block(b, s4, 256, 0.3, 7, dense_inception_block)
    d3 = diu_decoder_block(d4, s3, 128, 0.2, 5, inception_res_block)
    d2 = diu_decoder_block(d3, s2, 64, 0.1, 3, inception_res_block)
    d1 = diu_decoder_block(d2, s1, 32, 0, 1, inception_res_block)
    
    outputs = tf.keras.layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(d1)
    
    model = tf.keras.models.Model(inputs, outputs)
    return model


class WaveletPooling2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WaveletPooling2D, self).__init__(**kwargs)

    def call(self, inputs):
        def haar_wavelet_transform(image):
            # image: [height, width, channels]
            transformed_channels = []
            for c in range(image.shape[-1]):
                coeffs2 = pywt.dwt2(image[:, :, c], 'haar')
                LL, (LH, HL, HH) = coeffs2
                transformed_channels.append(LL)
            return np.stack(transformed_channels, axis=-1).astype(X_DTYPE)

        def pywt_transform(inputs_numpy):
            transformed_images = [haar_wavelet_transform(image) for image in inputs_numpy]
            return np.stack(transformed_images, axis=0)

        output = tf.py_function(func=pywt_transform, inp=[inputs], Tout=X_DTYPE)

        batch_size, height, width, channels = inputs.shape
        output.set_shape((batch_size, height // 2, width // 2, channels))
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, input_shape[3])


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_layer_one = tf.keras.layers.Dense(channels // self.reduction_ratio,
                                                      activation='relu',
                                                      kernel_initializer='he_normal')
        self.shared_layer_two = tf.keras.layers.Dense(channels,
                                                      kernel_initializer='glorot_normal')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)
        
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)
        
        attention = tf.sigmoid(avg_pool + max_pool)
        return inputs * attention

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.convolution = tf.keras.layers.Conv2D(filters=1,
                                                  kernel_size=self.kernel_size,
                                                  padding='same',
                                                  kernel_initializer='glorot_normal',
                                                  activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        combined = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.convolution(combined)
        return inputs * attention

class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def call(self, inputs):
        channel_attention = self.channel_attention(inputs)
        spatial_attention = self.spatial_attention(channel_attention)
        return spatial_attention

    
def build_conv_cascade(filters: int, drop_proba: float, drop_size: int) -> tf.keras.Sequential:

    kernel_initializer = 'he_normal'

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, 3, kernel_initializer=kernel_initializer, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        keras_cv.layers.DropBlock2D(drop_proba, drop_size),
        tf.keras.layers.Conv2D(filters, 3, kernel_initializer=kernel_initializer, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
    ])
    
def build_reduction_sequential(filters: int, drop_proba: float, drop_size: int, prev_node: tf.keras.layers.Layer):

    seq = build_conv_cascade(filters, drop_proba, drop_size)(prev_node)
    pooling = tf.keras.layers.MaxPool2D()(seq)

    return seq, pooling

def build_expansion_sequential(filters: int, drop_proba: float, drop_size: int, prev_node: tf.keras.layers.Layer, concat_with: tf.keras.Sequential) -> tf.keras.Sequential:

    kernel_initializer = 'he_normal'

    e = tf.keras.layers.Conv2DTranspose(filters, 2, 2, 'same', kernel_initializer=kernel_initializer)(prev_node)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.LeakyReLU()(e)
    concat_with = CBAM()(concat_with)
    e = tf.keras.layers.Concatenate()([e, concat_with])

    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, 3, 1, 'same', kernel_initializer=kernel_initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        keras_cv.layers.DropBlock2D(drop_proba, drop_size),
        tf.keras.layers.Conv2D(filters, 3, 1, 'same', kernel_initializer=kernel_initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
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
