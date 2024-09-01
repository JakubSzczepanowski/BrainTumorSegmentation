import tensorflow as tf
import tensorflow_wavelets.Layers.DWT as DWT

from keras_cv.src.backend import config

if config.keras_3():
    print('keras_3')
    from keras.ops import *  # noqa: F403, F401
    from keras.preprocessing.image import smart_resize  # noqa: F403, F401

    from keras_cv.src.backend import keras

    name_scope = keras.name_scope
else:
    try:
        print('else')
        from keras.src.ops import *  # noqa: F403, F401
        from keras.src.utils.image_utils import smart_resize  # noqa: F403, F401
    # Import error means Keras isn't installed, or is Keras 2.
    except ImportError:
        from keras_core.src.backend import vectorized_map  # noqa: F403, F401
        from keras_core.src.ops import *  # noqa: F403, F401
        from keras_core.src.utils.image_utils import (  # noqa: F403, F401
            smart_resize,
        )
    if config.backend() == "tensorflow":
        print('tensorflow')
        from keras_cv.src.backend.tf_ops import *  # noqa: F403, F401

from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.backend import random
from keras_cv.src.utils import conv_utils

class DropBlock2D(keras.layers.Layer):

    def __init__(
        self,
        rate,
        block_size,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"rate must be a number between 0 and 1. " f"Received: {rate}"
            )

        self._rate = rate
        (
            self._dropblock_height,
            self._dropblock_width,
        ) = conv_utils.normalize_tuple(
            value=block_size, n=2, name="block_size", allow_zero=False
        )
        self.seed = seed
        self._random_generator = random.SeedGenerator(self.seed)

    def call(self, x, training=None):
        if tf.is_symbolic_tensor(x) or not training or self._rate == 0.0:
            return x
        _, height, width, _ = ops.split(ops.shape(x), 4)

        # Unnest scalar values
        height = ops.squeeze(height)
        width = ops.squeeze(width)

        dropblock_height = ops.minimum(self._dropblock_height, height)
        dropblock_width = ops.minimum(self._dropblock_width, width)

        gamma = (
            self._rate
            * ops.cast(width * height, dtype="float32")
            / ops.cast(dropblock_height * dropblock_width, dtype="float32")
            / ops.cast(
                (width - self._dropblock_width + 1)
                * (height - self._dropblock_height + 1),
                "float32",
            )
        )

        # Forces the block to be inside the feature map.
        w_i, h_i = ops.meshgrid(ops.arange(width), ops.arange(height))
        valid_block = ops.logical_and(
            ops.logical_and(
                w_i >= int(dropblock_width // 2),
                w_i < width - (dropblock_width - 1) // 2,
            ),
            ops.logical_and(
                h_i >= int(dropblock_height // 2),
                h_i < width - (dropblock_height - 1) // 2,
            ),
        )

        valid_block = ops.reshape(valid_block, [1, height, width, 1])

        random_noise = random.uniform(
            ops.shape(x), seed=self._random_generator, dtype="float32"
        )
        valid_block = ops.cast(valid_block, dtype="float32")
        seed_keep_rate = ops.cast(1 - gamma, dtype="float32")
        block_pattern = (1 - valid_block + seed_keep_rate + random_noise) >= 1
        block_pattern = ops.cast(block_pattern, dtype="float32")

        window_size = [1, self._dropblock_height, self._dropblock_width, 1]

        # Double negative and max_pool is essentially min_pooling
        block_pattern = -ops.max_pool(
            -block_pattern,
            pool_size=window_size,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        # Slightly scale the values, to account for magnitude change
        percent_ones = ops.cast(ops.sum(block_pattern), "float32") / ops.cast(
            ops.size(block_pattern), "float32"
        )
        return (
            x
            / ops.cast(percent_ones, x.dtype)
            * ops.cast(block_pattern, x.dtype)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self._rate,
                "block_size": (self._dropblock_height, self._dropblock_width),
                "seed": self.seed,
            }
        )
        return config

def residual_block(x, filters, drop_proba, drop_size):
    skip = x
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = DropBlock2D(drop_proba, drop_size)(x)
    x = tf.keras.layers.LeakyReLU(0.01)(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = DropBlock2D(drop_proba, drop_size)(x)
    
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
    
    s1, p1 = encoder_block(inputs, 32, 0, 1)
    s2, p2 = encoder_block(p1, 64, 0.1, 3)
    s3, p3 = encoder_block(p2, 128, 0.2, 5)
    s4, p4 = encoder_block(p3, 256, 0.3, 7)
    
    b = residual_block(p4, 512, 0.3, 8)
    
    d4 = decoder_block(b, s4, 256, 0.3, 7)
    d3 = decoder_block(d4, s3, 128, 0.2, 5)
    d2 = decoder_block(d3, s2, 64, 0.1, 3)
    d1 = decoder_block(d2, s1, 32, 0, 1)
    
    outputs = tf.keras.layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(d1)
    
    model = tf.keras.models.Model(inputs, outputs)
    return model


class DWT_Layer(tf.keras.layers.Layer):
    def __init__(self, wavelet_name='haar', **kwargs):
        super(DWT_Layer, self).__init__(**kwargs)
        self.wavelet_name = wavelet_name

    def build(self, input_shape):
        self.dwt_layer = tf.keras.layers.Lambda(
            lambda x: DWT.DWT(wavelet_name=self.wavelet_name, concat=0)(x)
        )
        super(DWT_Layer, self).build(input_shape)

    def call(self, inputs):
        trans = self.dwt_layer(inputs)
        
        trans_LL = trans[:, :, :, :inputs.shape[-1]]

        return trans_LL

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, input_shape[3])

def inception_res_block(x, filters, drop_proba, drop_size):

    filters_per_path = filters // 3
    skip = x

    first_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    first_path = tf.keras.layers.BatchNormalization()(first_path)
    first_path = tf.keras.layers.LeakyReLU(0.01)(first_path)
    first_path = DropBlock2D(drop_proba, drop_size)(first_path)

    sec_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)
    sec_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=3, padding='same', kernel_initializer='he_normal')(sec_path)
    sec_path = tf.keras.layers.BatchNormalization()(sec_path)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)
    sec_path = DropBlock2D(drop_proba, drop_size)(sec_path)

    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=3, padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.BatchNormalization()(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=3, padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.BatchNormalization()(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = DropBlock2D(drop_proba, drop_size)(third_path)

    last_layer = tf.keras.layers.Concatenate()([first_path, sec_path, third_path])
    last_layer = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(last_layer)
    last_layer = tf.keras.layers.BatchNormalization()(last_layer)

    skip = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(skip)
    skip = tf.keras.layers.BatchNormalization()(skip)
    
    last_layer = tf.keras.layers.Add()([last_layer, skip])
    last_layer = tf.keras.layers.LeakyReLU(0.01)(last_layer)

    return last_layer

def dense_inception_res_block(x, filters, drop_proba, drop_size):

    filters_per_path = filters // 3
    skip = x

    first_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    first_path = tf.keras.layers.BatchNormalization()(first_path)
    first_path = tf.keras.layers.LeakyReLU(0.01)(first_path)
    first_path = DropBlock2D(drop_proba, drop_size)(first_path)

    sec_path = tf.keras.layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    sec_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(sec_path)
    sec_path = tf.keras.layers.BatchNormalization()(sec_path)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)
    sec_path = DropBlock2D(drop_proba, drop_size)(sec_path)

    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=(1, 3), padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.BatchNormalization()(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=(3, 1), padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.BatchNormalization()(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = DropBlock2D(drop_proba, drop_size)(third_path)

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
    
    first_path = DWT_Layer()(x)

    sec_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)
    sec_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(sec_path)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)
    sec_path = DropBlock2D(drop_proba, drop_size)(sec_path)

    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=3, padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2D(filters_per_path, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = DropBlock2D(drop_proba, drop_size)(third_path)

    last_layer = tf.keras.layers.Concatenate()([first_path, sec_path, third_path])
    last_layer = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(last_layer)
    last_layer = tf.keras.layers.BatchNormalization()(last_layer)
    last_layer = tf.keras.layers.LeakyReLU(0.01)(last_layer)
    last_layer = DropBlock2D(drop_proba, drop_size)(last_layer)

    return last_layer

def up_sampling_block(x, filters, drop_proba, drop_size):

    filters_per_path = filters // 3
    
    first_path = tf.keras.layers.UpSampling2D()(x)

    sec_path = tf.keras.layers.Conv2DTranspose(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)
    sec_path = tf.keras.layers.Conv2DTranspose(filters_per_path, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(sec_path)
    sec_path = tf.keras.layers.LeakyReLU(0.01)(sec_path)
    sec_path = DropBlock2D(drop_proba, drop_size)(sec_path)

    third_path = tf.keras.layers.Conv2DTranspose(filters_per_path, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2DTranspose(filters_per_path, kernel_size=3, padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = tf.keras.layers.Conv2DTranspose(filters_per_path, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(third_path)
    third_path = tf.keras.layers.LeakyReLU(0.01)(third_path)
    third_path = DropBlock2D(drop_proba, drop_size)(third_path)

    last_layer = tf.keras.layers.Concatenate()([first_path, sec_path, third_path])
    last_layer = tf.keras.layers.Conv2DTranspose(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(last_layer)
    last_layer = tf.keras.layers.BatchNormalization()(last_layer)
    last_layer = tf.keras.layers.LeakyReLU(0.01)(last_layer)
    last_layer = DropBlock2D(drop_proba, drop_size)(last_layer)

    return last_layer

def bottleneck_layer(x, filters):
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.01)(x)
    return x

def dense_inception_block(x, filters, drop_proba, drop_size, num_layers=3):
    inputs = [x]

    for i in range(num_layers):
        x = dense_inception_res_block(x, filters, drop_proba, drop_size)
        x = bottleneck_layer(x, filters)
        inputs.append(x)
        x = tf.keras.layers.Concatenate()(inputs)
    
    return x

def diu_encoder_block(x, filters, drop_proba, drop_size, block):
    x = block(x, filters, drop_proba, drop_size)
    x = CBAM()(x)
    p = down_sampling_block(x, filters, drop_proba, drop_size)
    p = CBAM()(p)
    return x, p

def diu_decoder_block(x, skip, filters, drop_proba, drop_size, block):
    x = up_sampling_block(x, filters, drop_proba, drop_size)
    x = CBAM()(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = block(x, filters, drop_proba, drop_size)
    x = CBAM()(x)
    return x

def build_diunet(input_shape, num_classes):
    inputs = tf.keras.layers.Input(input_shape)
    
    s1, p1 = diu_encoder_block(inputs, 32, 0.05, 2, inception_res_block)
    s2, p2 = diu_encoder_block(p1, 64, 0.1, 3, inception_res_block)
    s3, p3 = diu_encoder_block(p2, 128, 0.15, 5, inception_res_block)
    s4, p4 = diu_encoder_block(p3, 256, 0.2, 5, dense_inception_block)
    
    b = dense_inception_block(p4, 512, 0.2, 5)
    
    d4 = diu_decoder_block(b, s4, 256, 0.2, 5, dense_inception_block)
    d3 = diu_decoder_block(d4, s3, 128, 0.15, 5, inception_res_block)
    d2 = diu_decoder_block(d3, s2, 64, 0.1, 3, inception_res_block)
    d1 = diu_decoder_block(d2, s1, 32, 0.05, 2, inception_res_block)
    
    outputs = tf.keras.layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(d1)
    
    model = tf.keras.models.Model(inputs, outputs)
    return model


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
    def __init__(self, reduction_ratio=16, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channel_attention = ChannelAttention(self.reduction_ratio)
        self.spatial_attention = SpatialAttention(self.kernel_size)

        super().build(input_shape)

    def call(self, inputs):
        channel_attention = self.channel_attention(inputs)
        spatial_attention = self.spatial_attention(channel_attention)
        return spatial_attention
