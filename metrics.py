from keras import backend as K
import tensorflow as tf
import numpy as np

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def weighted_f1(y_true, y_pred):

    num_classes = K.int_shape(y_true)[-1]
    class_counts = K.sum(y_true, axis=[0, 1, 2])
    class_weights = tf.math.reciprocal(class_counts)

    f1_scores = []

    for class_index in range(num_classes):
        class_true = y_true[:, :, :, class_index]
        class_pred = y_pred[:, :, :, class_index]

        class_true = K.flatten(class_true)
        class_pred = K.flatten(class_pred)

        f1 = 2 * (K.sum(class_true * class_pred)+ K.epsilon()) / (K.sum(class_true) + K.sum(class_pred) + K.epsilon())

        weighted_f1 = f1 * class_weights[class_index]
        f1_scores.append(weighted_f1)

    average_f1 = K.sum(f1_scores) / K.sum(class_weights + K.epsilon())

    return average_f1

# def weighted_categorical_crossentropy(y_true, y_pred):

#     class_counts = K.sum(y_true, axis=[0, 1, 2])
#     class_weights = tf.math.reciprocal(class_counts)

#     class_weights = class_weights/K.sum(class_weights, keepdims=True)
    

#     return tf.keras.losses.CategoricalCrossentropy()(tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1]), tf.reshape(class_weights, (1, 4)))

weights = np.array([0.00109971, 0.1458139, 0.5603247, 0.29276177]).reshape((1,1,1,4))
# kWeights = K.constant(weights)
class_weights = tf.constant([0.00109971, 0.1458139, 0.5603247, 0.29276177], dtype=tf.float32)

class WeightedF1(tf.keras.losses.Loss):

    def __init__(self, class_weights = class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def call(self, y_true, y_pred):
        num_classes = K.shape(y_true)[-1]

        f1_scores = []

        for class_index in range(num_classes):
            class_true = K.cast(y_true[:, :, :, class_index], dtype=tf.float32)
            class_pred = y_pred[:, :, :, class_index]

            class_true = K.flatten(class_true)
            class_pred = K.flatten(class_pred)

            f1 = 2 * (K.sum(class_true * class_pred)+ K.epsilon()) / (K.sum(class_true) + K.sum(class_pred) + K.epsilon())

            weighted_f1 = f1 * self.class_weights[class_index]
            f1_scores.append(weighted_f1)

        f1_scores = tf.stack(f1_scores)
        average_f1 = K.sum(f1_scores) / K.sum(self.class_weights + K.epsilon())

        return 1 - average_f1

def weighted_f1_loss(y_true, y_pred):

    num_classes = tf.shape(y_true)[-1]

    f1_scores = []

    for class_index in range(num_classes):
        class_true = tf.cast(y_true[:, :, :, class_index], dtype=tf.float32)
        class_pred = y_pred[:, :, :, class_index]

        class_true = tf.flatten(class_true)
        class_pred = tf.flatten(class_pred)

        f1 = 2 * (tf.reduce_sum(class_true * class_pred)+ K.epsilon()) / (tf.reduce_sum(class_true) + tf.reduce_sum(class_pred) + K.epsilon())

        weighted_f1 = f1 * class_weights[class_index]
        f1_scores.append(weighted_f1)

    average_f1 = tf.reduce_sum(f1_scores) / tf.reduce_sum(class_weights + K.epsilon())

    return 1 - average_f1

# def weighted_categorical_crossentropy(y_true, y_pred):
#     yWeights = kWeights * y_pred         #shape (batch, 128, 128, 4)
#     yWeights = K.sum(yWeights, axis=-1)  #shape (batch, 128, 128)  

#     loss = K.categorical_crossentropy(y_true, y_pred) #shape (batch, 128, 128)
#     wLoss = yWeights * loss

#     return K.sum(wLoss, axis=(1,2))