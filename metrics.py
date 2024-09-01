
import tensorflow as tf
from keras import backend as K

class_weights = tf.constant([0.00098611, 0.15070848, 0.45178135, 0.39652405], dtype=tf.float32)
cfce = tf.keras.losses.CategoricalFocalCrossentropy(alpha=class_weights, gamma=1.5)
cce = tf.keras.losses.CategoricalCrossentropy()

def weighted_f1_loss(y_true, y_pred):

    num_classes = y_true.get_shape().as_list()[-1]

    f1_scores = []

    for class_index in range(num_classes):
        class_true = tf.cast(y_true[:, :, :, class_index], dtype=tf.float32)
        class_pred = y_pred[:, :, :, class_index]

        class_true = tf.reshape(class_true, [-1])
        class_pred = tf.reshape(class_pred, [-1])

        f1 = 2 * (tf.reduce_sum(class_true * class_pred)+ K.epsilon()) / (tf.reduce_sum(class_true) + tf.reduce_sum(class_pred) + K.epsilon())

        weighted_f1 = f1 * class_weights[class_index]
        f1_scores.append(weighted_f1)

    average_f1 = tf.reduce_sum(f1_scores) / tf.reduce_sum(class_weights + K.epsilon())

    return 1 - average_f1

def weighted_combined_loss(y_true, y_pred):
    return 0.5 * f1_loss(y_true, y_pred) + 0.5 * cfce(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return 0.5 * f1_loss(y_true, y_pred) + 0.5 * cce(y_true, y_pred)

def f1_loss(y_true, y_pred):

    num_classes = y_true.get_shape().as_list()[-1]

    f1_scores = []

    for class_index in range(num_classes):
        class_true = tf.cast(y_true[:, :, :, class_index], dtype=tf.float32)
        class_pred = y_pred[:, :, :, class_index]

        class_true = tf.reshape(class_true, [-1])
        class_pred = tf.reshape(class_pred, [-1])

        f1 = 2 * (tf.reduce_sum(class_true * class_pred)+ K.epsilon()) / (tf.reduce_sum(class_true) + tf.reduce_sum(class_pred) + K.epsilon())

        f1_scores.append(f1)

    average_f1 = tf.reduce_sum(f1_scores) / len(f1_scores)

    return 1 - average_f1