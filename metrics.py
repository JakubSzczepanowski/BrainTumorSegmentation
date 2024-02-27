from keras import backend as K

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

    f1_scores = []

    for class_index in range(num_classes):
        class_true = y_true[:, :, :, class_index]
        class_pred = y_pred[:, :, :, class_index]

        class_true = K.flatten(class_true)
        class_pred = K.flatten(class_pred)

        f1 = 2 * (K.sum(class_true * class_pred)+ K.epsilon()) / (K.sum(class_true) + K.sum(class_pred) + K.epsilon())

        weighted_f1 = f1 * class_counts[class_index]
        f1_scores.append(weighted_f1)

    average_f1 = K.sum(f1_scores) / K.sum(class_counts + K.epsilon())

    return average_f1