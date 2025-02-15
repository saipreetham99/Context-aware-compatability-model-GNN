# file: model/metrics.py
import tensorflow as tf


def sigmoid_accuracy(preds, labels):
    """
    Accuracy for binary class model.
    :param preds: predictions (logits)
    :param labels: ground truth label
    :return: average accuracy
    """
    correct_prediction = tf.equal(
        tf.cast(preds >= 0.0, tf.int64), tf.cast(labels, tf.int64)
    )
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def softmax_confusion_matrix(preds, labels):
    """
    Computes the confusion matrix. The rows are real labels, and columns the
    predictions.
    """
    int_preds = tf.cast(preds >= 0.0, tf.int32)
    return tf.math.confusion_matrix(labels, int_preds)


def sigmoid_cross_entropy(outputs, labels):
    """computes average binary cross entropy from logits"""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=outputs, labels=tf.cast(labels, tf.float32)
    )
    return tf.reduce_mean(loss)
