# file: model/initializations.py
import tensorflow as tf
import numpy as np


def he_init(shape, dtype=None):
    """He/MSRA initializer."""
    return tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_in", distribution="normal"
    )(shape, dtype=dtype)


def uniform_init(shape, dtype=None):
    """Glorot & Bengio (2010) uniform initializer."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=dtype)
