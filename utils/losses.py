import tensorflow as tf 

from tensorflow import math
from tensorflow.keras import losses


cross_entropy = losses.SparseCategoricalCrossentropy(from_logits=True, reduction=losses.Reduction.NONE)

def custom_cross_entrpy(y_true: tf.Tensor, y_pred: tf.Tensor):
    """cross entropy

    Args:
        y_true (tf.Tensor(bz, uttr_len)): true words
        y_pred (tf.Tensor(bz, uttr_len, vocab_size)): predict words
    """

    mask = math.logical_not(math.equal(y_true, 0))
    loss_ = cross_entropy(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def custom_mean_squared_error(y_true: tf.Tensor, y_pred: tf.Tensor):
    """mean squared error
    
    Args:
        y_true (tf.Tensor(bz, d_h): true encoded uttrs
        y_pred (tf.Tensor(bz, d_h): predict encoded uttrs
    """

    return math.reduce_mean(math.reduce_sum(math.square(y_true - y_pred), axis=-1))
    
def custom_kl_divergence(y_true: tf.Tensor, y_pred: tf.Tensor):
    """kl divergence
    
    Args:
        y_true (tf.Tensor(bz, cntxt_len)): true context orders
        y_pred (tf.Tensor(bz, cntxt_len)): predict context orders
    """

    return math.reduce_mean(math.reduce_sum(y_true * math.log(y_true / (y_pred + 1e-12) + 1e-12), axis=-1))