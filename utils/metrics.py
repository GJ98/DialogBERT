import tensorflow as tf 

from tensorflow import math


def custom_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor):
    """accuracy
    
    Args:
        y_true (tf.Tensor(bz, uttr_len): true words
        y_pred (tf.Tensor(bz, uttr_len): predict words
    """

    accuracies = tf.equal(y_true, y_pred)

    mask = math.logical_not(math.equal(y_true, 0))
    accuracies = math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

def custom_ppl(y_true: tf.Tensor, y_pred: tf.Tensor):
    """perplexity

    Args:
        y_true (tf.Tensor(bz, uttr_len)): true words
        y_pred (tf.Tensor(bz, uttr_len, vocab_size)): predict words
    """

    bz, uttr_len = y_true.shape
    bz = tf.size(y_true) / uttr_len

    bz_offset = tf.range(bz, dtype=tf.int32)
    bz_offset = bz_offset[:, tf.newaxis] * tf.ones(shape=(uttr_len,), dtype=tf.int32)
    uttr_offset = tf.range(uttr_len)
    uttr_offset = tf.ones(shape=(bz, 1), dtype=tf.int32) * uttr_offset

    idxs = tf.stack([bz_offset, uttr_offset, tf.cast(y_true, tf.int32)],axis=-1)  

    y_pred = tf.nn.softmax(y_pred, axis=-1)
    true_prob = tf.gather_nd(y_pred, idxs)

    mask = tf.cast(math.not_equal(y_true, 0), dtype=tf.float32)
    true_uttr_len = math.reduce_sum(mask, axis=-1)
    true_prob = math.log(true_prob + 1e-12) * mask

    ppl = math.pow(1.0/math.exp(math.reduce_sum(true_prob, axis=-1)), 1.0/true_uttr_len)
    ppl = math.reduce_mean(ppl)
    return ppl
