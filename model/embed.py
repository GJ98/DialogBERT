import numpy as np
import tensorflow as tf


def positional_encoding(max_len: int, d_emb: int):
    """positional_encoding

    Args:
        max_len (int): max length
        d_emb (int): embedding dim

    Returns:
        pos_enc(tf.Tensor(1, max_len, d_emb)): positional encoding
    """

    # default value of requires_grad = false
    pos_enc = np.zeros(shape=(max_len, d_emb))

    pos = np.arange(max_len)[:, np.newaxis]

    _2i = np.arange(d_emb, step=2)

    pos_enc[:, 0::2] = np.sin(pos / 10000 ** (_2i / d_emb))
    pos_enc[:, 1::2] = np.cos(pos / 10000 ** (_2i / d_emb))

    return tf.cast(pos_enc[tf.newaxis, ...], dtype=tf.float32)