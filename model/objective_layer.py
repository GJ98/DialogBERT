import tensorflow as tf
from tensorflow.keras import layers


class EncodingConverter(layers.Layer):

    def __init__(self, d_h: int):
        """encoding converter
        
        Args:
            d_h (int): hidden dim
            rate (float): dropout probability
        """
        super().__init__()

        self.w = layers.Dense(d_h)

    def call(self, enc_cntxts: tf.Tensor, mask_idxs: tf.Tensor):
        """forward propagation
        
        Args:
            enc_cntxts (tf.Tensor(bz, cntxt_len, d_h)): encoded contexts
            mask_idxs (tf.Tensor(bz)): masked indices

        Returns:
            output (tf.Tensor(bz, d_h)): predict encoded utterance
        """

        idxs = tf.stack([tf.range(mask_idxs.shape[0]), mask_idxs], axis=-1)

        mask_uttrs = tf.gather_nd(enc_cntxts, idxs)

        output = self.w(mask_uttrs)

        return output


class DORN(layers.Layer):

    def __init__(self, d_h: int):
        """DORN

        Args:
            d_h (int): hidden dim
        """
        super().__init__()

        self.w = layers.Dense(d_h)

    def call(self, enc_cntxts: tf.Tensor, pad_cntxts: tf.Tensor):
        """forward propagation
        
        Args:
            enc_cntxts (tf.Tensor(bz, cntxt_len, d_h)): encoded contexts
            pad_cntxts (tf.Tensor(bz, 1, 1, cntxt_len)): pad context
        
        Returns:
            output (tf.Tensor(bz, cntxt_len)): predict cntxts order
        """

        pad_cntxts = tf.squeeze(pad_cntxts, 1)

        # (bz x cntxt_len x d_h, bz x d_h x cntxt_len) -> bz x cntxt_len x cntxt_len
        weight = tf.matmul(self.w(enc_cntxts), enc_cntxts, transpose_b=True)

        # bz x cntxt_len x cntxt_len -> bz x cntxt_len
        x = tf.reduce_sum(weight * (1 - pad_cntxts), -1) / tf.reduce_sum((1 - pad_cntxts), axis=-1)

        output = tf.nn.softmax(x + tf.squeeze(pad_cntxts, 1) * -1e9, -1)

        return output