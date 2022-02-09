import tensorflow as tf
from tensorflow.keras import layers

from model.sub_layer import MultiHeadAttention, FeedForward
from model.embed import positional_encoding


class EncoderLayer(layers.Layer):

    def __init__(self, 
                 d_h: int, 
                 head: int, 
                 d_ff: int, 
                 rate: float):
        """tranformer encoder layer
        
        Args:
            d_h (int): attn hidden dim
            head (int): parallel attention layers
            d_ff (int): FFN hidden dim (4 * d_h)
            rate (float): dropout probability
        """
        super().__init__()

        self.attn = MultiHeadAttention(d_h=d_h,
                                       head=head)
        self.ffn = FeedForward(d_h=d_h,
                               d_ff=d_ff)
                            
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = layers.Dropout(rate=rate)
        self.dropout_2 = layers.Dropout(rate=rate)

    def call(self, x: tf.Tensor, mask: tf.Tensor, training: bool):
        """forward propagation
        
        Args:
            x (tf.Tensor(bz, len, d_h)): input
            mask (tf.Tensor(bz, 1, 1, len)): pad mask
            training (bool): train or not
            
        Returns:
            output (tf.Tensor(bz, len, d_h)): output
        """

        # Multi-Head Attention
        attn = self.attn(x, x, x, mask)
        attn = self.dropout_1(attn, training=training)
        attn = self.norm_1(x + attn)

        # Feed Forward
        ffn = self.ffn(attn)
        ffn = self.dropout_2(ffn, training=training)
        output = self.norm_2(ffn + attn)

        return output


class DecoderLayer(layers.Layer):

    def __init__(self, 
                 d_h: int, 
                 head: int, 
                 d_ff: int, 
                 rate: float):
        """transformer decoder layer

        Args:
            d_h (int): attn hidden dim
            head (int): parallel attention layers
            d_ff (int): FFN hidden dim (4 * d_h)
            rate (float): dropout probability
        """
        super().__init__()

        self.attn_1 = MultiHeadAttention(d_h=d_h,
                                         head=head)
        self.attn_2 = MultiHeadAttention(d_h=d_h,
                                         head=head)

        self.ffn = FeedForward(d_h=d_h,
                               d_ff=d_ff)

        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = layers.Dropout(rate=rate)
        self.dropout_2 = layers.Dropout(rate=rate)
        self.dropout_3 = layers.Dropout(rate=rate)

    def call(self, 
             enc: tf.Tensor, 
             dec: tf.Tensor, 
             tri_mask: tf.Tensor,
             pad_mask: tf.Tensor,
             training: bool):
        """forward propagation

        Args:
            enc (tf.Tensor(bz, enc_len, d_h)): encoder output
            dec (tf.Tensor(bz, dec_len, d_h)): prev decoder output
            tri_mask (tf.Tensor(bz, 1, dec_len, dec_len)): look ahead mask
            pad_mask (tf.Tensor(bz, 1, 1, enc_len)): pad mask
            training (bool): train or not

        Returns:
            output (tf.Tensor(bz, dec_len, d_h)): output
        """

        # Masked Multi-Head Attention
        attn_1 = self.attn_1(dec, dec, dec, tri_mask)
        attn_1 = self.dropout_1(attn_1, training=training)
        attn_1 = self.norm_1(attn_1 + dec)

        # Multi-Head Attention
        attn_2 = self.attn_2(enc, enc, attn_1, pad_mask)
        attn_2 = self.dropout_1(attn_2, training=training)
        attn_2 = self.norm_1(attn_2 + attn_1)

        # Feed Forward
        ffn = self.ffn(attn_2)
        ffn = self.dropout_2(ffn, training=training)
        output = self.norm_2(ffn + attn_2)

        return output
    

class ContextEncoder(layers.Layer):

    def __init__(self, 
                 max_len: int,
                 d_h: int, 
                 head: int, 
                 d_ff: int, 
                 n_layer: int, 
                 rate: float):
        """context encoder

        Args:
            max_len (int): max length
            d_h (int): attn hidden dim(=embedding dim)
            head (int): parallel attention layers
            d_ff (int): FFN hidden dim
            n_layer (int): number of layer
            rate (float): dropout probability
        """
        super().__init__()

        self.pos_enc = positional_encoding(max_len=max_len,
                                           d_emb=d_h)

        self.layers = [EncoderLayer(d_h=d_h,
                                    head=head,
                                    d_ff=d_ff,
                                    rate=rate) for _ in range(n_layer)]

        self.dropout = layers.Dropout(rate)
                                                    
    def call(self, cntxts: tf.Tensor, mask: tf.Tensor, training: bool):
        """forward propagation

        Args:
            cntxts (tf.Tensor(bz, len, d_h)): encoded contexts
            mask (tf.Tensor(bz, 1, 1, len)): pad mask
            training (bool): train or not

        Returns:
            output (tf.Tensor(bz, len, d_h)): output
        """ 

        #diff between train and inference
        cntxt_len = cntxts.shape[1]

        x = cntxts + self.pos_enc[:, :cntxt_len]
        x = self.dropout(x, training=training)

        for layer in self.layers:
            x = layer(x, mask, training)

        output = x

        return output


class UtteranceEncoder(layers.Layer):

    def __init__(self, 
                 vocab_size: int,
                 max_len: int,
                 d_h: int, 
                 head: int, 
                 d_ff: int, 
                 n_layer: int, 
                 rate: float):
        """utterance encoder

        Args:
            vocab_size (int): vocabulary size
            max_len (int): max length
            d_h (int): attn hidden dim(=embedding dim)
            head (int): parallel attention layers
            d_ff (int): FFN hidden dim
            n_layer (int): number of layer
            rate (float): dropout probability
        """
        super().__init__()

        self.embed = layers.Embedding(vocab_size, d_h)
        self.pos_enc = positional_encoding(max_len=max_len,
                                           d_emb=d_h)
        
        self.layers = [EncoderLayer(d_h=d_h,
                                    head=head,
                                    d_ff=d_ff,
                                    rate=rate) for _ in range(n_layer)]

        self.dropout = layers.Dropout(rate)
         
    def call(self, uttrs: tf.Tensor, mask: tf.Tensor, training: bool):
        """forward propagation

        Args:
            uttrs (tf.Tensor(bz, len)): utterances
            mask (tf.Tensor(bz, 1, 1, len)): pad mask
            training (bool): train or not

        Returns:
            output (tf.Tensor(bz, d_h)): output
        """ 
        
        #diff between train and inference
        uttr_len = uttrs.shape[1]

        x = self.embed(uttrs) + self.pos_enc[:, :uttr_len]
        x = self.dropout(x, training=training)

        for layer in self.layers:
            x = layer(x, mask, training)

        output = x

        return output[:, 0, :]
