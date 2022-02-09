import tensorflow as tf
from tensorflow.keras import layers, Model

from model.embed import positional_encoding
from model.layer import UtteranceEncoder, ContextEncoder, DecoderLayer
from model.objective_layer import EncodingConverter, DORN

from utils.metrics import custom_accuracy, custom_ppl
from utils.losses import custom_cross_entrpy, custom_kl_divergence, custom_mean_squared_error


class DialogBERT(Model):

    def __init__(self,
                 vocab_size: int,
                 uttr_len: int,
                 cntxt_len: int,
                 d_h: int, 
                 head: int, 
                 d_ff: int, 
                 uttr_layer: int, 
                 cntxt_layer: int,
                 rate: float):
        """DialogBERT
        
        Args:
            vocab_size (int): vocabulary size
            uttr_len (int): utterence max length
            cntxt_len (int): context max length
            d_h (int): attn hidden dim(=embedding dim)
            head (int): parallel attention layers
            d_ff (int): FFN hidden dim
            uttr_layer (int): number of layer in uttr encoder
            cntxt_layer (int): number of layer in cntxt encoder
            rate (float): dropout probability
        """
        super().__init__()

        self.d_h = d_h

        self.uttr = UtteranceEncoder(vocab_size=vocab_size,
                                     max_len=uttr_len,
                                     d_h=d_h,
                                     head=head,
                                     d_ff=d_ff,
                                     n_layer=uttr_layer,
                                     rate=rate)

        self.cntxt = ContextEncoder(max_len=cntxt_len,
                                    d_h=d_h,
                                    head=head,
                                    d_ff=d_ff,
                                    n_layer=cntxt_layer,
                                    rate=rate)

    def call(self, 
             cntxts: tf.Tensor,
             training: bool=None):
        """forward propagation

        Args:
            cntxts (tf.Tensor(bz, cntxt_len, uttr_len)): contexts
            training (bool): train or not

        Returns:
            outputs: (
                enc_uttrs (tf.Tensor(bz, cntxt_len, d_h)): encoded utterances
                enc_cntxts (tf.Tensor(bz, cntxt_len, d_h)): encoded contexts
                pad_cntxts (tf.Tensor(bz, 1, 1, cntxt_len)): pad mask context
            )
       """ 

        #diff between train and inference
        cntxt_len, uttr_len = cntxts.shape[1:]

        # pad masking
        pad_uttrs, pad_cntxts = self.get_mask(cntxts)

        # encode utterances
        # bz x cntxt_len x uttr_len -> bz*cntxt_len x uttr_len -> bz*cntxt_len x d_h
        uttrs = tf.reshape(cntxts, (-1, uttr_len))
        enc_uttrs = self.uttr(uttrs, pad_uttrs, training)

        # encode contexts
        # bz*cntxt_len x d_h -> bz x cntxt_len x d_h
        enc_uttrs = tf.reshape(enc_uttrs, (-1, cntxt_len, self.d_h))
        enc_cntxts = self.cntxt(enc_uttrs, pad_cntxts, training)

        return {
            'enc_uttrs': enc_uttrs,
            'enc_cntxts': enc_cntxts,
            'pad_cntxts': pad_cntxts
        }
    
    def get_mask(self, cntxts: tf.Tensor):
        """get cntxt pad mask

        Args:
            cntxts (tf.Tensor(bz, cntxt_len, uttr_len)): contexts

        Returns:
            pad_uttrs (tf.Tensor(bz*cntxt_len, 1, 1, uttr_len)): pad utterance
            pad_cntxts (tf.Tensor(bz, 1, 1, cntxt_len)): pad context
        """

        #diff between train and inference
        uttr_len = cntxts.shape[-1]

        pad_uttrs = tf.math.equal(cntxts, 0)
        pad_cntxts = tf.math.reduce_all(pad_uttrs, axis=-1)

        pad_uttrs = tf.cast(tf.reshape(pad_uttrs, (-1, uttr_len)), dtype=tf.float32)
        pad_cntxts = tf.cast(pad_cntxts, dtype=tf.float32)

        return pad_uttrs[:, tf.newaxis, tf.newaxis, :], pad_cntxts[:, tf.newaxis, tf.newaxis, :]


class Generator(Model):

    def __init__(self, 
                 vocab_size: int,
                 max_len: int,
                 d_h: int, 
                 head: int, 
                 d_ff: int, 
                 n_layer: int,
                 rate: float):
        """generator

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

        self.dec_layers = [DecoderLayer(d_h=d_h,
                                        head=head,
                                        d_ff=d_ff,
                                        rate=rate) for _ in range(n_layer)]

        self.proj = layers.Dense(vocab_size, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
                                                    
    def call(self, enc_cntxts: tf.Tensor, resps: tf.Tensor, pad_cntxts: tf.Tensor):
        """forward propagation

        Args:
            enc_cntxts (tf.Tensor(bz, cntxt_len, d_h)): encoder output
            resps (tf.Tensor(bz, uttr_len)): responses
            pad_cntxts (tf.Tensor(bz, 1, 1, cntxt_len)): pad context

        Returns:
            output (tf.Tensor(bz, uttr_len, vocab_size)): predict responses
        """

        #diff between train and inference
        resp_len = resps.shape[1]
        
        mask_uttrs = self.get_mask(resps)

        x = self.embed(resps) + self.pos_enc[:, :resp_len]

        for layer in self.dec_layers:
            x = layer(enc_cntxts, x, mask_uttrs, pad_cntxts)
        
        output = self.norm(self.proj(x))

        return output

    def get_mask(self, resps: tf.Tensor):
        """get uttr look ahead mask 

        Args:
            resps (tf.Tensor(bz, uttr_len)): responses

        Returns:
            mask_uttrs (tf.Tensor(bz, 1, uttr_len, uttr_len)): mask utterance
        """

        #diff between train and inference
        uttr_len = resps.shape[1]

        # pad mask: bz x uttr_len -> bz x 1 x 1 x uttr_len
        pad_mask = tf.cast(tf.math.equal(resps, 0), dtype=tf.float32)[:, tf.newaxis, tf.newaxis, :]
        # attn mask: uttr_len x uttr_len
        attn_mask = 1 - tf.linalg.band_part(tf.ones((uttr_len, uttr_len)), -1, 0)
        
        mask_uttrs = tf.maximum(pad_mask, attn_mask)

        return mask_uttrs


class DialogNug(Model):

    def __init__(self, 
                 vocab_size: int,
                 uttr_len: int,
                 cntxt_len: int,
                 d_h: int, 
                 head: int, 
                 d_ff: int, 
                 uttr_layer: int, 
                 cntxt_layer: int,
                 rate: float):
        """DialogBERT NUG only
        
        Args:
            vocab_size (int): vocabulary size
            uttr_len (int): utterence max length
            cntxt_len (int): context max length
            d_h (int): attn hidden dim(=embedding dim)
            head (int): parallel attention layers
            d_ff (int): FFN hidden dim
            uttr_layer (int): number of layer in uttr encoder
            cntxt_layer (int): number of layer in cntxt encoder
            rate (float): dropout probability
        """
        super().__init__()

        self.dialog = DialogBERT(vocab_size=vocab_size,
                                 uttr_len=uttr_len,
                                 cntxt_len=cntxt_len,
                                 d_h=d_h,
                                 head=head,
                                 d_ff=d_ff,
                                 uttr_layer=uttr_layer,
                                 cntxt_layer=cntxt_layer,
                                 rate=rate)

        self.gen = Generator(vocab_size=vocab_size,
                             max_len=uttr_len,
                             d_h=d_h,
                             head=head,
                             d_ff=d_ff,
                             n_layer=uttr_layer,
                             rate=rate)

    def call(self, inputs, training: bool):
        """forward propagation
        
        Args:
            inputs: (
                cntxts (tf.Tensor(bz, cntxt_len, uttr_len)): contexts
                resps (tf.Tensor(bz, uttr_len)): responses
            )
           training (bool): train or not

        Returns:
            resps (tf.Tensor(bz, uttr_len, vocab_size)): predict responses
        """

        # bz x cntxt_len x utters -> bz x cntxt_len x d_h
        dialog = self.dialog(inputs['cntxts'], training)

        predicts = self.gen(dialog['enc_cntxts'], inputs['resps'][:, :-1], dialog['pad_cntxts'])

        loss = custom_cross_entrpy(inputs['resps'][:, 1:], predicts)

        self.add_loss(loss)

        # Calculate metrics
        prediction = tf.cast(tf.argmax(predicts, axis=-1), dtype=tf.float32)
        acc = custom_accuracy(inputs['resps'][:, 1:], prediction)
        ppl = custom_ppl(inputs['resps'][:, 1:], predicts)
        self.add_metric(ppl, name='ppl')
        self.add_metric(acc, name='acc')

        return predicts
    

class DialogAll(Model):

    def __init__(self, 
                 vocab_size: int,
                 uttr_len: int,
                 cntxt_len: int,
                 d_h: int, 
                 head: int, 
                 d_ff: int, 
                 uttr_layer: int, 
                 cntxt_layer: int,
                 rate: float):
        """DialogBERT NUG, MUR and ORNS
        
        Args:
            vocab_size (int): vocabulary size
            uttr_len (int): utterence max length
            cntxt_len (int): context max length
            d_h (int): attn hidden dim(=embedding dim)
            head (int): parallel attention layers
            d_ff (int): FFN hidden dim
            uttr_layer (int): number of layer in uttr encoder
            cntxt_layer (int): number of layer in cntxt encoder
            rate (float): dropout probability
        """
        super().__init__()

        self.dialog = DialogBERT(vocab_size=vocab_size,
                                 uttr_len=uttr_len,
                                 cntxt_len=cntxt_len,
                                 d_h=d_h,
                                 head=head,
                                 d_ff=d_ff,
                                 uttr_layer=uttr_layer,
                                 cntxt_layer=cntxt_layer,
                                 rate=rate)

        # NUG(Next Utterance Generation)
        self.gen = Generator(vocab_size=vocab_size,
                             max_len=uttr_len,
                             d_h=d_h,
                             head=head,
                             d_ff=d_ff,
                             n_layer=uttr_layer,
                             rate=rate)

        # MUR(Masked Utterance Regression)
        self.enc_conv = EncodingConverter(d_h)

        # DUOR(Distributed Utterance Order Ranking)
        self.dorn = DORN(d_h)

    def call(self, inputs, training: bool):
        """forward propagation
        
        Args:
            inputs: (
                cntxts (tf.Tensor(bz, cntxt_len, uttr_len)): contexts
                mask_uttrs (tf.Tensor(bz, uttr_len)): masked utterances
                mask_idxs (tf.Tensor(bz)=None): masked utterances idx
                shufs (tf.Tensor(bz, cntxt_len)): shuffling indices
                resps (tf.Tensor(bz, uttr_len)): responses
            )
           training (bool): train or not
        """

        #diff between train and inference
        bz, cntxt_len = inputs['cntxts'].shape[:2]

        # bz x cntxt_len x utters -> bz x cntxt_len x d_h
        dialog = self.dialog(inputs['cntxts'], training)

        offset = tf.range(bz)

        # pad masking
        pad_mask_uttrs = self.get_mask(inputs['mask_uttrs'])
        enc_mask_uttrs = self.dialog.uttr(inputs['mask_uttrs'], pad_mask_uttrs, training)

        # Masking
        idxs = tf.stack([offset, inputs['mask_idxs']],axis=-1)  
        origin_uttrs = tf.gather_nd(dialog['enc_uttrs'], idxs)
        mask_uttrs = tf.tensor_scatter_nd_update(dialog['enc_uttrs'],
                                                 indices=idxs,
                                                 updates=enc_mask_uttrs)

        # Shuffling
        offset = offset[:, tf.newaxis] * tf.ones(shape=(cntxt_len,), dtype=tf.int32)
        idxs = tf.stack([offset, inputs['shufs']], axis=-1)
        shuf_uttrs = tf.gather_nd(dialog['enc_uttrs'], idxs)

        # bz x cntxt_len x d_h -> bz x cntxt_len x d_h
        mask_cntxts = self.dialog.cntxt(mask_uttrs, dialog['pad_cntxts'], training)
        shuf_cntxts = self.dialog.cntxt(shuf_uttrs, dialog['pad_cntxts'], training)

        # NUG: (bz x cntxt_len x d_h, bz x uttr_len) -> bz x uttr_len x vocab_size
        predicts_nug = self.gen(dialog['enc_cntxts'], inputs['resps'][:, :-1], dialog['pad_cntxts'])

        # MUR: bz x cntxt_len x d_h -> (bz x cntxt_len x d_h, bz) -> bz x d_h
        predicts_mur = self.enc_conv(mask_cntxts, inputs['mask_idxs'])
        
        # DUOR bz x cntxt_len x d_h -> bz x cntxt_len
        predicts_orns = self.dorn(shuf_cntxts, dialog['pad_cntxts'])

        # Calculate losses
        label_orns = tf.stop_gradient(tf.nn.softmax( \
            tf.cast(inputs['shufs'], tf.float32) + (tf.squeeze(dialog['pad_cntxts'], (1, 2)) * -1e9), -1))
        loss_mur = custom_mean_squared_error(origin_uttrs, predicts_mur)
        loss_orns = custom_kl_divergence(label_orns, predicts_orns)
        loss_nug = custom_cross_entrpy(inputs['resps'][:, 1:], predicts_nug)

        self.add_loss(loss_orns + loss_mur + loss_nug)
        self.add_metric(loss_nug, name='nug')
        self.add_metric(loss_mur, name='mur')
        self.add_metric(loss_orns, name='orns')

        # Calculate metrics
        prediction = tf.cast(tf.argmax(predicts_nug, axis=-1), dtype=tf.float32)
        acc = custom_accuracy(inputs['resps'][:, 1:], prediction)
        ppl = custom_ppl(inputs['resps'][:, 1:], predicts_nug)
        self.add_metric(ppl, name='ppl')
        self.add_metric(acc, name='acc')


    def get_mask(self, uttrs: tf.Tensor):
        """get uttr pad mask

        Args:
            uttrs (tf.Tensor(bz, uttr_len)): utterances
            
        Returns:
            pad_uttrs (tf.Tensor(bz, 1, 1, uttr_len)): pad utterance
        """

        pad_uttrs = tf.cast(tf.math.equal(uttrs, 0), tf.float32)

        return pad_uttrs[:, tf.newaxis, tf.newaxis, :]