import tensorflow as tf
from tensorflow.keras import Model


class GreedyChatbot(tf.Module):

    def __init__(self, 
                 dialog: Model,
                 gen: Model):
        """DialogBERT Greedy search Chatbot
        
        Args:
            dialog (tf.keras.Model): dialogBERT
            gen (tf.keras.Model): generator
        """
        super().__init__()

        self.dialog = dialog
        self.gen = gen  

    @tf.function(input_signature=[tf.TensorSpec(shape=[5, 30], dtype=tf.float32)])
    def __call__(self, inputs: tf.Tensor):
        """forward propagation
        
        Args:
            inputs (tf.Tensor(cntxt_len, uttr_len)): input

        Returns:
            output (tf.Tensor): response
        """

        # 0. initialize resp
        resp = tf.TensorArray(dtype=tf.float32, size=30)
        resp = resp.write(0, 5)

        # 1. encode cntxt
        dialog = self.dialog(inputs[tf.newaxis, ...])

        # 2. predict response
        for i in tf.range(29):
            predict = self.gen(dialog['enc_cntxts'], resp.stack()[tf.newaxis, :-1], dialog['pad_cntxts'])
            last_word = tf.cast(tf.argmax(predict[0, i], axis=-1), dtype=tf.float32)
            resp = resp.write(i + 1, last_word)
            if last_word == tf.constant(4.0): break
        
        return tf.cast(resp.stack(), dtype=tf.int32)


class BeamChatbot(tf.Module):

    def __init__(self, 
                 dialog: Model,
                 gen: Model,
                 beam_size: int,
                 vocab_size: int):
        """DialogBERT Beam search Chatbot
        
        Args:
            dialog (tf.keras.Model): dialogBERT
            gen (tf.keras.Model): generator
            beam_size (int): size of beam search
            vocab_size (int): vocabulary size
        """
        super().__init__()

        self.dialog = dialog
        self.gen = gen  
        self.beam_size = beam_size
        self.vocab_size = vocab_size

    @tf.function(input_signature=[tf.TensorSpec(shape=[5, 30], dtype=tf.float32)])
    def __call__(self, inputs: tf.Tensor):
        """forward propagation
        
        Args:
            inputs (tf.Tensor(cntxt_len, uttr_len)):

        Returns:
            output (tf.Tensor): response
        """

        cand = tf.TensorArray(dtype=tf.float32, size=self.beam_size, element_shape=tf.TensorShape([30]))
        cand_prob = tf.TensorArray(dtype=tf.float32, size=self.beam_size)
        word_prob = tf.TensorArray(dtype=tf.float32, size=self.beam_size, element_shape=tf.TensorShape([self.vocab_size]))
        is_end = tf.TensorArray(dtype=tf.int32, size=self.beam_size)

        # 0. initialize tensor array
        cand = cand.unstack(tf.ones(shape=(5, 1)) * tf.concat([tf.constant([5.0]), tf.zeros(29)], axis=-1))
        cand_prob = cand_prob.unstack(tf.ones(self.beam_size, dtype=tf.float32))
        is_end = is_end.unstack(tf.range(self.beam_size))

        # 1. encode cntxt
        dialog = self.dialog(inputs[tf.newaxis, ...])
        enc_cntxts = tf.tile(dialog['enc_cntxts'], tf.constant([self.beam_size, 1, 1]))
        pad_cntxts = tf.tile(dialog['pad_cntxts'], tf.constant([self.beam_size, 1, 1, 1]))

        # 2. predict response
        for word_i in tf.range(29):
            # Is all hypo end
            if tf.reduce_all(tf.math.equal(is_end.stack(), -1)): break

            # 2-1. run generator
            predict = self.gen(enc_cntxts, cand.stack()[:, :-1], pad_cntxts)

            # 2-2. calculate probability
            for hypo in is_end.stack():
                if hypo == tf.constant(-1): continue
                word_prob = word_prob.write(hypo, tf.nn.softmax(predict[hypo, word_i], axis=-1) * cand_prob.read(hypo))

            # 2-3. select hypotheses
            if word_i == 0: top_k = tf.math.top_k(word_prob.read(0), k=self.beam_size)
            else: top_k = top_k = tf.math.top_k(word_prob.concat(), k=self.beam_size) 

            # 2-4. update hypotheses
            i, before_cand = tf.constant(0), cand.stack()
            for hypo in is_end.stack():
                if hypo == tf.constant(-1): continue
                row, col = top_k.indices[i] // self.vocab_size, top_k.indices[i] % self.vocab_size
                after_cand = tf.concat([before_cand[row, :word_i+1], [col], before_cand[row, word_i+2:]], axis=-1)
                cand, cand_prob = cand.write(hypo, after_cand), cand_prob.write(hypo, top_k.values[i])
                i += 1
                # 2-4-1. Is hypo end
                if col == tf.constant(4): 
                    is_end = is_end.write(hypo, -1)
                    word_prob = word_prob.write(hypo, tf.zeros(shape=(self.vocab_size)))

        # 2-5. select response
        max_idx = tf.argmax(cand_prob.stack(), output_type=tf.int32)
        
        return tf.concat([tf.expand_dims(cand.read(max_idx), axis=0), cand.stack()], axis=0)


class CheatAllChatbot(tf.Module):

    def __init__(self, 
                 dialog: Model,
                 gen: Model):
        """DialogBERT Chatbot which give response
        
        Args:
            dialog (tf.keras.Model): dialogBERT
            gen (tf.keras.Model): generator
        """
        super().__init__()

        self.dialog = dialog
        self.gen = gen  

    @tf.function(input_signature=[tf.TensorSpec(shape=[6, 30], dtype=tf.float32)])
    def __call__(self, inputs: tf.Tensor):
        """forward propagation
        
        Args:
            inputs (tf.Tensor(cntxt_len, uttr_len)):

        Returns:
            output (tf.Tensor): response
        """

        # 1. encode cntxt
        dialog = self.dialog(inputs[tf.newaxis, :-1])

        # 2. predict response
        predict = self.gen(dialog['enc_cntxts'], inputs[tf.newaxis, -1], dialog['pad_cntxts'])

        return tf.cast(tf.argmax(predict[0], axis=-1), dtype=tf.int32)


class CheatFirstChatbot(tf.Module):

    def __init__(self, 
                 dialog: Model,
                 gen: Model):
        """DialogBERT Chatbot which give first true response word
        
        Args:
            dialog (tf.keras.Model): dialogBERT
            gen (tf.keras.Model): generator
        """
        super().__init__()

        self.dialog = dialog
        self.gen = gen  

    @tf.function(input_signature=[tf.TensorSpec(shape=[6, 30], dtype=tf.float32)])
    def __call__(self, inputs: tf.Tensor):
        """forward propagation
        
        Args:
            inputs (tf.Tensor(cntxt_len, uttr_len)):

        Returns:
            output (tf.Tensor): response
        """

        # 0. initialize resp
        resp = tf.TensorArray(dtype=tf.float32, size=30)
        resp = resp.write(0, 5)
        resp = resp.write(1, inputs[-1, 1])

        # 1. encode cntxt
        dialog = self.dialog(inputs[tf.newaxis, :-1])

        # 2. predict response
        for i in tf.range(1, 29):
            predict = self.gen(dialog['enc_cntxts'], resp.stack()[tf.newaxis, :-1], dialog['pad_cntxts'])
            last_word = tf.cast(tf.argmax(predict[0, i], axis=-1), dtype=tf.float32)
            resp = resp.write(i + 1, last_word)
            if last_word == tf.constant(4.0): break
        
        return tf.cast(resp.stack(), dtype=tf.int32)


