from __future__ import print_function


from hbconfig import Config
import tensorflow as tf

import transformer



class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self._set_batch_size(mode)
        self._init_placeholder(features, labels)
        self.build_graph()

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                # predictions=self.predictions,
                loss=self.loss,
                train_op=self.train_op)
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=self.loss)

    def _set_batch_size(self, mode):
        if mode == tf.estimator.ModeKeys.EVAL:
            Config.model.batch_size = Config.eval.batch_size
        else:
            Config.model.batch_size = Config.train.batch_size

    def _init_placeholder(self, features, labels):
        if type(features) == dict:
            self.encoder_inputs = features["enc_inputs"]
            self.decoder_inputs = features["dec_inputs"]

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.targets = labels

    def build_graph(self):
        graph = transformer.Graph(self.mode)
        graph.build(encoder_inputs=self.encoder_inputs,
                    decoder_inputs=self.decoder_inputs)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_loss(graph.logits)
            self._build_optimizer()
        else:
            encoder_outputs = graph.encoder_outputs
            logits = tf.reshape(graph.logits, [-1, 1, Config.data.target_vocab_size])

            # predict output with loop. [encoder_outputs, decoder_inputs (filled next token)]
            for i in range(1, Config.data.max_seq_length):
                next_token = tf.argmax(logits, axis=0)
                # set next_token to self.decoder_inputs

                decoder_emb_inp = graph.build_embed(self.decoder_inputs, encoder=False, reuse=True)
                decoder_outputs = graph.build_decoder(decoder_emb_inp, encoder_outputs, reuse=True)
                graph.build_output(decoder_outputs, reuse=True)
                logits = tf.concat((logits, tf.reshape(graph.logits, [-1, 1, Config.data.target_vocab_size])), axis=1)

            self._build_loss(logits)

    def _build_loss(self, logits):
        with tf.variable_scope('loss'):
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                self.loss = tf.losses.sparse_softmax_cross_entropy(
                            labels=self.targets,
                            logits=logits,
                            scope="cross-entropy")
            else:
                target_lengths = tf.reduce_sum(
                        tf.to_int32(tf.not_equal(self.targets, Config.data.PAD_ID)), 1)
                weight_masks = tf.sequence_mask(
                        lengths=target_lengths,
                        maxlen=Config.data.max_seq_length,
                        dtype=self.dtype, name='masks')

                self.loss = tf.contrib.seq2seq.sequence_loss(
                        logits=logits,
                        targets=self.targets,
                        weights=weight_masks,
                        name="sequence-loss")

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer=Config.train.get('optimizer', 'Adam'),
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'gradients', 'learning_rate'],
            name="train_op")
