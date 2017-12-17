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

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"prediction": self.prediction})
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                # predictions=self.predictions,
                loss=self.loss,
                train_op=self.train_op
            )

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
        graph = transformer.Graph(
                    encoder_inputs=self.encoder_inputs,
                    decoder_inputs=self.decoder_inputs)
        graph.build(self.mode)

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.predictions = graph.predictions
        else:
            # self.train_predictions = graph.train_predictions
            self._build_loss(graph.logits)
            self._build_optimizer()

    def _build_loss(self, logits):
        with tf.variable_scope('loss'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(
                    labels=self.targets,
                    logits=logits,
                    scope="cross-entropy")

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer=Config.train.get('optimizer', 'Adam'),
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'gradients', 'learning_rate'],
            name="train_op")
