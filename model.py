from __future__ import print_function


from hbconfig import Config
import tensorflow as tf

from model_helper import Encoder
from model_helper import Episode



class Transformer:

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
                predictions=self.predictions,
                loss=self.loss,
                train_op=self.train_op,
                eval_metric_ops={
                    # implements BLEU metric
                }
            )

    def _set_batch_size(self, mode):
        if mode == tf.estimator.ModeKeys.EVAL:
            Config.model.batch_size = Config.eval.batch_size
        else:
            Config.model.batch_size = Config.train.batch_size

    def _init_placeholder(self, features, labels):
        self.input_data = features
        if type(features) == dict:
            self.input_data = features["input_data"]
        self.targets = labels

    def build_graph(self):
        self._build_embed()
        self._build_encoder()
        self._build_decoder()

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss()
            self._build_optimizer()

    def _build_embed(self):
        with tf.variable_scope('Embedding'):
            # TODO : embedding + positional encoding

            with tf.variable_scope('positional-encoding'):


    def _build_encoder(self):
        with tf.variable_scope("Encoder"):
            # TODO : multi-head attention -> position-wise feed forward

            with tf.variable_scope("self-attention"):


    def _build_decoder(self):
        with tf.variable_scope("Decoder"):
            # TODO : masked multi-head attention -> encoder-decoder attention (multi-head) -> position-wise feed forward

            with tf.variable_scope("self-attention"):

            with tf.variable_scope("encoder-decoder-attention"):


    def _build_output(self):
        with tf.variable_scope("Output"):
            # TODO: Linear -> Softmax
            pass

    def _build_loss(self):
        with tf.variable_scope('loss'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                    self.targets,
                    self.logits,
                    scope="cross-entropy")
            reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.loss = tf.add(cross_entropy, reg_term)

    def _build_optimizer(self):
        self.train_op = layers.optimize_loss(
            self.loss, tf.train.get_global_step(),
            optimizer=Config.train.get('optimizer', 'Adam'),
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'gradients', 'learning_rate'],
            name="train_op")
