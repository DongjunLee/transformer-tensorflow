from __future__ import print_function


from hbconfig import Config
import tensorflow as tf

import nltk
import transformer



class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.dtype = tf.float32

        self.mode = mode
        self.params = params

        self.loss, self.train_op, self.metrics, self.predictions = None, None, None, None
        self._init_placeholder(features, labels)
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions={"prediction": self.predictions})

    def _init_placeholder(self, features, labels):
        self.encoder_inputs = features["enc_inputs"]
        self.targets = labels

        self.batch_size = tf.shape(self.encoder_inputs)[0]
        start_tokens = tf.fill([self.batch_size, 1], Config.data.START_ID)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            # slice last pad token
            target_slice_last_1 = tf.slice(self.targets, [0, 0],
                    [self.batch_size, Config.data.max_seq_length-1])
            self.decoder_inputs = tf.concat([start_tokens, target_slice_last_1], axis=1)

            tf.identity(self.decoder_inputs[0], 'train/dec_0')
        else:
            pad_tokens = tf.zeros([self.batch_size, Config.data.max_seq_length-1], dtype=tf.int32) # 0: PAD ID
            self.decoder_inputs = tf.concat([start_tokens, pad_tokens], axis=1)

            tf.identity(self.decoder_inputs[0], 'test/dec_0')

    def build_graph(self):
        graph = transformer.Graph(self.mode)
        output, predictions = graph.build(encoder_inputs=self.encoder_inputs,
                             decoder_inputs=self.decoder_inputs)

        self.predictions = predictions
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(output)
            self._build_optimizer()
            self._build_metric()

    def _build_loss(self, logits):
        with tf.variable_scope('loss'):
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

    def _build_metric(self):

        def blue_score(labels, predictions,
                       weights=None, metrics_collections=None,
                       updates_collections=None, name=None):

            def _nltk_blue_score(labels, predictions):

                # slice after <eos>
                predictions = predictions.tolist()
                for i in range(len(predictions)):
                    prediction = predictions[i]
                    if Config.data.EOS_ID in prediction:
                        predictions[i] = prediction[:prediction.index(Config.data.EOS_ID)+1]

                rev_target_vocab = Config.data.rev_target_vocab

                labels = [
                    [[rev_target_vocab.get(w_id, "") for w_id in label if w_id != Config.data.PAD_ID]]
                    for label in labels.tolist()]
                predictions = [
                    [rev_target_vocab.get(w_id, "") for w_id in prediction]
                    for prediction in predictions]

                if Config.train.print_verbose:
                    print("label: ", labels[0][0])
                    print("prediction: ", predictions[0])

                return float(nltk.translate.bleu_score.corpus_bleu(labels, predictions))

            score = tf.py_func(_nltk_blue_score, (labels, predictions), tf.float64)
            return tf.metrics.mean(score * 100.0)

        self.metrics = {
            "bleu": blue_score(self.targets, self.predictions)
        }
