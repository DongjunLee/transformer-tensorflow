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
                loss=self.loss,
                eval_metric_ops=self._build_metric())

    def _init_placeholder(self, features, labels):
        if type(features) == dict:
            self.encoder_inputs = features["enc_inputs"]
            self.decoder_inputs = features["dec_inputs"]

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.targets = labels

    def build_graph(self):
        graph = transformer.Graph(self.mode)
        output = graph.build(encoder_inputs=self.encoder_inputs,
                    decoder_inputs=self.decoder_inputs)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_loss(output)
            self._build_optimizer()
        else:
            def _filled_next_token(inputs, logits, decoder_index):
                next_token = tf.reshape(tf.argmax(logits, axis=1, output_type=tf.int32), [Config.model.batch_size, 1])
                left_zero_pads = tf.zeros([Config.model.batch_size, decoder_index], dtype=tf.int32)
                right_zero_pads = tf.zeros([Config.model.batch_size, (Config.data.max_seq_length-decoder_index-1)], dtype=tf.int32)
                next_token = tf.concat((left_zero_pads, next_token, right_zero_pads), axis=1)

                return inputs + next_token

            encoder_outputs = graph.encoder_outputs
            decoder_inputs = _filled_next_token(self.decoder_inputs, output, 1)
            sequence_logits = tf.reshape(output, [-1, 1, Config.data.target_vocab_size])

            # predict output with loop. [encoder_outputs, decoder_inputs (filled next token)]
            for i in range(2, Config.data.max_seq_length):
                decoder_emb_inp = graph.build_embed(decoder_inputs, encoder=False, reuse=True)
                decoder_outputs = graph.build_decoder(decoder_emb_inp, encoder_outputs, reuse=True)
                next_output = graph.build_output(decoder_outputs, reuse=True)

                decoder_inputs = _filled_next_token(decoder_inputs, next_output, i)
                sequence_logits = tf.concat((sequence_logits, tf.reshape(next_output, [-1, 1, Config.data.target_vocab_size])),
                                            axis=1)

            self._build_loss(sequence_logits)
            self.predictions = decoder_inputs

    def _build_loss(self, logits):
        with tf.variable_scope('loss'):
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                self.loss = tf.losses.sparse_softmax_cross_entropy(
                            labels=self.targets,
                            logits=logits,
                            scope="cross-entropy")
            else:
                # Slice start_token (decoder_inputs have start_token always)
                targets = tf.slice(self.targets, [0, 1],
                                   [Config.model.batch_size, Config.data.max_seq_length-1])
                target_lengths = tf.reduce_sum(
                        tf.to_int32(tf.not_equal(self.targets, Config.data.PAD_ID)), 1) - 1
                weight_masks = tf.sequence_mask(
                        lengths=target_lengths,
                        maxlen=Config.data.max_seq_length - 1,
                        dtype=self.dtype, name='masks')

                self.loss = tf.contrib.seq2seq.sequence_loss(
                        logits=logits,
                        targets=targets,
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
                rev_target_vocab = Config.data.rev_target_vocab

                labels = [
                    [[rev_target_vocab.get(w_id, "") for w_id in label]]
                    for label in labels.tolist()]
                predictions = [
                    [rev_target_vocab.get(w_id, "") for w_id in prediction]
                    for prediction in predictions.tolist()]

                if Config.train.print_verbose:
                    print("label: ", labels[0][0])
                    print("prediction: ", predictions[0])

                return nltk.translate.bleu_score.corpus_bleu(labels, predictions)

            score = tf.py_func(_nltk_blue_score, (labels, predictions), tf.float64)
            return tf.metrics.mean(score * 100)

        return {
            "bleu": blue_score(self.targets, self.predictions)
        }
