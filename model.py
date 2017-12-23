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
        self.encoder_inputs = features["enc_inputs"]
        self.targets = labels

        start_tokens = tf.fill([Config.model.batch_size, 1], Config.data.START_ID)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            # slice last pad token
            target_slice_last_1 = tf.slice(self.targets, [0, 0],
                    [Config.model.batch_size, Config.data.max_seq_length-1])
            self.decoder_inputs = tf.concat([start_tokens, target_slice_last_1], axis=1)

            tf.identity(self.decoder_inputs[0], 'train/dec_0')
        else:
            pad_tokens = tf.zeros([Config.model.batch_size, Config.data.max_seq_length-1], dtype=tf.int32) # 0: PAD ID
            self.decoder_inputs = tf.concat([start_tokens, pad_tokens], axis=1)

            tf.identity(self.decoder_inputs[0], 'test/dec_0')

    def build_graph(self):
        graph = transformer.Graph(self.mode)
        output = graph.build(encoder_inputs=self.encoder_inputs,
                    decoder_inputs=self.decoder_inputs)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_loss(output)
            self._build_optimizer()
        else:
            def _filled_next_token(inputs, logits, decoder_index):
                tf.identity(tf.argmax(logits[0], axis=1, output_type=tf.int32), f'test/pred_{decoder_index}')

                next_token = tf.slice(
                        tf.argmax(logits, axis=2, output_type=tf.int32),
                        [0, decoder_index-1],
                        [Config.model.batch_size, 1])
                left_zero_pads = tf.zeros([Config.model.batch_size, decoder_index], dtype=tf.int32)
                right_zero_pads = tf.zeros([Config.model.batch_size, (Config.data.max_seq_length-decoder_index-1)], dtype=tf.int32)
                next_token = tf.concat((left_zero_pads, next_token, right_zero_pads), axis=1)

                return inputs + next_token

            encoder_outputs = graph.encoder_outputs
            decoder_inputs = _filled_next_token(self.decoder_inputs, output, 1)

            tf.identity(decoder_inputs[0], 'test/dec_1')
            tf.identity(tf.argmax(output[0], axis=1, output_type=tf.int32), 'test/pred_1')

            # predict output with loop. [encoder_outputs, decoder_inputs (filled next token)]
            for i in range(2, Config.data.max_seq_length):
                decoder_emb_inp = graph.build_embed(decoder_inputs, encoder=False, reuse=True)
                decoder_outputs = graph.build_decoder(decoder_emb_inp, encoder_outputs, reuse=True)
                next_output = graph.build_output(decoder_outputs, reuse=True)

                decoder_inputs = _filled_next_token(decoder_inputs, next_output, i)
                tf.identity(decoder_inputs[0], f'test/dec_{i}')
                tf.identity(tf.argmax(next_output[0], axis=1, output_type=tf.int32), f'test/pred_{i}')

            self._build_loss(next_output)

	    # slice start_token
            decoder_input_start_1 = tf.slice(decoder_inputs, [0, 1],
                    [Config.model.batch_size, Config.data.max_seq_length-1])
            self.predictions = tf.concat(
                    [decoder_input_start_1, tf.zeros([Config.model.batch_size, 1], dtype=tf.int32)], axis=1)

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

                return nltk.translate.bleu_score.corpus_bleu(labels, predictions)

            score = tf.py_func(_nltk_blue_score, (labels, predictions), tf.float64)
            return tf.metrics.mean(score * 100.0)

        return {
            "bleu": blue_score(self.targets, self.predictions)
        }
