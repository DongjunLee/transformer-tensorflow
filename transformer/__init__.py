
from hbconfig import Config
import numpy as np
import tensorflow as tf

from .attention import positional_encoding
from .encoder import Encoder
from .decoder import Decoder



class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self,
              encoder_inputs=None,
              decoder_inputs=None):

        self.batch_size = tf.shape(encoder_inputs)[0]

        encoder_emb_inp = self.build_embed(encoder_inputs, encoder=True)
        self.encoder_outputs = self.build_encoder(encoder_emb_inp)

        decoder_emb_inp = self.build_embed(decoder_inputs, encoder=False, reuse=True)
        decoder_outputs = self.build_decoder(decoder_emb_inp, self.encoder_outputs)
        output =  self.build_output(decoder_outputs)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            predictions = tf.argmax(output, axis=2)
            return output, predictions
        else:
            next_decoder_inputs = self._filled_next_token(decoder_inputs, output, 1)

            # predict output with loop. [encoder_outputs, decoder_inputs (filled next token)]
            for i in range(2, Config.data.max_seq_length):
                decoder_emb_inp = self.build_embed(next_decoder_inputs, encoder=False, reuse=True)
                decoder_outputs = self.build_decoder(decoder_emb_inp, self.encoder_outputs, reuse=True)
                next_output = self.build_output(decoder_outputs, reuse=True)

                next_decoder_inputs = self._filled_next_token(next_decoder_inputs, next_output, i)

            # slice start_token
            decoder_input_start_1 = tf.slice(next_decoder_inputs, [0, 1],
                    [self.batch_size, Config.data.max_seq_length-1])
            predictions = tf.concat(
                    [decoder_input_start_1, tf.zeros([self.batch_size, 1], dtype=tf.int32)], axis=1)
            return next_output, predictions

    def build_embed(self, inputs, encoder=True, reuse=False):
        with tf.variable_scope("Embeddings", reuse=reuse, dtype=self.dtype) as scope:
            # Word Embedding
            embedding_encoder = tf.get_variable(
                "embedding_encoder", [Config.data.source_vocab_size, Config.model.model_dim], self.dtype)
            embedding_decoder = tf.get_variable(
                "embedding_decoder", [Config.data.target_vocab_size, Config.model.model_dim], self.dtype)

            # Positional Encoding
            with tf.variable_scope("positional-encoding"):
                positional_encoded = positional_encoding(Config.model.model_dim,
                                                         Config.data.max_seq_length,
                                                         dtype=self.dtype)

            # Add
            position_inputs = tf.tile(tf.range(0, Config.data.max_seq_length), [self.batch_size])
            position_inputs = tf.reshape(position_inputs,
                                         [self.batch_size, Config.data.max_seq_length]) # batch_size x [0, 1, 2, ..., n]

            if encoder:
                embedding_inputs = embedding_encoder
            else:
                embedding_inputs = embedding_decoder

            encoded_inputs = tf.add(tf.nn.embedding_lookup(embedding_inputs, inputs),
                             tf.nn.embedding_lookup(positional_encoded, position_inputs))

            return tf.nn.dropout(encoded_inputs, 1.0 - Config.model.dropout)

    def build_encoder(self, encoder_emb_inp, reuse=False):
        with tf.variable_scope("Encoder", reuse=reuse):
            encoder = Encoder(num_layers=Config.model.num_layers,
                              num_heads=Config.model.num_heads,
                              linear_key_dim=Config.model.linear_key_dim,
                              linear_value_dim=Config.model.linear_value_dim,
                              model_dim=Config.model.model_dim,
                              ffn_dim=Config.model.ffn_dim)

            return encoder.build(encoder_emb_inp)

    def build_decoder(self, decoder_emb_inp, encoder_outputs,reuse=False):
        with tf.variable_scope("Decoder", reuse=reuse):
            decoder = Decoder(num_layers=Config.model.num_layers,
                              num_heads=Config.model.num_heads,
                              linear_key_dim=Config.model.linear_key_dim,
                              linear_value_dim=Config.model.linear_value_dim,
                              model_dim=Config.model.model_dim,
                              ffn_dim=Config.model.ffn_dim)

            return decoder.build(decoder_emb_inp, encoder_outputs)

    def build_output(self, decoder_outputs, reuse=False):
        with tf.variable_scope("Output", reuse=reuse):
            logits = tf.layers.dense(decoder_outputs, Config.data.target_vocab_size)

        self.train_predictions = tf.argmax(logits[0], axis=1, name="train/pred_0")
        return logits

    def _filled_next_token(self, inputs, logits, decoder_index):
        tf.identity(tf.argmax(logits[0], axis=1, output_type=tf.int32), f'test/pred_{decoder_index}')

        next_token = tf.slice(
                tf.argmax(logits, axis=2, output_type=tf.int32),
                [0, decoder_index-1],
                [self.batch_size, 1])
        left_zero_pads = tf.zeros([self.batch_size, decoder_index], dtype=tf.int32)
        right_zero_pads = tf.zeros([self.batch_size, (Config.data.max_seq_length-decoder_index-1)], dtype=tf.int32)
        next_token = tf.concat((left_zero_pads, next_token, right_zero_pads), axis=1)

        return inputs + next_token
