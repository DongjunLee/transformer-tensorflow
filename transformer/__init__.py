
from hbconfig import Config
import numpy as np
import tensorflow as tf

from .attention import positional_encoding
from .encoder import Encoder
from .decoder import Decoder



class Graph:

    def __init__(self,
                 encoder_inputs=None,
                 decoder_inputs=None,
                 dtype=tf.float32):

        self.encoder_inputs = encoder_inputs
        self.encoder_input_lengths = tf.reduce_sum(
            tf.to_int32(tf.not_equal(self.encoder_inputs, Config.data.PAD_ID)), 1,
            name="encoder_input_lengths")

        self.decoder_inputs = decoder_inputs
        self.decoder_input_lengths = tf.reduce_sum(
            tf.to_int32(tf.not_equal(self.decoder_inputs, Config.data.PAD_ID)), 1,
            name="decoder_input_lengths")

        self.dtype = dtype

    def build(self, mode):
        self.mode = mode

        self._build_embed()
        self._build_encoder()
        self._build_decoder()
        self._build_output()

    def _build_embed(self):
        with tf.variable_scope("Embeddings", dtype=self.dtype) as scope:
            # Word Embedding
            self.embedding_encoder = tf.get_variable(
                "embedding_encoder", [Config.data.source_vocab_size, Config.model.model_dim], self.dtype)
            self.embedding_decoder = tf.get_variable(
                "embedding_decoder", [Config.data.target_vocab_size, Config.model.model_dim], self.dtype)

            # Positional Encoding
            with tf.variable_scope("positional-encoding"):
                self.positional_encoding = positional_encoding(Config.model.model_dim, Config.data.max_seq_length, dtype=self.dtype)

            # Add
            position_inputs = tf.tile(tf.range(0, Config.data.max_seq_length), [Config.model.batch_size])
            position_inputs = tf.reshape(position_inputs, [Config.model.batch_size, Config.data.max_seq_length]) # batch_size x [0, 1, 2, ..., n]

            self.encoder_emb_inp = tf.add(tf.nn.embedding_lookup(self.embedding_encoder, self.encoder_inputs),
                                          tf.nn.embedding_lookup(self.positional_encoding, position_inputs))
            self.decoder_emb_inp = tf.add(tf.nn.embedding_lookup(self.embedding_decoder, self.decoder_inputs),
                                          tf.nn.embedding_lookup(self.positional_encoding, position_inputs))


    def _build_encoder(self):
        with tf.variable_scope("Encoder"):
            encoder = Encoder(num_layers=Config.model.num_layers,
                              num_heads=Config.model.num_heads,
                              linear_key_dim=Config.model.linear_key_dim,
                              linear_value_dim=Config.model.linear_value_dim,
                              model_dim=Config.model.model_dim,
                              ffn_dim=Config.model.ffn_dim)

            self.encoder_outputs = encoder.build(self.encoder_emb_inp)

    def _build_decoder(self):
        with tf.variable_scope("Decoder"):
            decoder = Decoder(num_layers=Config.model.num_layers,
                              num_heads=Config.model.num_heads,
                              linear_key_dim=Config.model.linear_key_dim,
                              linear_value_dim=Config.model.linear_value_dim,
                              model_dim=Config.model.model_dim,
                              ffn_dim=Config.model.ffn_dim)

            self.decoder_outputs = decoder.build(self.decoder_emb_inp, self.encoder_outputs)

    def _build_output(self):
        with tf.variable_scope("Output"):
            flatted_output = tf.reshape(self.decoder_outputs, [Config.model.batch_size, -1])
            self.logits = tf.layers.dense(flatted_output, Config.data.target_vocab_size)

        self.train_predictions = tf.argmax(self.logits[0], axis=0, name="train/pred_0")
