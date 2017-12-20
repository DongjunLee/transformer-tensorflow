
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

        encoder_emb_inp = self.build_embed(encoder_inputs, encoder=True)
        self.encoder_outputs = self.build_encoder(encoder_emb_inp)

        decoder_emb_inp = self.build_embed(decoder_inputs, encoder=False, reuse=True)
        decoder_outputs = self.build_decoder(decoder_emb_inp, self.encoder_outputs)
        self.build_output(decoder_outputs)

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
            position_inputs = tf.tile(tf.range(0, Config.data.max_seq_length), [Config.model.batch_size])
            position_inputs = tf.reshape(position_inputs,
                                         [Config.model.batch_size, Config.data.max_seq_length]) # batch_size x [0, 1, 2, ..., n]

            if encoder:
                embedding_inputs = embedding_encoder
            else:
                embedding_inputs = embedding_decoder
            return tf.add(tf.nn.embedding_lookup(embedding_inputs, inputs),
                          tf.nn.embedding_lookup(positional_encoded, position_inputs))

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
            flatted_output = tf.reshape(decoder_outputs, [Config.model.batch_size, -1])
            self.logits = tf.layers.dense(flatted_output, Config.data.target_vocab_size)

        self.train_predictions = tf.argmax(self.logits[0], axis=0, name="train/pred_0")
