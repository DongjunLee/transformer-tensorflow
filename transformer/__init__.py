
from hbconfig import Config
import numpy as np
import tensorflow as tf

from .attention import Attention
from .layer import FFN



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

    def _build_embed(self):
        with tf.variable_scope ("embeddings", dtype=self.dtype) as scope:
            # Word Embedding
            self.embedding_encoder = tf.get_variable(
                "embedding_encoder", [Config.data.vocab_size, Config.model.model_dim], self.dtype)
            self.embedding_decoder = tf.get_variable(
                "embedding_decoder", [Config.data.vocab_size, Config.model.model_dim], self.dtype)

            # Positional Encoding
            dim, sentence_length = Config.model.model_dim, Config.data.max_seq_length

            encoded_vec = np.array([pos/10000**(2*i/dim) for pos in range(sentence_length) for i in range(dim)])
            encoded_vec[::2] = np.sin(encoded_vec[::2])
            encoded_vec[1::2] = np.cos(encoded_vec[1::2])

            self.positional_encoding = tf.convert_to_tensor(
                    encoded_vec.reshape([sentence_length, dim]), name="positional_encoding")

            # Add
            position_inputs = tf.tile(tf.range(0, sentence_length), Config.model.batch_size)
            position_inputs = tf.reshape(position_inputs, [Config.model.batch_size, Config.data.max_seq_length])

            self.encoder_emb_inp = tf.add(tf.nn.embedding_lookup(self.embedding_encoder, self.encoder_inputs),
                                          tf.nn.embedding_lookup(self.positional_encoding, encoder_position))
            self.decoder_emb_inp = tf.add(tf.nn.embedding_lookup(self.embedding_decoder, self.decoder_inputs),
                                          tf.nn.embedding_lookup(self.positional_encoding, position_inputs))

            self.encoder_previous_input = tf.identity(self.encoder_emb_inp)
            self.decoder_previous_input = tf.identity(self.decoder_emb_inp)

    def _build_encoder(self):
        with tf.variable_scope("Encoder"):

            for i in range(1, Config.model.num_layers+1):
                with tf.variable_scope(f"layer-{i}"):

                    with tf.variable_scope("self-attention"):
                        # Multi-Head Attention
                        attention = Attention(num_heads=Config.model.num_heads,
                                              masked=False,
                                              linear_key_dim=Config.model.linear_key_dim,
                                              linear_value_dim=Config.model.linear_value_dim,
                                              model_dim=Config.model.model_dim)
                        output = attention.multi_head(q=self.encoder_previous_input,
                                                      k=self.encoder_previous_input,
                                                      v=self.encoder_previous_input)

                    # Add and Norm (with Residual connection)
                    output = tf.contrib.layers.layer_norm(
                            tf.add(self.encoder_previous_input, output))
                    self.encoder_previous_input = tf.identity(output)

                    # Position-wise FFN
                    ffn = FFN(w1_dim=Config.model.ffn_dim,
                              w2_dim=Config.model.model_dim)
                    output = ffn.dense_relu_dense(output)

                    # Add and Norm (with Residual connection)
                    output = tf.contrib.layers.layer_norm(
                            tf.add(self.encoder_previous_input, output))
                    self.encoder_previous_input = tf.identity(output)

            self.encoder_output = tf.identity(output)

    def _build_decoder(self):
        with tf.variable_scope("Decoder"):
            # TODO : masked multi-head attention -> encoder-decoder attention (multi-head) -> position-wise feed forward

            for i in range(1, Config.model.num_layers+1):
                with tf.variable_scope(f"layer-{i}"):

                    with tf.variable_scope("self-attention"):
                        # Multi-Head Attention
                        attention = Attention(num_heads=Config.model.num_heads,
                                              masked=False,
                                              linear_key_dim=Config.model.linear_key_dim,
                                              linear_value_dim=Config.model.linear_value_dim,
                                              model_dim=Config.model.model_dim)
                        output = attention.multi_head(q=self.decoder_previous_input,
                                                      k=self.decoder_previous_input,
                                                      v=self.decoder_previous_input)

                    # Add and Norm (with Residual connection)
                    output = tf.contrib.layers.layer_norm(
                            tf.add(self.decoder_previous_input, output))
                    self.decoder_previous_input = tf.identity(output)

                    with tf.variable_scope("encoder-decoder-attention"):
                        attention = Attention(num_heads=Config.model.num_heads,
                                              masked=True,
                                              linear_key_dim=Config.model.linear_key_dim,
                                              linear_value_dim=Config.model.linear_value_dim,
                                              model_dim=Config.model.model_dim)
                        output = attention.multi_head(q=self.decoder_previous_input,
                                                      k=self.encoder_output,
                                                      v=self.encoder_output)

                    # Position-wise FFN
                    ffn = FFN(w1_dim=Config.model.ffn_dim,
                              w2_dim=Config.model.model_dim)
                    output = ffn.dense_relu_dense(output)

                    # Add and Norm (with Residual connection)
                    output = tf.contrib.layers.layer_norm(
                            tf.add(self.encoder_previous_input, output))
                    self.encoder_previous_input = tf.identity(output)

            self.decoder_output = tf.identity(output)


    def _build_output(self):
        with tf.variable_scope("Output"):
            output = tf.layers(self.decoder_output, Config.data.target_vocab_size)
            self.logits = tf.nn.softmax(output)
