
import tensorflow as tf

from .attention import Attention
from .layer import FFN



class Encoder:
    """Encoder class"""

    def __init__(self,
                 num_layers=8,
                 num_heads=8,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=50,
                 ffn_dim=50,
                 dropout=0.2):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout

    def build(self, encoder_inputs):
        o1 = tf.identity(encoder_inputs)

        for i in range(1, self.num_layers+1):
            with tf.variable_scope(f"layer-{i}"):
                o2 = self._add_and_norm(o1, self._self_attention(q=o1,
                                                                 k=o1,
                                                                 v=o1), num=1)
                o3 = self._add_and_norm(o2, self._positional_feed_forward(o2), num=2)
                o1 = tf.identity(o3)

        return o3

    def _self_attention(self, q, k, v):
        with tf.variable_scope("self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=False,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope(f"add-and-norm-{num}"):
            return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

    def _positional_feed_forward(self, output):
        with tf.variable_scope("feed-forward"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout)
            return ffn.dense_relu_dense(output)

