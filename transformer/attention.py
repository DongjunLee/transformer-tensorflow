
import numpy as np
import tensorflow as tf



__all__ = [
    "positional_encoding", "Attention"
]


def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


class Attention:
    """Attention class"""

    def __init__(self,
                 num_heads=1,
                 masked=False,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=100):

        self.num_heads = num_heads
        self.masked = masked
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim

    def multi_head(self, q, k, v):
        self.q = q
        self.k = k
        self.v = v

        self._linear_projection()
        # TODO: multi-head split
        output = self._scaled_dot_product()
        # TODO: concat

        return tf.layers.dense(output, self.model_dim)

    def _linear_projection(self):
        self.q = tf.layers.dense(self.q, self.linear_key_dim)
        self.k = tf.layers.dense(self.q, self.linear_key_dim)
        self.v = tf.layers.dense(self.q, self.linear_value_dim)

    def _scaled_dot_product(self):
        o1 = tf.matmul(self.q, tf.transpose(self.k, [0, 2, 1]))
        o2 = o1 / (self.linear_key_dim**0.5)

        if self.masked:
            # TODO: implements masked
            pass

        o3 = tf.nn.softmax(o2)
        return tf.matmul(o3, self.v)






