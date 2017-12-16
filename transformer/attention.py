
import tensorflow as tf



class Attention:
    """Attention class"""

    def __init__(self):
        pass

    def multi_head(self, q, k, v,
                   masked=False,
                   num_heads=8,
                   linear_key_dim=50,
                   linear_value_dim=50,
                   model_dim=100):

        self._linear_projection()
        # TODO: multi-head split
        output = self._scaled_dot_product()
        # TODO: concat

        return tf.layers.dense(output, model_dim)

    def _linear_projection(self):
        self.q = tf.layers.dense(self.q, self.linear_key_dim)
        self.k = tf.layers.dense(self.q, self.linear_key_dim)
        self.v = tf.layers.dense(self.q, self.linear_value_dim)

    def _scaled_dot_product(self):
        o1 = tf.matmul(self.q, tf.transpose(self.k))
        o2 = o1 / (self.linear_key_dim**0.5)

        if self.masked:
            # TODO: implements masked
            pass

        o3 = tf.nn.softmax(o2)
        return tf.matmul(o3, self.v)






