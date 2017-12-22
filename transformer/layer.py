
import tensorflow as tf



class FFN:
    """FFN class (Position-wise Feed-Forward Networks)"""

    def __init__(self,
                 w1_dim=200,
                 w2_dim=100,
                 dropout=0.1):

        self.w1_dim = w1_dim
        self.w2_dim = w2_dim
        self.dropout = dropout

    def dense_relu_dense(self, inputs):
        output = tf.layers.dense(inputs, self.w1_dim, activation=tf.nn.relu)
        output =tf.layers.dense(output, self.w2_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def conv_relu_conv(self):
        raise NotImplementedError("i will implement it!")
