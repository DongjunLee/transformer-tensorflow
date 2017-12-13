
import tensorflow as tf


__all__ = [
    "Attention", "Encoder", "Decoder",
]


def positional_encoding(self):
    """positional encoding"""
    pass

class Attention:
    """Attention class"""

    def __init__(self):
        pass

    def multi_head(self, q, k, v,
                   num_heads=8,
                   linear_key_dim=50,
                   linear_value_dim=50):
        pass

    def scaled_dot_product(self, q, k, v,
                           mask=False):
        pass


class Encoder:
    """Encoder class"""

    def __init__(self):
        pass


class Decoder:
    """Decoder class"""

    def __init__(self):
        pass


class FFN:
    """FFN class (Position-wise Feed-Forward Networks)"""

    def __init__(self):
        pass

    def dense_relu_dense(self):
        pass

    def conv_relu_conv(self):
        pass


class Add_and_Norm:
    """Add_and_Norm class (Residual connection then LayerNorm)"""

    def __init__(self):
        pass


