
from hbconfig import Config
import tensorflow as tf



class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


def get_inputs(data, buffer_size=10000, batch_size=64, scope="train"):

    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        with tf.name_scope(scope):

            nonlocal data
            enc_inputs, dec_inputs, targets = data

            # Define placeholders
            enc_placeholder = tf.placeholder(
                tf.int32, [None, None], name="enc_placeholder")
            dec_placeholder = tf.placeholder(
                tf.int32, [None, None], name="dec_placeholder")
            target_placeholder = tf.placeholder(
                tf.int32, [None], name="target_placeholder")

            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (enc_placeholder, dec_placeholder, target_placeholder))

            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(Config.train.batch_size)

            iterator = dataset.make_initializable_iterator()
            next_enc, next_dec, next_target = iterator.get_next()

            tf.identity(next_enc[0], 'enc_0')
            tf.identity(next_dec[0], 'dec_0')
            tf.identity(next_target[0], 'target_0')

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={enc_placeholder: enc_inputs,
                               dec_placeholder: dec_inputs,
                               target_placeholder: targets})

            features = {"enc_inputs": next_enc,
                        "dec_inputs": next_dec}

            # Return batched (features, labels)
            return features, next_target

    # Return function and hook
    return inputs, iterator_initializer_hook
