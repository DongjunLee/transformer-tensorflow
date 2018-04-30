#-*- coding: utf-8 -*-

import argparse
import os

from hbconfig import Config
import numpy as np
import tensorflow as tf

import data_loader
from model import Model
import utils




def main(ids, vocab):

    X = np.array(data_loader._pad_input(ids, Config.data.max_seq_length), dtype=np.int32)
    X = np.reshape(X, (1, Config.data.max_seq_length))

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"enc_inputs": X},
        num_epochs=1,
        shuffle=False)

    estimator = _make_estimator()
    result = estimator.predict(input_fn=predict_input_fn)

    prediction = next(result)["prediction"]

    rev_vocab = utils.get_rev_vocab(vocab)
    def to_str(sequence):
        tokens = [
            rev_vocab.get(x, '') for x in sequence if x != Config.data.PAD_ID]
        return ' '.join(tokens)

    return to_str(prediction)


def _make_estimator():
    params = tf.contrib.training.HParams(**Config.model.to_dict())
    # Using CPU
    run_config = tf.contrib.learn.RunConfig(
        model_dir=Config.train.model_dir,
        session_config=tf.ConfigProto(
            device_count={'GPU': 0}
        ))

    model = Model()
    return tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--src', type=str, default='example source sentence',
                        help='input source sentence')
    args = parser.parse_args()

    Config(args.config)
    Config.train.batch_size = 1

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    # set data property
    data_loader.set_max_seq_length(['train_ids.enc', 'train_ids.dec', 'test_ids.enc', 'test_ids.dec'])

    source_vocab = data_loader.load_vocab("source_vocab")
    target_vocab = data_loader.load_vocab("target_vocab")

    Config.data.rev_source_vocab = utils.get_rev_vocab(source_vocab)
    Config.data.rev_target_vocab = utils.get_rev_vocab(target_vocab)
    Config.data.source_vocab_size = len(source_vocab)
    Config.data.target_vocab_size = len(target_vocab)

    print("------------------------------------")
    print("Source: " + args.src)
    token_ids = data_loader.sentence2id(source_vocab, args.src)
    prediction = main(token_ids, target_vocab)

    print(" > Result: " + prediction)
