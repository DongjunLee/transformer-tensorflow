# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import operator
import os
import random
import re

from nltk.tokenize import TweetTokenizer
import numpy as np
from hbconfig import Config
import tensorflow as tf
from tqdm import tqdm



tokenizer = TweetTokenizer()


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


def get_dataset_batch(data, buffer_size=10000, batch_size=64, scope="train"):

    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        with tf.name_scope(scope):

            nonlocal data
            enc_inputs, targets = data

            # Define placeholders
            enc_placeholder = tf.placeholder(
                tf.int32, [None, None], name="enc_placeholder")
            target_placeholder = tf.placeholder(
                tf.int32, [None, None], name="target_placeholder")

            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (enc_placeholder, target_placeholder))

            if scope == "train":
                dataset = dataset.repeat(None)  # Infinite iterations
            else:
                dataset = dataset.repeat(1)  # one Epoch

            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            next_enc, next_target = iterator.get_next()

            tf.identity(next_enc[0], 'enc_0')
            tf.identity(next_target[0], 'target_0')

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={enc_placeholder: enc_inputs,
                               target_placeholder: targets})

            # Return batched (features, labels)
            return {"enc_inputs": next_enc}, next_target

    # Return function and hook
    return inputs, iterator_initializer_hook


def prepare_dataset(questions, answers):

    # random convos to create the test set
    test_ids = random.sample([i for i in range(len(questions))], Config.data.testset_size)

    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []
    for filename in filenames:
        files.append(open(os.path.join(Config.data.base_path, Config.data.processed_path, filename), 'wb'))

    for i in tqdm(range(len(questions))):

        question = questions[i]
        answer = answers[i]

        if i in test_ids:
            files[2].write((question + "\n").encode('utf-8'))
            files[3].write((answer + '\n').encode('utf-8'))
        else:
            files[0].write((question + '\n').encode('utf-8'))
            files[1].write((answer + '\n').encode('utf-8'))

    for file in files:
        file.close()


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def build_vocab(in_fname, out_fname, normalize_digits=True):
    print("Count each vocab frequency ...")

    def count_vocab(fname):
        vocab = {}
        with open(fname, 'rb') as f:
            for line in tqdm(f.readlines()):
                line = line.decode('utf-8')
                for token in tokenizer.tokenize(line):
                    if not token in vocab:
                        vocab[token] = 0
                    vocab[token] += 1
        return vocab

    in_path = os.path.join(Config.data.base_path, Config.data.raw_data_path, in_fname)
    out_path = os.path.join(Config.data.base_path, Config.data.raw_data_path, out_fname)

    source_vocab = count_vocab(in_path)
    target_vocab = count_vocab(out_path)

    print("total vocab size:", len(source_vocab), len(target_vocab))

    def write_vocab(fname, sorted_vocab):
        dest_path = os.path.join(Config.data.base_path, Config.data.processed_path, fname)
        with open(dest_path, 'wb') as f:
            f.write(('<pad>' + '\n').encode('utf-8'))
            f.write(('<unk>' + '\n').encode('utf-8'))
            f.write(('<s>' + '\n').encode('utf-8'))
            f.write(('<\s>' + '\n').encode('utf-8'))
            index = 4
            for word, count in tqdm(sorted_vocab):
                if count < Config.data.word_threshold:
                    break

                f.write((word + '\n').encode('utf-8'))
                index += 1

    sorted_source_vocab = sorted(source_vocab.items(), key=operator.itemgetter(1), reverse=True)
    sorted_target_vocab = sorted(target_vocab.items(), key=operator.itemgetter(1), reverse=True)

    write_vocab("source_vocab", sorted_source_vocab)
    write_vocab("target_vocab", sorted_target_vocab)

def load_vocab(vocab_fname):
    print("load vocab ...")
    with open(os.path.join(Config.data.base_path, Config.data.processed_path, vocab_fname), 'rb') as f:
        words = f.read().decode('utf-8').splitlines()
        print("vocab size:", len(words))
    return {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer.tokenize(line)]


def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab'
    if mode == "enc":
        vocab_path = 'source_' + vocab_path
    elif mode == "dec":
        vocab_path = 'target_' + vocab_path

    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    vocab = load_vocab(vocab_path)
    in_file = open(os.path.join(Config.data.base_path, Config.data.raw_data_path, in_path), 'rb')
    out_file = open(os.path.join(Config.data.base_path, Config.data.processed_path, out_path), 'wb')

    lines = in_file.read().decode('utf-8').splitlines()
    for line in tqdm(lines):
        ids = []

        sentence_ids = sentence2id(vocab, line)
        ids.extend(sentence_ids)
        if mode == 'dec':
            ids.append(vocab['<\s>'])
            ids.append(vocab['<pad>'])

        out_file.write(b' '.join(str(id_).encode('utf-8') for id_ in ids) + b'\n')


def process_data():
    print('Preparing data to be model-ready ...')

    # create path to store all the train & test encoder & decoder
    make_dir(Config.data.base_path + Config.data.processed_path)

    build_vocab('train.enc', 'train.dec')

    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')


def make_train_and_test_set(shuffle=True):
    print("make Training data and Test data Start....")

    if Config.data.get('max_seq_length', None) is None:
        set_max_seq_length(['train_ids.enc', 'train_ids.dec', 'test_ids.enc', 'test_ids.dec'])

    train_enc, train_dec = load_data('train_ids.enc', 'train_ids.dec')
    test_enc, test_dec = load_data('test_ids.enc', 'test_ids.dec', train=False)

    assert len(train_enc) == len(train_dec)
    assert len(test_enc) == len(test_dec)

    print(f"train data count : {len(train_dec)}")
    print(f"test data count : {len(test_dec)}")

    if shuffle:
        print("shuffle dataset ...")
        train_p = np.random.permutation(len(train_dec))
        test_p = np.random.permutation(len(test_dec))

        return ((train_enc[train_p], train_dec[train_p]),
                (test_enc[test_p], test_dec[test_p]))
    else:
        return ((train_enc, train_dec),
                (test_enc, test_dec))

def load_data(enc_fname, dec_fname, train=True):
    enc_input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, enc_fname), 'r')
    dec_input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, dec_fname), 'r')

    enc_data, dec_data = [], []
    for e_line, d_line in tqdm(zip(enc_input_data.readlines(), dec_input_data.readlines())):
        e_ids = [int(id_) for id_ in e_line.split()]
        d_ids = [int(id_) for id_ in d_line.split()]

        if len(e_ids) == 0 or len(d_ids) == 0:
            continue

        if len(e_ids) <= Config.data.max_seq_length and len(d_ids) < Config.data.max_seq_length:
            enc_data.append(_pad_input(e_ids, Config.data.max_seq_length))
            dec_data.append(_pad_input(d_ids, Config.data.max_seq_length))

    print(f"load data from {enc_fname}, {dec_fname}...")
    return np.array(enc_data, dtype=np.int32), np.array(dec_data, dtype=np.int32)


def _pad_input(input_, size):
    return input_ + [Config.data.PAD_ID] * (size - len(input_))


def set_max_seq_length(dataset_fnames):

    max_seq_length = Config.data.get('max_seq_length', 10)

    for fname in dataset_fnames:
        input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, fname), 'r')

        for line in input_data.readlines():
            ids = [int(id_) for id_ in line.split()]
            seq_length = len(ids)

            if seq_length > max_seq_length:
                max_seq_length = seq_length

    Config.data.max_seq_length = max_seq_length
    print(f"Setting max_seq_length to Config : {max_seq_length}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    process_data()
