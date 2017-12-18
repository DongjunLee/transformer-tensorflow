#-- coding: utf-8 -*-

import argparse
import logging

from hbconfig import Config
import tensorflow as tf

import data_loader
from model import Model
import hook



def experiment_fn(run_config, params):

    model = Model()
    estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)

    source_vocab = data_loader.load_vocab("source_vocab")
    target_vocab = data_loader.load_vocab("target_vocab")

    Config.data.source_vocab_size = len(source_vocab)
    Config.data.target_vocab_size = len(target_vocab)

    train_data, test_data = data_loader.make_train_and_test_set()

    train_input_fn, train_input_hook = data_loader.get_dataset_batch(train_data, scope="train")
    test_input_fn, test_input_hook = data_loader.get_dataset_batch(test_data, scope="test")

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=Config.train.train_steps,
        min_eval_frequency=Config.train.min_eval_frequency,
        train_monitors=[
            train_input_hook,
            hook.print_variables(
                variables=['train/enc_0'],
                rev_vocab=get_rev_vocab(source_vocab),
                every_n_iter=Config.train.check_hook_n_iter),
            hook.print_variables(
                variables=['train/dec_0', 'train/target_0', 'train/pred_0'],
                rev_vocab=get_rev_vocab(target_vocab),
                every_n_iter=Config.train.check_hook_n_iter)],
        eval_hooks=[test_input_hook]
    )
    return experiment


def get_rev_vocab(vocab):
    if vocab is None:
        return None
    return {idx: key for key, idx in vocab.items()}


def main(mode):
    params = tf.contrib.training.HParams(**Config.model.to_dict())

    run_config = tf.contrib.learn.RunConfig(
            model_dir=Config.train.model_dir,
            save_checkpoints_steps=Config.train.save_checkpoints_steps)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=mode,
        hparams=params
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode (train/test/train_and_evaluate)')
    args = parser.parse_args()

    tf.logging._logger.setLevel(logging.INFO)

    Config(args.config)
    print("Config: ", Config)

    main(args.mode)
