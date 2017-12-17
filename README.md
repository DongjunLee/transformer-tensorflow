# Dynamic Memory Network

TensorFlow implementation of [Attention Is All You Need](https://arxiv.org/abs/1706.03762). (2017. 6)

![images](images/transformer-architecture.jpg)


## Requirements

- Python 3.6
- TensorFlow 1.4
- hb-config
- nltk
- tqdm


## Features

- Using Higher-APIs in TensorFlow
	- [Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
	- [Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment)
	- [Dataset](https://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset)

## Todo

- Implements Multi-Head num_heads > 1 case (split -> ... -> concat)
- Implements Multi-Head masked opt


## Config

example: check_tiny.yml

```yml
data:
  processed_path: 'tiny_processed_data'
  max_seq_length: 20
  word_threshold: 1

  PAD_ID: 0
  UNK_ID: 1
  START_ID: 2
  EOS_ID: 3

model:
  num_layers: 1
  model_dim: 16
  num_heads: 1
  linear_key_dim: 10
  linear_value_dim: 10
  ffn_dim: 16
  dropout: 0.2

train:
  batch_size: 2
  learning_rate: 0.00001
  train_steps: 50000
  model_dir: 'logs/check_tiny'
  save_checkpoints_steps: 1000
  check_hook_n_iter: 100
  min_eval_frequency: 100
  optimizer: 'Adam'  ('Adagrad', 'Adam', 'Ftrl', 'Momentum', 'RMSProp', 'SGD')

eval:
  batch_size: -1   (Using all test data)
```

## Usage

Install requirements.

```pip install -r requirements.txt```

Then, start train and evalueate model
```
python main.py --config check-tiny --mode train_and_evaluate
```

### Tensorboar

```tensorboard --logdir logs```

- check-tiny example

![images](images/check-tiny-loss.png)


## Reference

- [hb-research/notes - Attention Is All You Need](https://github.com/hb-research/notes)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017. 6) by A Vaswani (Google Brain Team)
- [tensor2tensor](https://github.com/tensorflow/tensor2tensor) - A library for generalized sequence to sequence models (official code)
