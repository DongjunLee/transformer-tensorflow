#!/bin/bash

BASE_PATH="data/"
RAW_DATA_PATH="$BASE_PATH/iwslt15_en-vi/"

mkdir $RAW_DATA_PATH

wget -q --show-progress https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en
wget -q --show-progress https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi

mv train.en ${RAW_DATA_PATH}/train.enc
mv train.vi ${RAW_DATA_PATH}/train.dec

wget -q --show-progress https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en
wget -q --show-progress https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi

mv tst2013.en ${RAW_DATA_PATH}/test.enc
mv tst2013.vi ${RAW_DATA_PATH}/test.dec
