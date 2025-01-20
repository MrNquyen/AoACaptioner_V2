import os
import json
import argparse
import string
import h5py
import torch
import skimage.io
import sys

import numpy as np
import torchvision.models as models

from random import shuffle, seed
from PIL import Image
from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

# root_path = 'F:\\UNIVERSITY\\NCKH\\NCKH_V2\\baseline_final\\AoANet_Official\\AoANet'
root_path = './'
sys.path.append(root_path)
os.chdir(root_path)

from utils import load_json, save_json

def create_bpe_trainer(corpus):
  # Define constants
  unk_token = "UNK"
  spl_tokens = ["<unk>", "<eos>", "<\eos>"]

  tokenizer = Tokenizer(BPE())
  trainer = trainers.BpeTrainer(special_tokens = spl_tokens)
  tokenizer.pre_tokenizer = Whitespace()

  # Training
  tokenizer.train_from_iterator(corpus, trainer=trainer)

  # Return
  return tokenizer


def save_bpe(tokenizer, save_path):
  # Save
  tokenizer.save(save_path)


def load_bpe(save_path):
  load_tokenizer = Tokenizer.from_file(save_path)
  return load_tokenizer


def tokenize(sent, tokenizer):
  encoded_ids = tokenizer.encode(sent)
  return encoded_ids


def main(params):
  # Load data
  input_json_path = params['input_json']
  data_raw = load_json(input_json_path)
  data_raw = data_raw['images']

  # Captions
  captions = [
    item['sentences'][0]['raw']
    for item in data_raw
  ]

  # BPE
  bpe = create_bpe_trainer(captions)
  save_bpe(bpe, params['save_tokenizer_path'])



if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', default='data/save_annotation/data_raw.json', help='input json file to process into hdf5')
  parser.add_argument('--save_tokenizer_path', default='data/save_tokenizer/tokenizer_raw_bpe.json', help='Save tokenizer Path')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
