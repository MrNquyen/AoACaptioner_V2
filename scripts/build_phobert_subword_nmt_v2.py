import os
import json
import argparse
from random import shuffle, seed
import string
import h5py
import numpy as np
import torch
import torchvision.models as models
import skimage.io
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import sys
root_path = 'F:\\UNIVERSITY\\NCKH\\NCKH_V2\\baseline_final\\AoANet_Official\\AoANet'
sys.path.append(root_path)

from utils import load_json, save_json, load_image

### FUNCTION ###

def tokenize(text, tokenizer, max_length=64):
  encoded = tokenizer(
    text,
    padding="max_length",
    max_length=max_length,
    truncation=True,
  )
  return encoded

def build_vocab(imgs, params, tokenizer):
  word2idx = tokenizer.get_vocab()
  idx2word = {i:w for w, i in word2idx.items()}
  vocab = list(word2idx.keys())

  # Final Caption
  max_length = params['max_length']
  for _, img in enumerate(imgs):
    img['final_captions'] = []
    for sent in img['sentences']:
      txt = ' '.join(sent['tokens'])
      encoded = tokenize(txt, tokenizer, max_length)
      encoded_tokens = [idx2word.get(token, '<unk>') for token in encoded.input_ids]
      img['final_captions'].append(encoded_tokens)
  return word2idx, idx2word


def encode_captions(imgs, params, wtoi):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  N = len(imgs)
  M = sum(len(img['final_captions']) for img in imgs) # total number of captions

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1
  for i,img in tqdm(enumerate(imgs)):
    n = len(img['final_captions'])
    assert n > 0, 'error: some image has no captions'

    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(img['final_captions']):
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
      caption_counter += 1
      for k,w in enumerate(s):
        if k < max_length:
          Li[j,k] = wtoi[w]
          
    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1
    
    counter += n
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print('encoded captions to array of size ', L.shape)
  return L, label_start_ix, label_end_ix, label_length

def main(params):
  imgs = load_json(params['input_json'])
  imgs = imgs['images']

  captions = [" ".join(img['sentences'][0]['tokens']) for img in imgs]
  # tokenizer = create_bpe_trainer(captions)
  # save_bpe(tokenizer, params['save_tokenizer_path'])
  phobert_save_path = params['save_tokenizer_path']
  tokenizer = AutoTokenizer.from_pretrained(phobert_save_path)
  seed(123) # make reproducible
  
  # create the vocab
  # itow = {i:w for w, i in wtoi.items()} # a 1-indexed vocab translation table
  # vocab = list(wtoi.keys()) # inverse table

  # wtoi = tokenizer.get_vocab()
  wtoi, itow = build_vocab(imgs, params, tokenizer)
  # itow = {i:w for w, i in wtoi.items()} # a 1-indexed vocab translation table
  
  # encode captions in large arrays, ready to ship to hdf5 file
  L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

  # create output h5 file
  N = len(imgs)
  f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
  f_lb.create_dataset("labels", dtype='uint32', data=L)
  f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
  f_lb.close()

  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['images'] = []
  out['tokenizer_path'] = params['save_tokenizer_path']
  for i,img in enumerate(imgs):
    
    jimg = {}
    jimg['split'] = img['split']
    if 'file_name' in img: jimg['file_path'] = os.path.join(img['file_path'], img['file_name']) # copy it over, might need
    if 'cocoid' in img:
      jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)
    elif 'imgid' in img:
      jimg['id'] = img['imgid']
    
    if params['images_root'] != '':
      _img = load_image(os.path.join(params['images_root'], img['file_path'], img['file_name']))
      jimg['width'], jimg['height'] = _img.size

    out['images'].append(jimg)
  
  save_json(params['output_json'], out)
  print('wrote ', params['output_json'])

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # COnfig
  save_dir = f'{root_path}/data'
  para = {
    # Input json
    'input_json_path': f'{save_dir}/save_annotation/data_segmented.json',
    # Ouput json
    'output_json_path': f'{save_dir}/save_input/data_segmented_phobert.json',
    # Output_h5
    'output_h5': f'{save_dir}/save_input/data_segmented_phobert',
    # Image_root
    'images_root': f'{save_dir}/images',
  }
  save_tokenizer_path = 'vinai/phobert-base'

  # input json
  parser.add_argument('--input_json', default=para['input_json_path'], help='input json file to process into hdf5')
  parser.add_argument('--output_json', default=para['output_json_path'], help='output json file')
  parser.add_argument('--save_tokenizer_path', default=save_tokenizer_path, help='output json file')
  parser.add_argument('--output_h5', default=para['output_h5'], help='output h5 file')
  parser.add_argument('--images_root', default=para['images_root'], help='root location in which images are stored, to be prepended to file_path in input json')

  # options
  parser.add_argument('--max_length', default=64, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--symbol_count', default=500, type=int, help='only words that occur more than this number of times will be put in vocab')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)


