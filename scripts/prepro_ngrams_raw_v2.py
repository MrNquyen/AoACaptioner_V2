"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

import os
import json
import argparse
import sys

from collections import defaultdict

# root_path = 'F:\\UNIVERSITY\\NCKH\\NCKH_V2\\baseline_final\\AoANet_Official\\AoANet'
root_path = os.getcwd()
print(f'Curent direction is: {root_path}')
sys.path.append(root_path)

# import misc.utils as utils
from utils import load_json, load_h5, save_h5, save_json, save_pickle
from scripts.bpe_encoding_raw import load_bpe, tokenize

def precook(s, n=4, out=False):
  """
  Takes a string as input and returns an object that can be given to
  either cook_refs or cook_test. This is optional: cook_refs and cook_test
  can take string arguments as well.
  :param s: string : sentence to be converted into ngrams
  :param n: int    : number of ngrams for which representation is calculated
  :return: term frequency vector for occuring ngrams
  """
  words = s.split()
  counts = defaultdict(int)
  for k in range(1,n+1):
    for i in range(len(words)-k+1):
      ngram = tuple(words[i:i+k])
      counts[ngram] += 1
  return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
  crefs = []
  for ref in refs:
    # ref is a list of 5 captions
    crefs.append(cook_refs(ref))
  return crefs

def compute_doc_freq(crefs):
  '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
      :return: None
  '''
  document_frequency = defaultdict(float)
  for refs in crefs:
    # refs, k ref captions of one image
    for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
      document_frequency[ngram] += 1
      # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
  return document_frequency


def build_dict(imgs, wtoi, tokenizer=None):
  count_imgs = 0

  refs_words = []
  refs_idxs = []
  for img in imgs:
    if img['split'] == 'train':
      ref_words = []
      ref_idxs = []
      for sent in img['sentences']:
        if tokenizer!=None:
          caption = ' '.join(sent['tokens'])
          encoded = tokenize(caption, tokenizer)
          decoded = tokenizer.decode(encoded.ids)
          sent['tokens'] = decoded.split(" ")
        tmp_tokens = sent['tokens'] + ['<eos>']
        tmp_tokens = [_ if _ in wtoi else '<unk>' for _ in tmp_tokens]
        ref_words.append(' '.join(tmp_tokens))
        ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
      refs_words.append(ref_words)
      refs_idxs.append(ref_idxs)
      count_imgs += 1
  print('total imgs:', count_imgs)

  ngram_words = compute_doc_freq(create_crefs(refs_words))
  ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
  return ngram_words, ngram_idxs, count_imgs


def main(params):
  imgs = load_json(params['input_json'])
  imgs = imgs['images']

  dict_json = load_json(params['dict_json'])
  itow = dict_json['ix_to_word']
  wtoi = {w:i for i,w in itow.items()}

  # Load bpe
  tokenizer = None
  if 'tokenizer_path' in params:
    tokenizer = load_bpe(params['tokenizer_path'])

  ngram_words, ngram_idxs, ref_len = build_dict(imgs, wtoi, tokenizer)

  save_pickle({'document_frequency': ngram_words, 'ref_len': ref_len}, params['output_pkl']+'-words.p')
  save_pickle({'document_frequency': ngram_idxs, 'ref_len': ref_len}, params['output_pkl']+'-idxs.p')
if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # Para
  save_dir = f'{root_path}\\data'
  para = {
    # Input json
    'input_json_path': f'{save_dir}\\save_annotation\\data_raw.json',
    # Dict json
    'dict_json_path': f'{save_dir}\\save_input\\data_raw_bpe.json',
    # Tokenizer json
    'tokenizer_json_path': f'{save_dir}\\save_tokenizer\\tokenizer_raw_bpe.json',
    # Ouput json
    'output_pkl': f'{save_dir}\\save_input\\data_raw_ngram_pkl',
  }

  # input json
  parser.add_argument('--input_json', default=para['input_json_path'], help='input json file to process into hdf5')
  parser.add_argument('--dict_json', default=para['dict_json_path'], help='output json file')
  parser.add_argument('--tokenizer_path', default=para['tokenizer_json_path'], help='bpe tokenizer path')
  parser.add_argument('--output_pkl', default=para['output_pkl'], help='output pickle file')
  # parser.add_argument('--split', default=para['split'], help='test, val, train, all')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  main(params)
