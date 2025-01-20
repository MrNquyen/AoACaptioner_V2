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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
from tqdm import tqdm

import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
import skimage.io
import sys

# root_path = 'F:\\UNIVERSITY\\NCKH\\NCKH_V2\\baseline_final\\AoANet_Official\\AoANet'
root_path = os.getcwd()
print(f'Current direction is: {root_path}')
device = 'cpu' if torch.cpu.is_available() else 'cuda'

from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
from utils import load_json, save_json
import misc.resnet_v2 as ResnetNetModule
import misc


def main(params):
  resnet = misc.resnet_v2.ResnetNetModule(params['model'])
  net = resnet.get_model()
  my_resnet = myResnet(net)
  my_resnet.cuda(5)
  my_resnet.eval()

  imgs = load_json(params['input_json'])
  imgs = imgs['images']
  N = len(imgs)

  seed(123) # make reproducible
  output_dir = params['save_dir']+f'/{params['model']}'
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  dir_fc = output_dir+f'/{params['model']}_fc'
  dir_att = output_dir+f'/{params['model']}_att'
  if not os.path.isdir(dir_fc):
    os.mkdir(dir_fc)
  if not os.path.isdir(dir_att):
    os.mkdir(dir_att)

  for i,img in tqdm(enumerate(imgs)):
    img_id = str(img['imgid'])
    if f'{img_id}.npy' in os.listdir(dir_fc) and f'{img_id}.npz' in os.listdir(dir_att):
      continue
    # load the image
    I = skimage.io.imread(os.path.join(params['images_root'], img['file_path'], img['file_name']))
    # handle grayscale input images
    if len(I.shape) == 2:
      I = I[:,:,np.newaxis]
      I = np.concatenate((I,I,I), axis=2)

    I = I.astype('float32')/255.0
    if I.shape == 3:
      # Do nothing, already in (C, H, W) format
      print("Tensor has 3 dimensions, no change needed.")
    elif I.shape == 4:
      # If the tensor has 4 dimensions (H, W, C, Batch), convert to 3D (C, H, W)
      # Here, we select the first image from the batch (index 0)
      I = I[:, :, :, 0]  # Select the first image in the batch (or any image you want)
    I = I.transpose([2,0,1])
    if I.shape[0] != 3:
      I = I[:3, :, :]
    I = torch.from_numpy(I).cuda(5)
    try:
      I = preprocess(I)
    except:
      print(I.shape)
      break
    with torch.no_grad():
      tmp_fc, tmp_att = my_resnet(I, params['att_size'])
    
    # write to pkl
    fc_save_path = os.path.join(dir_fc, img_id)
    att_save_path = os.path.join(dir_att, img_id)
    np.save(fc_save_path, tmp_fc.data.cpu().float().numpy())
    np.savez_compressed(att_save_path, feat=tmp_att.data.cpu().float().numpy())

    if i % 1000 == 0:
      print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
  print('wrote ', output_dir)

if __name__ == "__main__":

  save_dir = f'{root_path}\\data'
  para = {
    # Input json
    'input_json_path': f'{save_dir}\\save_annotation\\data_raw.json',
    # Ouput dir
    'save_dir': f'{save_dir}',
    
    # Image_root
    'images_root': f'{save_dir}\\images',
    # Image_root
    'att_size': 14,
    # Image_root
    'model': 'resnet101',
  }

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', default=para['input_json_path'], help='input json file to process into hdf5')
  parser.add_argument('--save_dir', default='data', help='output h5 file')

  # options
  parser.add_argument('--images_root', default=para['images_root'], help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--att_size', default=para['att_size'], type=int, help='14x14 or 7x7')
  parser.add_argument('--model', default=para['model'], type=str, help='resnet101, resnet152')
  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
