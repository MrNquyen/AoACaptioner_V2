from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os

import opts_aoa_raw as opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import torch

# Set sys path
# root_path = 'F:\\UNIVERSITY\\NCKH\\NCKH_V2\\baseline_final\\AoANet_Official\\AoANet'
root_path = os.getcwd()
print(f'Current direction is: {root_path}')
sys.path.append(root_path)

import misc.utils as utils_misc
from utils import save_json

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Input arguments and options
parser = argparse.ArgumentParser()

# Para
root_path = 'F:\\UNIVERSITY\\NCKH\\NCKH_V2\\baseline_final\\AoANet_Official\\AoANet'
save_model_path = f'{root_path}\\data\\ckpt\\aoa'

para = {
  'model_path': f'{save_model_path}\\model-best.pth',
  'cnn_model': 'resnet101',
  'infos_path': f'{save_model_path}\\infos_raw-best.pkl',
}

# Input paths
parser.add_argument('--model_path', type=str, default=para['model_path'],
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default=para['cnn_model'],
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default=para['infos_path'],
                help='path to infos to evaluate')
opts.add_eval_options(parser)

opt = parser.parse_args()
print('-'*20)
print(f'Split is {opt.split}')
print('-'*20)
# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils_misc.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model_path))
model.to(device)
model.eval()
crit = utils_misc.LanguageModelCriterion()

# Logging
print(f'Evaluate on {opt.split} set')


# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
opt.datset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

save_eval_dir = opt.save_eval_dir
if opt.dump_json == 1:
    # dump the json

    save_path = f'{save_eval_dir}/vis_best.json'
    save_json(save_path, split_predictions)
