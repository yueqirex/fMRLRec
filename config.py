import torch
import numpy as np
import argparse
import random

from datasets import *
from model import *

RAW_DATASET_ROOT_FOLDER = 'data'
RAW_IMAGE_ROOT_FOLDER = 'photos'
EXPERIMENT_ROOT = 'experiments'

STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'

PROJECT_NAME = 'recsys'


def set_template(args):
    args.min_uc = 5
    args.min_sc = 5
    args.split = 'leave_one_out'
    
    if args.dataset_code == None:
        print('******************** Dataset Selection ********************')
        dataset_code = {'b': 'beauty', 'c': 'clothing', 's': 'sports', 't': 'toys'}
        
        args.dataset_code = dataset_code[input(
            'Input ' + ', '.join([k+' for '+v for k, v in dataset_code.items()]) + ': ')]
    
    batch = 32
    args.train_batch_size = batch
    args.val_batch_size = args.train_batch_size * 2
    args.test_batch_size = args.train_batch_size * 2

    if torch.cuda.is_available(): args.device = 'cuda'
    else: args.device = 'cpu'
    args.optimizer = 'AdamW'
    if args.lr is None: args.lr = 1e-4
    if args.weight_decay is None: args.weight_decay = 0.01
    if args.lru_dropout is None: args.lru_dropout = 0.1
    if args.lru_attn_dropout is None: args.lru_attn_dropout = 0.1

    args.metric_ks = [5, 10, 20, 50]
    args.lru_head_size = None
    args.mrl_hidden_sizes = [int(size) for size in args.mrl_hidden_sizes.split(',')]

    assert not (args.test_best_model and args.test_mrl)


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default=None)
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=2)
parser.add_argument('--min_sc', type=int, default=1)
parser.add_argument('--split', type=str, default='leave_one_out')
parser.add_argument('--seed', type=int, default=42)

################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
# optimizer & lr#
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=100)
# evaluation #
parser.add_argument('--eval_mode', type=str, default=None)
parser.add_argument('--val_strategy', type=str, default='epoch', choices=['epoch', 'iteration'])
parser.add_argument('--val_iterations', type=int, default=1000)  # for iteration val_strategy
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20, 50])
parser.add_argument('--best_metric', type=str, default='NDCG@10')
parser.add_argument('--use_wandb', type=bool, default=False)

################
# Model
################
parser.add_argument('--model_code', type=str, default='lru')
# LRU specs, used for other models as well #
parser.add_argument('--lru_max_len', type=int, default=50)
parser.add_argument('--lru_num_blocks', type=int, default=1)
parser.add_argument('--lru_dropout', type=float, default=None)
parser.add_argument('--lru_attn_dropout', type=float, default=None)
parser.add_argument('--lru_use_bias', type=bool, default=True)
parser.add_argument('--lru_r_min', type=float, default=0.0)
parser.add_argument('--lru_r_max', type=float, default=0.1)
# language
parser.add_argument('--use_language_encoder', type=bool, default=True)
parser.add_argument('--language_encoder', type=str, default='BAAI/bge-large-en-v1.5')
parser.add_argument('--item_attributes', type=list, default=['title', 'price', 'brand', 'categories'])
# image
parser.add_argument('--use_img_encoder', type=bool, default=True)
parser.add_argument('--img_encoder', type=str, default='google/siglip-large-patch16-256')
# mrl
parser.add_argument('--mrl_hidden_sizes', type=str, default="8,16,32,64,128,256,512,1024")
parser.add_argument('--test_mrl', type=bool, default=False)
parser.add_argument('--test_best_model', type=bool, default=False)
parser.add_argument('--ckpt_path', type=str, default=None)

parser.add_argument('--freeze_embed', type=bool, default=True)

################

args = parser.parse_args()