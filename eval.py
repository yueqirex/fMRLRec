import os
import yaml
import datetime
import pdb
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import set_seed
from datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *
from trainer.utils import *


def train(args, export_root=None):
    set_seed(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/LN_IMAGE/' + args.model_code
    dirs = os.listdir(export_root)
    if 'best_results.csv' in dirs: dirs.remove('best_results.csv')
    print(dirs)
    for folder in dirs:
        print(f'Evaluation {folder}...')

        if 'test_mrl_metrics.json' in os.listdir(export_root+'/'+folder):
            print('Already evaluated, skipping...')
            continue

        args.dataset_code = folder.split('_')[0]
        dataset, train, val, test = dataloader_factory(args)
        args.dataset = dataset
    
        model = LRU(args)
        trainer = LRUTrainer(args, model, train, val, test, export_root+'/'+folder, args.use_wandb)
        trainer.test_mrl()


def plot(args): plot_mrl_results(args)


if __name__ == "__main__":
    set_template(args)
    if args.eval_mode == 'plot':
        plot(args)
    else:
        train(args)