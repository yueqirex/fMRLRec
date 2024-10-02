import os
import yaml
import datetime
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import set_seed
from datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *


def train(args, export_root=None):
    set_seed(args.seed)
    if args.use_language_encoder and args.use_img_encoder:
        branch = 'LN_IMAGE'
    elif args.use_language_encoder or args.use_img_encoder:
        branch = 'LN' if args.use_language_encoder else 'IMAGE'
    else:
        branch = 'ID'
    if export_root == None:
        current_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        export_root = EXPERIMENT_ROOT + f'/{branch}' + '/' + args.model_code + '/' + args.dataset_code + \
            '_dp' + str(args.lru_dropout) + '_rmin' + str(args.lru_r_min) + '_' + current_date_time

    dataset, train, val, test = dataloader_factory(args)
    args.dataset = dataset
    
    if args.test_mrl:
        for i, s in enumerate(args.mrl_hidden_sizes):
            print(f'Testing MRL size {s}...')
            args_ = copy(args)
            args_.mrl_hidden_sizes = args.mrl_hidden_sizes[:i+1]
            # args_.language_embedding_size = args_.mrl_hidden_sizes[-1]
            model = LRU(args_)
            trainer = LRUTrainer(args_, model, train, val, test, export_root, args.use_wandb)
            trainer.test()
        plot_mrl_results(args)
        
    elif args.test_best_model:
        model = LRU(args)
        trainer = LRUTrainer(args, model, train, val, test, export_root, args.use_wandb)
        trainer.test()
    else:
        model = LRU(args)
        trainer = LRUTrainer(args, model, train, val, test, export_root, args.use_wandb)
        # save args
        args_dict = vars(copy(args))
        if 'dataset' in args_dict: del args_dict['dataset']
        with open(export_root + '/args.yaml', 'w') as f: yaml.dump(args_dict, f)
        # exit()
        trainer.train()
        trainer.test()


if __name__ == "__main__":
    set_template(args)
    train(args)