from config import *

import json
from collections import defaultdict
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path
import pdb

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim as optim

import matplotlib.pyplot as plt


def plot_mrl_results(args):
    # parse plots save path
    export_root = EXPERIMENT_ROOT + '/LN_IMAGE/' + args.model_code
    dirs = os.listdir(export_root)
    dirs.remove('best_results.csv')
    for folder in dirs:
        args.dataset_code = folder.split('_')[0]
        exp_folder = os.path.join(export_root, folder)  
        print(f'===NOTE: Evaluating {exp_folder}...')
        plot_path = os.path.join(exp_folder, 'plots')
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)
        else:
            print('=== NOTE: already evaluated, skip===')
            continue

        mrl_metrics = {}
        with open(os.path.join(exp_folder, f'test_mrl_metrics.json'), 'r') as f: metrics_data = json.load(f)
        for s in args.mrl_hidden_sizes:
            metrics = metrics_data[str(s)]
            mrl_metrics[s] = {m:{k:metrics[m+'@'+str(k)] for k in args.metric_ks} for m in ['Recall', 'NDCG', 'MRR']}
            
        for m in ['Recall', 'NDCG', 'MRR']:
            for k in args.metric_ks:
                ls = [mrl_metrics[s][m][k] for s in args.mrl_hidden_sizes]
                plt.plot(args.mrl_hidden_sizes, ls, marker='o', label=f'{m}@{k}')
                plt.xlabel('Model size')
                plt.ylabel(f'{m}@{k}')
                plt.title(f'MRL plot for {args.dataset_code} {m}@{k}')
                plt.grid(True)
                plt.legend()
            plt.savefig(os.path.join(plot_path, f'MRL_plot_{args.dataset_code}_{m}.png'))
            plt.clf()
            plt.close()
            print(f'===NOTE: MRL plot saved for {args.dataset_code} {m} successfully!===')

        for m in ['Recall', 'NDCG', 'MRR']:
            for k in args.metric_ks:
                plt.figure()
                ls = [mrl_metrics[s][m][k] for s in args.mrl_hidden_sizes]
                plt.plot(args.mrl_hidden_sizes, ls, marker='o', label=f'{m}@{k}')
                plt.xlabel('Model size')
                plt.ylabel(f'{m}@{k}')
                plt.title(f'MRL plot for {args.dataset_code} {m}@{k}')
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(plot_path, f'MRL_plot_{args.dataset_code}_{m}@{k}.png'))
                plt.clf()
                plt.close()
                print(f'===NOTE: MRL plot saved for {args.dataset_code} {m}@{k} successfully!===')


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                         for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def absolute_recall_mrr_ndcg_for_ks(scores, labels, ks):
    metrics = {}
    labels = F.one_hot(labels, num_classes=scores.size(1))
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)

    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(
                labels.device), labels.sum(1).float())).mean().cpu().item()
        
        metrics['MRR@%d' % k] = \
            (hits / torch.arange(1, k+1).unsqueeze(0).to(
                labels.device)).sum(1).mean().cpu().item()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                             for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)