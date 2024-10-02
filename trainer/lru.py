from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *
from .base import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import pdb
import numpy as np
import pandas as pd
from abc import *
from pathlib import Path


class LRUTrainer(BaseTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, use_wandb)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        
    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)
        labels = labels.repeat(len(self.args.mrl_hidden_sizes), 1)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, labels = batch
        # pdb.set_trace()
        scores = self.model(seqs)[:, -1, :]
        scores = scores.reshape(
            len(self.args.mrl_hidden_sizes), -1, scores.shape[-1]) #(M, B, N)
        
        scores = scores[-1]  # largest dimension only #(B, N)
        scores[:, 0] = -1e9  # padding
        
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        return metrics
    
    def calculate_metrics_mrl(self, batch):
        seqs, labels = batch
        scores = self.model(seqs)[:, -1, :]
        scores = scores.reshape(
            len(self.args.mrl_hidden_sizes), -1, scores.shape[-1]) #(M, B, N)
        
        all_metrics = []
        for idx in range(len(self.args.mrl_hidden_sizes)):
            cur_scores = scores[idx]  # largest dimension only #(B, N)
            cur_scores[:, 0] = -1e9  # padding
            metrics = absolute_recall_mrr_ndcg_for_ks(cur_scores, labels.view(-1), self.metric_ks)
            all_metrics.append(metrics)
        
        return all_metrics