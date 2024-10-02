from .base import AbstractDataloader

import os
import torch
import random
import pickle
import numpy as np
import torch.utils.data as data_utils


def InfiniteSampling(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampling(self.num_samples))

    def __len__(self):
        return 2 ** 31


class LRUDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

    @classmethod
    def code(cls):
        return 'lru'

    def get_pytorch_dataloaders(self, infinite=False):
        train_loader = self._get_train_loader(infinite=infinite)
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader
    
    def _get_train_dataset(self):
        dataset = LRUTrainDataset(
            self.args, self.train, self.max_len, self.rng)
        return dataset
    
    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = LRUValidDataset(self.args, self.train, self.val, self.max_len, self.rng)
        elif mode == 'test':
            dataset = LRUTestDataset(self.args, self.train, self.val, self.test, self.max_len, self.rng)
        return dataset

    def _get_train_loader(self, infinite=False):
        dataset = self._get_train_dataset()
        if not infinite:
            return data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                         shuffle=True, pin_memory=True, num_workers=self.args.num_workers)
        else:
            return data_utils.DataLoader(dataset, sampler=InfiniteSampler(dataset),
                                         batch_size=self.args.train_batch_size, pin_memory=True, 
                                         num_workers=self.args.num_workers)

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        # make sure shuffling is disabled for the following ranking stage
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')


class LRUTrainDataset(data_utils.Dataset):
    def __init__(self,
                 args,
                 u2seq,
                 max_len,
                 rng,
                 ):
        self.args = args
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        labels = seq[-self.max_len:]
        tokens = seq[:-1][-self.max_len:]

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens

        mask_len = self.max_len - len(labels)
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)


class LRUValidDataset(data_utils.Dataset):
    def __init__(self,
                 args,
                 u2seq,
                 u2answer,
                 max_len,
                 rng,
                 ):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        self.users = sorted(self.u2answer.keys())
        self.max_len = max_len
        self.rng = rng
    
    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(answer)


class LRUTestDataset(data_utils.Dataset):
    def __init__(self,
                 args,
                 u2seq,
                 u2val,
                 u2answer,
                 max_len,
                 rng,
                 ):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        self.users = sorted(self.u2answer.keys())
        self.max_len = max_len
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]
        answer = self.u2answer[user]

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(answer)