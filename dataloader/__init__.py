from datasets import dataset_factory
from .lru import *


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = LRUDataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return dataset, train, val, test
