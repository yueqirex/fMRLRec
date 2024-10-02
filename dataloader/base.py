from abc import *
import numpy as np


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.save_folder = dataset.data_dict_path

        self.rng = np.random
        self.max_len = args.lru_max_len

        self.train = dataset.train
        self.val = dataset.val
        self.test = dataset.test
        self.meta_dict = dataset.meta_dict
        
        self.user_count = dataset.user_count
        self.item_count = dataset.item_count
        args.num_users = self.user_count
        args.num_items = self.item_count

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass