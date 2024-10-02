import pickle
import shutil
import tempfile
import os
from pathlib import Path
import gzip
from abc import *
from config import *

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dict_path = Path(RAW_DATASET_ROOT_FOLDER).joinpath(self.code())
        self.meta_dict_path = Path(RAW_DATASET_ROOT_FOLDER).joinpath(self.code())

        print('Loading meta data...')
        self.user2id, self.item2id, self.id2item = self.load_datamap(self.data_dict_path)
        self.item2image = self.load_item2image(self.data_dict_path)
        self.meta_dict = self.load_metadict(self.meta_dict_path, self.item2id)

        print('Loading ratings...')
        self.user_items = self.load_ratings(self.data_dict_path)
        self.user_count = len(self.user_items)
        self.item_count = len(set().union(*self.user_items.values()))

        self.train = {k: v[:-2] for k, v in self.user_items.items()}
        self.val = {k: [v[-2]] for k, v in self.user_items.items()}
        self.test = {k: [v[-1]] for k, v in self.user_items.items()}

    @classmethod
    def raw_code(cls):
        return cls.code()
    
    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def load_ratings(self):
        pass

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)
    
    def _get_rawimage_root_path(self):
        return Path(RAW_IMAGE_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_embeddings_path(self):
        folder = self._get_rawdata_folder_path()
        return folder.joinpath(
            self.args.language_encoder.split('/')[-1].lower() + '_embeddings.npy')
    
    def _get_preprocessed_img_embeddings_path(self):
        folder = self._get_rawdata_folder_path()
        return folder.joinpath(
            self.args.img_encoder.split('/')[-1].lower() + '_img_embeddings.npy')