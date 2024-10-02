import os
import copy
import gzip
import torch
from tqdm import tqdm

import json
import numpy as np
import pickle

from PIL import Image
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer

from .base import *


class AmazonDataset(AbstractDataset):
    def load_datamap(self, path):
        with open(path.joinpath('datamaps.json'), "r") as f:
            datamaps = json.load(f)
        
        user2id = datamaps['user2id']
        user2id = {k: int(v) for k, v in user2id.items()}
        
        item2id = datamaps['item2id']
        item2id = {k: int(v) for k, v in item2id.items()}

        id2item = datamaps['id2item']
        id2item = {int(k): v for k, v in id2item.items()}
        
        return user2id, item2id, id2item
    
    def load_item2image(self, path):
        with open(path.joinpath('item2img_dict.pkl'), "rb") as f:
            item2img = pickle.load(f)
        return item2img

    def load_ratings(self, path):
        self.generate_embeddings()
        self.generate_img_embeddings()

        lines = []
        with open(path.joinpath('sequential_data.txt'), 'r') as f:
            for line in f:
                lines.append(line.rstrip('\n'))
        
        user_items = {}
        for line in lines:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[int(user)] = items

        return user_items
    
    def load_metadict(self, path, item2id):
        def parse(path):
            g = gzip.open(path, 'r')
            for l in g:
                yield eval(l)   
        
        meta_dict = {}
        for meta in parse(path.joinpath('meta.json.gz')):
            if meta['asin'] in item2id:
                meta_dict[item2id[meta['asin']]] = meta
            else:
                continue

        return meta_dict
    
    def generate_embeddings(self):
        def process_attr(attr):
            if isinstance(attr, list):
                attr = ', '.join(attr[0])
            elif isinstance(attr, float) or isinstance(attr, int):
                attr = str(attr)
            return attr.strip()
        
        if self._get_preprocessed_embeddings_path().is_file():
            print('Item embeddings already processed...')
            return

        meta_dict = self.load_metadict(self.meta_dict_path, self.item2id)
        all_text = []
        for item_id in self.id2item.keys():
            text = ''
            if 'title' in meta_dict[item_id]:
                text += 'Title: ' + process_attr(meta_dict[item_id]['title']) + '\n'
            if 'price' in meta_dict[item_id]:
                text +=  'Price: ' + process_attr(meta_dict[item_id]['price']) + '\n'
            if 'brand' in meta_dict[item_id]:
                text +=  'Brand: ' + process_attr(meta_dict[item_id]['brand']) + '\n'
            if 'categories' in meta_dict[item_id]:
                text +=  'Categories: ' + process_attr(meta_dict[item_id]['categories'])
            if len(text) == 0:
                text = 'Untitled Item'
            all_text.append(text)

        model = SentenceTransformer(self.args.language_encoder)
        embeddings = model.encode(all_text, normalize_embeddings=True)
        np.save(self._get_preprocessed_embeddings_path(), embeddings)
        del model, embeddings

    def generate_img_embeddings(self): # siglip image
        def embed_siglip(image_ls, model, bs=256):
            image_features = []
            for i in tqdm(range(0, len(image_ls), bs)):
                with torch.no_grad():
                    inputs = processor(images=image_ls[i:i+bs], return_tensors="pt").to(device)
                    image_embeds = model.get_image_features(**inputs)
                    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                    image_features.append(image_embeds)
            return torch.cat(image_features, dim=0)
        
        if self._get_preprocessed_img_embeddings_path().is_file():
            print(f'Image embeddings already processed...')
            return

        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(self.args.img_encoder).to(device)
        processor = AutoProcessor.from_pretrained(self.args.img_encoder)

        image_ls = []
        for idx in tqdm(self.id2item.keys()):
            f_name = self.item2image[self.id2item[idx]].split('/')[1]
            f_path = self._get_rawimage_root_path().joinpath(
                self.args.dataset_code, f_name)
            if os.path.isfile(f_path):  # this can be inefficient
                image_ls.append(Image.open(f_path).convert("RGB"))
            else:
                raise Exception(f'image not found for item {idx}')

        image_embeddings = embed_siglip(image_ls, model)
        np.save(self._get_preprocessed_img_embeddings_path(), image_embeddings.detach().cpu().numpy())
        del processor, model, image_embeddings

        
class BeautyDataset(AmazonDataset):
    @classmethod
    def code(cls):
        return 'beauty'


class ClothingDataset(AmazonDataset):
    @classmethod
    def code(cls):
        return 'clothing'


class SportsDataset(AmazonDataset):
    @classmethod
    def code(cls):
        return 'sports'


class ToysDataset(AmazonDataset):
    @classmethod
    def code(cls):
        return 'toys'