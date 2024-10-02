from .amazon import *


DATASETS = {
    BeautyDataset.code(): BeautyDataset,
    ClothingDataset.code(): ClothingDataset,
    SportsDataset.code(): SportsDataset,
    ToysDataset.code(): ToysDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)