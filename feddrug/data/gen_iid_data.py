"""Partitioned version of ESOL dataset."""

from typing import List, Tuple, cast
from core.train_utils import init_featurizer, mkdir_p
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from core.dataset import dataset_loader, collate_molgraphs, BalancedBatchSampler
from core.dataset import dataset_loader
XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]

def load_dataset(args, num_partitions=3):
    train_set, val_set, test_set = dataset_loader(args)
    num_samples = len(train_set)
    num_samples_per_device, _ = divmod(num_samples, num_partitions)
    l = [num_samples_per_device]*num_partitions
    l[-1] += _
    train_partitions = random_split(train_set, l)
    return train_partitions, val_set, test_set

def load_fed_data(args) -> PartitionedDataset:
    """Create partitioned version of CIFAR-10."""
    list_of_dataloaders = []
    num_partitions = args['num_clients']
    train_partions, val_set, test_set = load_dataset(args, num_partitions)
    print(len(train_partions[0]))
    for xy_train in train_partions:
        sampler = None
        shuffle = None
        train_dl = DataLoader(dataset=xy_train, sampler=sampler, shuffle=shuffle, batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
        test_dl = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
        print(len(train_dl))
        list_of_dataloaders.append((train_dl, test_dl))

    return list_of_dataloaders, test_dl