import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from tutorial_dataset import *
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False,config=None):
        super().__init__()
        self.batch_size = batch_size
        self.config=config
        self.datasets={}
        self.train_dataloader =self._train_dataloader

        # self.val_dataloader = self._val_dataloader

        self.test_dataloader = self._test_dataloader

        self.wrap = wrap

    def prepare_data(self):
        a=1


    def setup(self, stage=None):
        if self.config.dataset_choose=='Circle':
            self.datasets['train']=MyDataset(status='Train')
            self.datasets['validation'] = MyDataset(status='Test')
            self.datasets['test'] = MyDataset(status='Test')
        if self.config.dataset_choose=='Xenium':
            self.datasets['train'] = Xenium_dataset(data_root=self.config.data_root,SR_times=self.config.SR_times,status='Train',gene_num=self.config.gene_num)
            self.datasets['validation'] = Xenium_dataset(data_root=self.config.data_root, SR_times=self.config.SR_times,status='Test', gene_num=self.config.gene_num)
            self.datasets['test'] = Xenium_dataset(data_root=self.config.data_root, SR_times=self.config.SR_times,status='Test', gene_num=self.config.gene_num)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
        a=1

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, num_workers=16, shuffle=True,
                          worker_init_fn=None)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=16,
                          worker_init_fn=None,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=16, worker_init_fn=None, shuffle=shuffle)








