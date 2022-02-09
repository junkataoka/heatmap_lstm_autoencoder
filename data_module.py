#%%
import torch
from torch.utils.data import Dataset, DataLoader 
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import os

class timeseries(Dataset):
  def __init__(self, root, input_file, target_file):
    self.input = torch.load(os.path.join(root, input_file)).float()
    self.target = torch.load(os.path.join(root, target_file)).float()
    self.len = self.input.shape[0]

  def __getitem__(self, idx):
    input_tensor = self.input[idx]
    target_tensor = self.target[idx]

    return input_tensor, target_tensor

  def __len__(self):
    return self.len
  
class TSDataModule(pl.LightningDataModule):
  def __init__(self, opt, root: str, input_file, target_file, batch_size):
    super(TSDataModule, self).__init__()
    self.opt = opt
    self.root = root
    self.input_file = input_file
    self.target_file = target_file
    self.batch_size = batch_size

  def setup(self, stage=None):


    self.data = timeseries(self.root, self.input_file, self.target_file)
    train_set_size = int(len(self.data) * 0.8)
    val_set_size = len(self.data) - train_set_size 
    self.train_set, self.val_set = torch.utils.data.random_split(self.data, [train_set_size, val_set_size])

  def train_dataloader(self):

    if self.opt.is_distributed:
      print("Ditributed sampler")
      train_sampler = DistributedSampler(self.train_set, shuffle=True, drop_last=True)
    else: train_sampler = None

    return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=train_sampler is None, sampler=train_sampler)

  def val_dataloader(self):
    if self.opt.is_distributed:
      val_sampler = DistributedSampler(self.val_set, shuffle=True, drop_last=True)
    else: val_sampler = None

    return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=val_sampler is None, sampler=val_sampler)

  def test_dataloader(self):
    if self.opt.is_distributed:
      test_sampler = DistributedSampler(self.data, shuffle=True, drop_last=True)
    else: test_sampler = None

    return DataLoader(self.data, batch_size=self.batch_size, shuffle=test_sampler is None, sampler=test_sampler)

  