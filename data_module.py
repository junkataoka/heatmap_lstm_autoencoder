#%%
import torch
from torch.utils.data import Dataset, DataLoader 
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import os
from glob import glob

class timeseries(Dataset):
  def __init__(self, root, input_file, target_file):
    self.input_path = glob(os.path.join(root, input_file, "/*"))
    self.target_path = os.path.join(root, target_file, "/*")
    self.len = len(self.input_path)

  def __getitem__(self, idx):
    input_tensor = torch.load(self.input[idx])
    target_tensor = torch.load(self.target[idx])

    return input_tensor, target_tensor

  def __len__(self):
    return self.len
  
class TSDataModule(pl.LightningDataModule):
  def __init__(self, opt, root: str, src_input_file, src_target_file, tar_input_file, tar_target_file, batch_size):
    super(TSDataModule, self).__init__()
    self.opt = opt
    self.root = root
    self.src_input_file = src_input_file
    self.src_target_file = src_target_file
    self.tar_input_file = tar_input_file
    self.tar_target_file = tar_target_file
    self.batch_size = batch_size

  def setup(self, stage=None):

    self.src_data = timeseries(self.root, self.src_input_file, self.src_target_file)
    self.tar_data = timeseries(self.root, self.tar_input_file, self.tar_target_file)
    # train_set_size = int(len(self.data) * 0.8)
    # val_set_size = len(self.data) - train_set_size 
    # self.train_set, self.val_set = torch.utils.data.random_split(self.data, [train_set_size, val_set_size])

  def train_dataloader(self):

    if self.opt.is_distributed:
      print("Ditributed sampler")
      src_train_sampler = DistributedSampler(self.src_data, shuffle=True, drop_last=True)
      tar_train_sampler = DistributedSampler(self.tar_data, shuffle=True, drop_last=True)

    else: 
      src_train_sampler = None
      tar_train_sampler = None

    src_dataloader = DataLoader(self.src_train_set, batch_size=self.batch_size, shuffle=src_train_sampler is None, sampler=src_train_sampler)
    tar_dataloader = DataLoader(self.tar_train_set, batch_size=self.batch_size, shuffle=tar_train_sampler is None, sampler=tar_train_sampler)

    loaders = {"src": src_dataloader, "tar": tar_dataloader}

    return loaders

  def val_dataloader(self):
    if self.opt.is_distributed:
      print("Ditributed sampler")
      src_train_sampler = DistributedSampler(self.src_data, shuffle=True, drop_last=True)
      tar_train_sampler = DistributedSampler(self.tar_data, shuffle=True, drop_last=True)

    else: 
      src_train_sampler = None
      tar_train_sampler = None

    src_dataloader = DataLoader(self.src_train_set, batch_size=self.batch_size, shuffle=src_train_sampler is None, sampler=src_train_sampler)
    tar_dataloader = DataLoader(self.tar_train_set, batch_size=self.batch_size, shuffle=tar_train_sampler is None, sampler=tar_train_sampler)

    loaders = {"src": src_dataloader, "tar": tar_dataloader}

    return loaders

  def test_dataloader(self):

    if self.opt.is_distributed:
      print("Ditributed sampler")
      src_train_sampler = DistributedSampler(self.src_data, shuffle=True, drop_last=True)
      tar_train_sampler = DistributedSampler(self.tar_data, shuffle=True, drop_last=True)

    else: 
      src_train_sampler = None
      tar_train_sampler = None

    src_dataloader = DataLoader(self.src_train_set, batch_size=self.batch_size, shuffle=src_train_sampler is None, sampler=src_train_sampler)
    tar_dataloader = DataLoader(self.tar_train_set, batch_size=self.batch_size, shuffle=tar_train_sampler is None, sampler=tar_train_sampler)

    loaders = {"src": src_dataloader, "tar": tar_dataloader}

    return loaders