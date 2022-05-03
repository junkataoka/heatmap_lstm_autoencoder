#%%
from random import triangular
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import os
from glob import glob
from torchvision import transforms

class timeseries(Dataset):
  def __init__(self, root, input_file, target_file, transform=None):
    self.input = torch.load(os.path.join(root, input_file)).float()
    self.target = torch.load(os.path.join(root, target_file)).float()
    self.len = self.input.shape[0]
    self.transform = transform

  def __getitem__(self, idx):
    input_tensor = self.input[idx]
    if self.transform:
      t, c, h, w = input_tensor.shape
      for i in range(t):
        input_tensor[i] = self.transform(input_tensor[i])

    target_tensor =self.target[idx]


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


    self.train_transform = transforms.Compose([
      # transforms.Pad(2),
      transforms.Resize(50)
      # transforms.RandomCrop(50),
      # transforms.RandomHorizontalFlip(),
      # transforms.RandomVerticalFlip()
    ])

    self.val_transform = transforms.Compose([
      # transforms.Pad(2),
      transforms.Resize(50)
      # transforms.Pad(2),
      # transforms.CenterCrop(50),
    ])

    self.src_train_data = timeseries(self.root, self.src_input_file, self.src_target_file, transform=self.train_transform)
    self.tar_train_data = timeseries(self.root, self.tar_input_file, self.tar_target_file, transform=self.train_transform)
    self.src_val_data = timeseries(self.root, self.src_input_file, self.src_target_file, transform=self.val_transform)
    self.tar_val_data = timeseries(self.root, self.tar_input_file, self.tar_target_file, transform=self.val_transform)
    # train_set_size = int(len(self.data) * 0.8)
    # val_set_size = len(self.data) - train_set_size
    # self.train_set, self.val_set = torch.utils.data.random_split(self.data, [train_set_size, val_set_size])

  def train_dataloader(self):

    if self.opt.is_distributed:
      print("Ditributed sampler")
      src_train_sampler = DistributedSampler(self.src_train_data, shuffle=True, drop_last=True)
      tar_train_sampler = DistributedSampler(self.tar_train_data, shuffle=True, drop_last=True)

    else:
      src_train_sampler = None
      tar_train_sampler = None

    src_dataloader = DataLoader(self.src_train_data, batch_size=self.batch_size, shuffle=src_train_sampler is None, sampler=src_train_sampler, drop_last=src_train_sampler is None)
    tar_dataloader = DataLoader(self.tar_train_data, batch_size=self.batch_size, shuffle=tar_train_sampler is None, sampler=tar_train_sampler, drop_last=tar_train_sampler is None)

    return [src_dataloader, tar_dataloader]


  def val_dataloader(self):

    if self.opt.is_distributed:
      print("Ditributed sampler")
      src_train_sampler = DistributedSampler(self.src_val_data, shuffle=False, drop_last=True)
      tar_train_sampler = DistributedSampler(self.tar_val_data, shuffle=False, drop_last=True)

    else:
      src_train_sampler = None
      tar_train_sampler = None

    src_dataloader = DataLoader(self.src_train_data, batch_size=self.batch_size, shuffle=False, sampler=src_train_sampler, drop_last=src_train_sampler is None)
    tar_dataloader = DataLoader(self.tar_train_data, batch_size=self.batch_size, shuffle=False, sampler=tar_train_sampler, drop_last=tar_train_sampler is None)

    return [src_dataloader, tar_dataloader]

  def test_dataloader(self):


    src_dataloader = DataLoader(self.src_val_data, batch_size=self.batch_size, shuffle=False, sampler=None, drop_last=False)
    tar_dataloader = DataLoader(self.tar_val_data, batch_size=self.batch_size, shuffle=False, sampler=None, drop_last=False)

    return [src_dataloader, tar_dataloader]