#%%
import torch
from torch.utils.data import Dataset, DataLoader 
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
  def __init__(self, root: str, input_file, target_file, batch_size):
    super().__init__()
    self.root = root
    self.input_file = input_file
    self.target_file = target_file
    self.batch_size = batch_size

  def setup(self, stage=None):

    self.train_data = timeseries(self.root, self.input_file, self.target_file)

  def train_dataloader(self):
      return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
      return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

  def test_dataloader(self):
      return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

  