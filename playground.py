#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

#%%
inp = torch.load("./dataset/input_original.pt")
tar = torch.load("./dataset/target.pt")

#%%
kf = KFold(n_splits=10, random_state=0, shuffle=True)
kf
