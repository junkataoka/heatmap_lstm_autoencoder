# %%
from typing import DefaultDict
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import seaborn as sns
from torchvision import transforms
# %%
pred_paths = glob("pred/target*")
pred_paths.sort()

pred_l = []
tar_l = []

for idx, path in enumerate(pred_paths):
    pred_tensor = torch.load(pred_paths[idx], map_location=torch.device('cpu'))
    pred_l.append(pred_tensor)
    pred_tensor = pred_tensor.view(15, 50, 50)

prediction = torch.stack(pred_l)
prediction = prediction.reshape(len(pred_paths), 15, 1, 50 ,50)
target = torch.load("dataset/target.pt").cpu()
#%%
target.shape
#%%
diff = torch.abs(target - prediction)
step_error = diff.mean((0,  3, 4)).cpu().data.numpy()
area_error = diff.mean((0,  1, 2)).cpu().data.numpy()
step_error =step_error.reshape(15)
temp = prediction[0, 0, :, :, :].detach().squeeze(2).squeeze(0)
#%%
plt.figure(figsize=(12, 8))
sns.set(font_scale=2)
sns.heatmap(temp)
plt.xticks([]),plt.yticks([])
plt.savefig("Figure/area_error.png", dpi=300)
#%%
plt.figure(figsize=(12, 8))
plt.bar(torch.arange(1,16), step_error)
plt.ylabel("Mean Aboslute Difference", fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Time Step", fontsize=16)
plt.xticks(fontsize=16)
plt.savefig("Figure/step_error.png", dpi=300)
# %%
print(diff.mean())
