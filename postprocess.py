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
# %%
pred_paths = glob("dataset/*_pred.pt")
tar_paths = glob("dataset/*_tar.pt")
pred_paths.sort()
tar_paths.sort()

pred_l = []
tar_l = []
for idx, path in enumerate(pred_paths):

    pred_tensor = torch.load(pred_paths[idx], map_location=torch.device('cpu'))
    pred_l.append(pred_tensor)
    pred_tensor = pred_tensor.view(15, 50, 50)
    tar_tensor = torch.load(tar_paths[idx], map_location=torch.device('cpu'))
    tar_l.append(tar_tensor)
    tar_tensor = tar_tensor.view(15, 50, 50)

    for i in range(pred_tensor.size(0)):
        pred = pred_tensor[i]
        tar = tar_tensor[i]
        tar_df = pd.DataFrame(tar.numpy())
        tar_df.to_csv(f"Target/{idx}_{i}.csv", index=False)
        pred_df = pd.DataFrame(pred.numpy())
        pred_df.to_csv(f"Pred/{idx}_{i}.csv", index=False)

#%%
target = torch.stack(tar_l, dim=0)
prediction = torch.stack(pred_l, dim=0)
diff = torch.abs(target - prediction)

step_error = diff.mean((0, 1, 3, 4, 5))
area_error = diff.mean((0, 1, 2, 3))

plt.figure(figsize=(12, 8))
sns.set(font_scale=2)
sns.heatmap(area_error)
plt.xticks([]),plt.yticks([])
plt.savefig("Figure/area_error.png", dpi=300)
#%%
plt.figure(figsize=(12, 8))
plt.bar(torch.arange(1,16), step_error)
plt.ylabel("Mean Aboslute Difference")
plt.yticks(fontsize=16)
plt.xlabel("Time Step", fontsize=16)
plt.xticks(fontsize=16)
plt.savefig("Figure/step_error.png", dpi=300)
# %%