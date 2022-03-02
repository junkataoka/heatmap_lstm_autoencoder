#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

#%%
inp = torch.load("./dataset/input_original.pt")
tar = torch.load("./dataset/target.pt")

#%%
inp[0, :, -1, 0, 0].cpu()

#%%
plt.plot(tar[10, :, -1, :, :].mean((-1,-2)).cpu())
plt.plot(inp[10, :, -1, 0, 0].cpu())
#%%
#%%
n = 400
print(tar[n, :, -1, :, :].mean((-1,-2)).cpu())
print(inp[n, :, -1, 0, 0])
#%%