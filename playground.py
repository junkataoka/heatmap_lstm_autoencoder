#%%
import numpy as np
import torch
import pandas as pd
#%%
out = {}
for i in range(1, 10):
    err = torch.load(f"temp/test_input_fold{i}.pt_err.pt").cpu().data.numpy()
    inp = torch.load(f"dataset/test_input_original_fold{i}.pt").cpu().data.numpy()
    recipe = inp[:, :, -1, 0, 0]
    for j in range(err.shape[0]):
        out[err[j]] = recipe[j]


sorted_out = dict(sorted(out.items()))

# temp = []
# res = dict()
# for key, val in sorted_out.items():
#     if val not in temp:
#         temp.append(val)
#         res[key] = val
sorted_out

#%%
df = pd.DataFrame(sorted_out)
df = df.T.drop_duplicates().T

# %%
df.to_csv("recipe_for_experiment.csv")
# %%
df.iloc[:, :20]

# %%
