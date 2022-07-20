#%%
import numpy as np
import torch
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# %% Create recipe csv files
recipe_num = 81
temps = [120, 150, 180]
sample = pd.read_csv("recipe_experiment/recipe_1_1.csv", header=None)
num_add_recipe = 3
recipe_list = { "M7":[105, 130, 160, 190, 230, 270, 290],
               "M8":[110, 140, 170, 200, 240, 280, 290],
                "M9":[120, 150, 180, 220, 260, 280, 300]}

c = 0
for key, val in recipe_list.items():
    for t in range(len(val)):
        file_name = f"recipe_experiment/recipe_{c + 1}_{t+1}.csv"
        sample.iloc[:, :] = val[t]
        sample.to_csv(file_name, index=False, header=False)
    c += 1



# %% Create
data = pd.read_csv("experiment_20220323.csv")
result = []
result = [33, 66, 99, 132, 171, 204, 214, 224,
            234, 244, 254, 264, 274, 284, 294]

result = [i * 2 for i in result]

# %%
small_data = data.iloc[result, :]

die_M9 = small_data["M9_center"].values
subtrate_M9 = small_data["M9_corner"].values
pcb_M9 = small_data["M9_board"].values

die_M8 = small_data["M8_center"].values
subtrate_M8 = small_data["M8_corner"].values
pcb_M8 = small_data["M8_board"].values

die_M7 = small_data["M7_center"].values
subtrate_M7 = small_data["M7_corner"].values
pcb_M7 = small_data["M7_board"].values

# %%
def generate_heatmap(heatmap_root, die, subtrate, pcb, recipe_num, board_num):
    img = np.zeros((50, 50, 15))
    pcb_out = np.ones((30, 30, 15))
    subtrate_out = np.ones((24, 24, 15))
    die_out = np.ones((12, 17, 15))

    for i in range(15):
        die_out[:, :, i] = die[i]
        subtrate_out[:, :, i] = subtrate[i]
        pcb_out[:, :, i] = pcb[i]

    img[:30, :30, :] = pcb_out
    img[:24, :24, :] = subtrate_out
    img[:12, :17, :] = die_out

    for i in range(15):
        df = pd.DataFrame(img[:, :, i])
        df.to_csv(f"{heatmap_root}/IMG_{board_num}_{recipe_num}_{i+1}.csv", index=False, header=False)

generate_heatmap("heatmap_experiment", die_M7, subtrate_M7, pcb_M7, 1, 1)
generate_heatmap("heatmap_experiment", die_M8, subtrate_M8, pcb_M8, 2, 1)
generate_heatmap("heatmap_experiment", die_M9, subtrate_M9, pcb_M9, 3, 1)

#%% Import packages
import pandas as pd
import matplotlib.pyplot as plt
#%% Generate pictures from lab experiment domain
solder_temp = []
for j in range(1, 4):
    for i in range(1, 16):
        df = pd.read_csv(f"heatmap_experiment/IMG_1_1_{i}.csv", header=None)
        solder_temp.append(df.iloc[0, 0] - 272.15)
    plt.figure(figsize=(4,3))
    plt.plot([i for i in range(1, 16)], solder_temp, label="temperature profile")
    plt.xlabel("Time step")
    plt.ylabel("Temperature")
    plt.legend()
    plt.savefig(f"Figure/profile_cv{j}.png", dpi=300, bbox_inches='tight')

# %% Generate plot for validation
solder_temp = []
pcb_temp = []
sub_temp = []
for i in range(1, 16):
    df = pd.read_csv(f"heatmap_experiment/IMG_1_2_{i}.csv", header=None)
    solder_temp.append(df.iloc[0, 0] - 272.15)
    pcb_temp.append(df.iloc[0, 20] - 272.15)
    sub_temp.append(df.iloc[0, 28] - 272.15)
plt.plot(solder_temp, label="solder")
plt.plot(pcb_temp, label="pcb")
plt.plot(sub_temp, label="substrate")
plt.legend()

# %%
df = pd.read_csv(f"heatmap_experiment/IMG_1_2_1.csv", header=None)
plt.imshow(df)
#%%
df = pd.read_csv(f"heatmap_simulation/IMG_1_2_1.csv", header=None)
plt.imshow(df)