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

#%%



# %%

# df = pd.read_csv("Output_Experiment/IMG_1_84_5.csv")
# plt.figure(figsize=(4,4))
# plt.imshow(df)
# plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")

# #%%
# data = torch.load("dataset/tar_y_train.pt", map_location=torch.device("cpu"))
# # %%
# data = pd.read_csv("logs_avg_diff_src_src.csv", header=None)
# plt.figure(figsize=(4, 3))
# plt.plot(data.iloc[:, 2])
# plt.xlabel("Epoch")
# plt.ylabel("Estimation Error")
# plt.savefig("src_loss_train.png", dpi=300, bbox_inches="tight")
# # %%

# data = pd.read_csv("logs_avg_diff_tar_tar.csv", header=None)
# plt.figure(figsize=(4, 3))
# plt.plot(data.iloc[:, 2])
# plt.xlabel("Epoch")
# plt.ylabel("Estimation Error")
# plt.savefig("tar_loss_train.png", dpi=300, bbox_inches="tight")
# # %%

# def create_input(geom_num, recipes):

#     root = "INPUT"
#     die_path = f"M{geom_num}_DIE.csv"
#     pcb_path = f"M{geom_num}_PCB.csv"
#     trate_path = f"M{geom_num}_Substrate.csv"
#     out = np.empty((1, len(recipes), 4, 50, 50))
#     die_img = np.genfromtxt(os.path.join(root, die_path), delimiter=",")
#     pcb_img = np.genfromtxt(os.path.join(root, pcb_path), delimiter=",")
#     trace_img = np.genfromtxt(os.path.join(root, trate_path), delimiter=",")
#     recipe_img = np.zeros_like(die_img)
#     for i in range(len(recipes)):
#         recipe_img[:, :] = recipes[i]
#         arr = np.concatenate([die_img[np.newaxis, ...], pcb_img[np.newaxis, ...],
#                         trace_img[np.newaxis, ...], recipe_img[np.newaxis, ...]], axis=0)
#         out[0, i,  :, :, :] = arr
#     inp = torch.tensor(out)
#     return inp

# inp_target = torch.load("dataset/tar_x_train.pt", map_location="cuda:0")
# inp_target = inp_target[0, :]
# inp_target.unsqueeze_(0)
# inp_target = inp_target.type(torch.cuda.FloatTensor)
# src_mean = torch.load("dataset/source_mean.pt", map_location="cuda:0")
# src_sd = torch.load("dataset/source_sd.pt", map_location="cuda:0")

# recipes = [120, 150, 180, 210, 240, 280, 300]
# steps = [33, 66, 99, 132, 171, 204, 214, 224,
#             234, 244, 254, 264, 274, 284, 294]
# inp = create_input(7, recipes)
# inp = inp.cuda()
# inp_normalized = (inp - src_mean + 1e-5)/(src_sd+1e-5)
# inp_normalized = inp_normalized.type(torch.cuda.FloatTensor)
# #%%
# inp_target[:, 1, -1, 1, 1]
# # %%
# inp_normalized[:, 1, -1, 1, 1]
