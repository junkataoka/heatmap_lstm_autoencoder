# %%
from typing import DefaultDict
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models import ConvLSTMCell
import os
import numpy as np
import random
from sklearn.model_selection import KFold, train_test_split

class args:
    seed = 0

torch.manual_seed(args.seed)
np.random.seed(args.seed)
#%%

#%%
def generate_target(root_recipe, num_area, num_geom, num_recipe):

    out = np.empty((num_recipe, num_geom, num_area))

    for i in range(num_recipe):
        for j in range(num_geom):
            for k in range(num_area):
                recipe_path = f"recipe_{i+1}_{k+1}.csv"
                recipe_img = np.genfromtxt(os.path.join(root_recipe, recipe_path), delimiter=",")
                out[i, j, k] = recipe_img[0, 0]

    out = out.reshape((num_recipe*num_geom, num_area))

    return out

def generate_input(root_geom, root_heatmap, seq_len, num_geom, num_recipe):

    out = np.empty((num_recipe, num_geom, seq_len, 4, 50, 50))

    for i in range(num_recipe):
        for j in range(num_geom):
            for k in range(seq_len):

                die_path = f"M{j+1}_DIE.csv"
                pcb_path = f"M{j+1}_PCB.csv"
                trace_path = f"M{j+1}_Substrate.csv"
                heatmap_path = f"IMG_{j+1}_{i+1}_{k+1}.csv"

                die_img = np.genfromtxt(os.path.join(root_geom, die_path), delimiter=",")
                pcb_img = np.genfromtxt(os.path.join(root_geom, pcb_path), delimiter=",")
                trace_img = np.genfromtxt(os.path.join(root_geom, trace_path), delimiter=",")
                heatmap_img = np.genfromtxt(os.path.join(root_heatmap, heatmap_path), delimiter=",")

                arr = np.concatenate([die_img[np.newaxis, ...], pcb_img[np.newaxis, ...],
                                trace_img[np.newaxis, ...], heatmap_img[np.newaxis, ...]], axis=0)
                out[i, j,  k, :, :, :] = arr
    out = out.reshape(num_recipe*num_geom, seq_len, 4, 50, 50)

    return out

#%%
print("Generating target data")
a = generate_target(root_recipe="recipe_simulation", num_area=7, num_geom=12, num_recipe=81)
target_tensor = torch.tensor(a).cuda()
torch.save(target_tensor, f"./dataset/source_target.pt")

a = generate_target(root_recipe="recipe_experiment", num_area=7, num_geom=1, num_recipe=3)
target_tensor = torch.tensor(a).cuda()
torch.save(target_tensor, f"./dataset/target_target.pt")

target_target_cv1_train = torch.index_select(target_tensor, dim=0, index=torch.tensor([0,1]).cuda())
target_target_cv1_test = torch.index_select(target_tensor, dim=0, index=torch.tensor([2]).cuda())
torch.save(target_target_cv1_train, "./dataset/target_target_cv1_train.pt")
torch.save(target_target_cv1_test, "./dataset/target_target_cv1_test.pt")

target_target_cv2_train = torch.index_select(target_tensor, dim=0, index=torch.tensor([0,2]).cuda())
target_target_cv2_test = torch.index_select(target_tensor, dim=0, index=torch.tensor([1]).cuda())

torch.save(target_target_cv2_train, "./dataset/target_target_cv2_train.pt")
torch.save(target_target_cv2_test, "./dataset/target_target_cv2_test.pt")

target_target_cv3_train = torch.index_select(target_tensor, dim=0, index=torch.tensor([1,2]).cuda())
target_target_cv3_test = torch.index_select(target_tensor, dim=0, index=torch.tensor([0]).cuda())
torch.save(target_target_cv3_train, "./dataset/target_target_cv3_train.pt")
torch.save(target_target_cv3_test, "./dataset/target_target_cv3_test.pt")

# %%
print("Generating input data")
a = generate_input(root_geom="./geo_img", root_heatmap="./heatmap_simulation",
                   seq_len=15, num_geom=12, num_recipe=81)
input_tensor = torch.tensor(a).cuda()
mean = torch.mean(input_tensor, dim=(0, 1, 3, 4), keepdim=True)
sd = torch.std(input_tensor, dim=(0, 1, 3, 4), keepdim=True)
input_tensor_normalized = (input_tensor - mean + 1e-5) / (sd + 1e-5)
torch.save(input_tensor_normalized, "./dataset/source_input.pt")
torch.save(mean, "./dataset/source_mean.pt")
torch.save(sd, "./dataset/source_sd.pt")
#%%
print("Generating target input data")
a = generate_input(root_geom="./geo_img", root_heatmap="./heatmap_experiment",
                   seq_len=15, num_geom=1, num_recipe=3)
input_tensor = torch.tensor(a).cuda()
input_tensor_normalized = (input_tensor - mean + 1e-5) / (sd + 1e-5)
torch.save(input_tensor_normalized, "./dataset/target_input.pt")

# %%
target_input_cv1_train = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([0,1]).cuda())
target_input_cv1_test = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([2]).cuda())
torch.save(target_input_cv1_train, "./dataset/target_input_cv1_train.pt")
torch.save(target_input_cv1_test, "./dataset/target_input_cv1_test.pt")

target_input_cv2_train = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([0,2]).cuda())
target_input_cv2_test = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([1]).cuda())
torch.save(target_input_cv2_train, "./dataset/target_input_cv2_train.pt")
torch.save(target_input_cv2_test, "./dataset/target_input_cv2_test.pt")

target_input_cv3_train = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([1,2]).cuda())
target_input_cv3_test = torch.index_select(input_tensor_normalized, dim=0, index=torch.tensor([0]).cuda())
torch.save(target_input_cv3_train, "./dataset/target_input_cv3_train.pt")
torch.save(target_input_cv3_test, "./dataset/target_input_cv3_test.pt")
