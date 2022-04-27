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
    num_recipe_src = 81
    num_recipe_tar = 3
    num_geom_src = 9
    num_geom_tar = 3
    seq_len = 15
    num_area = 7
    seed = 0

torch.manual_seed(args.seed)
np.random.seed(args.seed)
#%%

#%%
def generate_input(root, num_recipe, num_area, num_geom, left_geom, left_recipe):

    out = np.empty((num_recipe, num_area, num_geom, 4, 50, 50))

    for i in range(num_recipe):
        for j in range(num_area):
            for k in range(num_geom):
                if k not in (7, 8, 9):
                    die_path = f"M{k+left_geom}_DIE.csv"
                    pcb_path = f"M{k+left_geom}_PCB.csv"
                    trace_path = f"M{k+left_geom}_Substrate.csv"
                    recipe_path = f"recipe_{i+left_recipe}_{j+1}.csv"
                    die_img = np.genfromtxt(os.path.join(root, die_path), delimiter=",")
                    pcb_img = np.genfromtxt(os.path.join(root, pcb_path), delimiter=",")
                    trace_img = np.genfromtxt(os.path.join(root, trace_path), delimiter=",")
                    recipe_img = np.genfromtxt(os.path.join(root, recipe_path), delimiter=",")
                    arr = np.concatenate([die_img[np.newaxis, ...], pcb_img[np.newaxis, ...],
                                    trace_img[np.newaxis, ...], recipe_img[np.newaxis, ...]], axis=0)
                    out[i, j, k, :, :, :] = arr

    return out

def generate_target(root, num_recipe, seq_len, num_geom, left_geom, left_recipe):

    out = np.empty((num_recipe, seq_len, num_geom, 1, 50, 50))
    for i in range(num_recipe):
        for j in range(seq_len):
            for k in range(num_geom):
                target_path = f"IMG_{k+left_geom}_{i+left_recipe}_{j+1}.csv"
                img = np.genfromtxt(os.path.join(root, target_path), delimiter=",")
                img = img[np.newaxis, ...]
                out[i, j, k, :, :, :] = img
    return out

def generate_target_target(root, num_recipe, seq_len, num_geom, left_geom, left_recipe):

    out = np.empty((num_recipe, seq_len, 1, 50, 50))
    for i in range(num_recipe):
        for j in range(seq_len):
            target_path = f"IMG_{i+left_geom}_{i+left_recipe}_{j+1}.csv"
            img = np.genfromtxt(os.path.join(root, target_path), delimiter=",")
            img = img[np.newaxis, ...]
            out[i, j, :, :, :] = img
    return out

def generate_target_input(root, num_recipe, num_area, num_geom, left_geom, left_recipe):

    out = np.empty((num_recipe, num_area, 4, 50, 50))

    for i in range(num_recipe):
        for j in range(num_area):
            die_path = f"M{i+left_geom}_DIE.csv"
            pcb_path = f"M{i+left_geom}_PCB.csv"
            trace_path = f"M{i+left_geom}_Substrate.csv"
            recipe_path = f"recipe_{i+left_recipe}_{j+1}.csv"
            die_img = np.genfromtxt(os.path.join(root, die_path), delimiter=",")
            pcb_img = np.genfromtxt(os.path.join(root, pcb_path), delimiter=",")
            trace_img = np.genfromtxt(os.path.join(root, trace_path), delimiter=",")
            recipe_img = np.genfromtxt(os.path.join(root, recipe_path), delimiter=",")
            arr = np.concatenate([die_img[np.newaxis, ...], pcb_img[np.newaxis, ...],
                            trace_img[np.newaxis, ...], recipe_img[np.newaxis, ...]], axis=0)
            out[i, j,  :, :, :] = arr

    return out
#%%

#%%
print("Generating target data")
a = generate_target("./Output", num_recipe=args.num_recipe_src, seq_len=args.seq_len, num_geom=args.num_geom_src, left_geom=1, left_recipe=1)
target_tensor = torch.tensor(a).cuda()
target_tensor = target_tensor.permute(0,2,1,3,4,5)
target_tensor = target_tensor.reshape(args.num_recipe_src*args.num_geom_src, args.seq_len, 1, 50, 50) - 273.15
torch.save(target_tensor, f"./dataset/source_target.pt")

# #%%
print("Generating input data")
a = generate_input("./INPUT", num_recipe=args.num_recipe_src, num_area=args.num_area, num_geom=args.num_geom_src, left_geom=1, left_recipe=1)
input_tensor = torch.tensor(a).cuda()
input_tensor = input_tensor.permute(0,2,1,3,4,5)
input_tensor = input_tensor.reshape(args.num_recipe_src*args.num_geom_src, args.num_area, 4, 50, 50)
mean = torch.mean(input_tensor, dim=(0, 3, 4), keepdim=True)
sd = torch.std(input_tensor, dim=(0, 3, 4), keepdim=True)
input_tensor_normalized = (input_tensor - mean + 1e-5) / (sd + 1e-5)
torch.save(input_tensor_normalized, "./dataset/source_input.pt")
torch.save(mean, "./dataset/source_mean.pt")
torch.save(sd, "./dataset/source_sd.pt")

print("Generating target input data")
a = generate_target_input("./INPUT_Experiment", num_recipe=args.num_recipe_tar, num_area=args.num_area, num_geom=args.num_geom_tar, left_geom=7, left_recipe=82)
input_tensor = torch.tensor(a).cuda()
# input_tensor = input_tensor.permute(0,2,1,3,4,5)
# input_tensor = input_tensor.reshape(args.num_recipe_tar*args.num_geom_tar, args.num_area, 4, 50, 50)
# mean = torch.mean(input_tensor, dim=(0, -2, -1), keepdim=True)
# sd = torch.std(input_tensor, dim=(0, -2, -1), keepdim=True)
input_tensor_normalized = (input_tensor - mean + 1e-5) / (sd + 1e-5)
torch.save(input_tensor_normalized, "./dataset/target_input.pt")

#%%
print("Generating target target data")
a = generate_target_target("./Output_Experiment", num_recipe=args.num_recipe_tar, seq_len=args.seq_len, num_geom=args.num_geom_tar, left_geom=7, left_recipe=82)
target_tensor = torch.tensor(a).cuda() - 273.15
# target_tensor = target_tensor.permute(0,2,1,3,4,5)
# target_tensor = target_tensor.reshape(args.num_recipe_tar*args.num_geom_tar, args.seq_len, 1, 50, 50) - 273.15
torch.save(target_tensor, f"./dataset/target_target.pt")

tar_x_train, tar_x_test, tar_y_train, tar_y_test = train_test_split(input_tensor_normalized, target_tensor, test_size = 2, random_state=args.seed)

torch.save(tar_x_train, f"./dataset/tar_x_train_exp2.pt")
torch.save(tar_x_test, f"./dataset/tar_x_test_exp2.pt")
torch.save(tar_y_train, f"./dataset/tar_y_train_exp2.pt")
torch.save(tar_y_test, f"./dataset/tar_y_test_exp2.pt")

#%%
x = torch.load("dataset/source_target.pt", map_location=torch.device('cpu'))
plt.figure(figsize=(4, 3))
plt.imshow(x[0,0,0, :].cpu())
plt.xticks([]),plt.yticks([])
plt.savefig("source_input.png", dpi=300, bbo_inches="tight")