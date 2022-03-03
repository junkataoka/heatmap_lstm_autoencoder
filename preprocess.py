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
from sklearn.model_selection import KFold

class args:
    num_recipe =81
    num_geom = 12
    seq_len = 15
    num_area = 15
    num_target_pseudo_data = 20
    seed = 0

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#%%
def generate_input(root, num_recipe, num_area, num_geom):

    out = np.empty((num_recipe, num_area, num_geom, 4, 50, 50))

    for i in range(num_recipe):
        for j in range(num_area):
            for k in range(num_geom):
                die_path = f"M{k+1}_DIE.csv"
                pcb_path = f"M{k+1}_PCB.csv"
                trace_path = f"M{k+1}_Substrate.csv"
                recipe_path = f"recipe_{i+1}_{j+1}.csv"
                die_img = np.genfromtxt(os.path.join(root, die_path), delimiter=",")
                pcb_img = np.genfromtxt(os.path.join(root, pcb_path), delimiter=",")
                trace_img = np.genfromtxt(os.path.join(root, trace_path), delimiter=",")
                recipe_img = np.genfromtxt(os.path.join(root, recipe_path), delimiter=",")
                arr = np.concatenate([die_img[np.newaxis, ...], pcb_img[np.newaxis, ...],
                                trace_img[np.newaxis, ...], recipe_img[np.newaxis, ...]], axis=0)
                out[i, j, k, :, :, :] = arr

    return out

def generate_target(root, num_recipe, seq_len, num_geom):

    out = np.empty((num_recipe, seq_len, num_geom, 1, 50, 50))
    for i in range(num_recipe):
        for j in range(seq_len):
            for k in range(num_geom):
                target_path = f"IMG_{k+1}_{i+1}_{j+1}.csv"
                img = np.genfromtxt(os.path.join(root, target_path), delimiter=",")
                img = img[np.newaxis, ...]
                out[i, j, k, :, :, :] = img
    return out
#%%
# print("Generating target data")
# a = generate_target("./Output", num_recipe=args.num_recipe, seq_len=args.seq_len, num_geom=args.num_geom)
# target_tensor = torch.tensor(a).cuda()
# target_tensor = target_tensor.permute(0,2,1,3,4,5)
# target_tensor = target_tensor.reshape(args.num_recipe*args.num_geom, args.seq_len, 1, 50, 50) - 273.15
# torch.save(target_tensor, f"./dataset/target.pt")

# #%%
# print("Generating input data")
# a = generate_input("./INPUT", num_recipe=args.num_recipe, num_area=args.num_area, num_geom=args.num_geom)
# input_tensor = torch.tensor(a).cuda()
# input_tensor = input_tensor.permute(0,2,1,3,4,5)
# input_tensor = input_tensor.reshape(args.num_recipe*args.num_geom, args.num_area, 4, 50, 50)
# mean = torch.mean(input_tensor, dim=(0, 3, 4), keepdim=True)
# sd = torch.std(input_tensor, dim=(0, 3, 4), keepdim=True)
# input_tensor_normalized = (input_tensor - mean + 1e-5) / (sd + 1e-5)
# torch.save(input_tensor_normalized, f"./dataset/input.pt")
# torch.save(input_tensor, f"./dataset/input_original.pt")

#%% Create pseudo target dataset
target_tensor = torch.load('./dataset/target.pt')
input_tensor = torch.load('./dataset/input.pt')
kf = KFold(n_splits=10, random_state=args.seed, shuffle=True)
c = 1
for train, test in kf.split(target_tensor):

    torch.save(input_tensor[train], f"./dataset/train_input_fold{c}.pt")
    torch.save(target_tensor[train], f"./dataset/train_target_fold{c}.pt")
    torch.save(input_tensor[test], f"./dataset/test_input_fold{c}.pt")
    torch.save(target_tensor[test], f"./dataset/test_target_fold{c}.pt")

    c += 1
