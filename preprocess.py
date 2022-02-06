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

# %%
class args:
    num_recipe =81 
    num_geom = 12
    seq_len = 15
    num_area = 15

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
                                trace_img[np.newaxis, ...], recipe_img[np.newaxis, ...]])
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
#%%
if not os.path.exists("dataset/target.pt"):
    print("Generating target data")
    a = generate_target("./Output", num_recipe=args.num_recipe, seq_len=args.seq_len, num_geom=args.num_geom)
    target_tensor = torch.tensor(a)
    target_tensor = target_tensor.view(args.num_recipe*args.num_geom, args.seq_len, 1, 50, 50)
    # mn = torch.mean(target_tensor, 0, keepdim=True)
    # sd = torch.std(target_tensor, 0, keepdim=True)
    # target_tensor = (target_tensor - mn + 1e-5) / (sd+1e-5)
    # torch.save(mn, "./dataset/target_mn.pt")
    # torch.save(sd, "./dataset/target_sd.pt")
    torch.save(target_tensor, "./dataset/target.pt")
#%%
if not os.path.exists("dataset/input.pt"):
    print("Generating input data")
    a = generate_input("./INPUT", num_recipe=args.num_recipe, num_area=args.num_area, num_geom=args.num_geom)
    target_tensor = torch.tensor(a)
    target_tensor = target_tensor.view(args.num_recipe*args.num_geom, args.num_area, 4, 50, 50)
    print(target_tensor.shape)
    mn = torch.mean(target_tensor, 0, keepdim=True)
    sd = torch.std(target_tensor, 0, keepdim=True)
    target_tensor = (target_tensor - mn + 1e-5) / (sd+1e-5)
    torch.save(mn, "./dataset/input_mn.pt")
    torch.save(sd, "./dataset/input_sd.pt")
    torch.save(target_tensor, "./dataset/input.pt")