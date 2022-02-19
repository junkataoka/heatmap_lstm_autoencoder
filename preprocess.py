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

class args:
    num_recipe =81 
    num_geom = 12
    seq_len = 15
    num_area = 15
    num_target_pseudo_data = 20
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

print("Generating target data")
a = generate_target("./Output", num_recipe=args.num_recipe, seq_len=args.seq_len, num_geom=args.num_geom)
target_tensor = torch.tensor(a)
target_tensor = target_tensor.view(args.num_recipe*args.num_geom, args.seq_len, 1, 50, 50)
for i in range(target_tensor.shape[0]):
    torch.save(target_tensor[i], f"./dataset/target/idx{i}_target.pt")

print("Generating input data")
a = generate_input("./INPUT", num_recipe=args.num_recipe, num_area=args.num_area, num_geom=args.num_geom)
input_tensor = torch.tensor(a)
input_tensor = input_tensor.view(args.num_recipe*args.num_geom, args.num_area, 4, 50, 50)
mean = torch.mean(input_tensor, dim=(0, 3, 4), keepdim=True)
sd = torch.std(input_tensor, dim=(0, 3, 4), keepdim=True)
input_tensor = (input_tensor - mean + 1e-5) / (sd + 1e-5)
for i in range(input_tensor.shape[0]):
    torch.save(input_tensor[i], f"./dataset/input/idx{i}_input.pt")

# Create pseudo target dataset
high = input_tensor.shape[0]
target_batch_idx = torch.randint(low=0, high=high, size=(args.num_target_pseudo_data,))
source_batch_idx = [i for i in range(high) if i not in target_batch_idx]

source_input_tensor = input_tensor[source_batch_idx]
source_target_tensor = target_tensor[source_batch_idx]
for i in range(source_input_tensor.shape[0]):
    torch.save(source_input_tensor[i], f"./dataset/source_input/idx{i}_input.pt")
    torch.save(source_target_tensor[i], f"./dataset/source_target/idx{i}_target.pt")

target_input_tensor = input_tensor[target_batch_idx]
target_target_tensor = target_tensor[target_batch_idx]
noise = torch.randn((args.num_target_pseudo_data, 15, 1, 23, 23))
emp = torch.zeros((args.num_target_pseudo_data, 15, 1 , 50, 50))
emp[:, :, :, :23, :23] = noise
ps_target_target_tensor = target_target_tensor + emp
mu = ps_target_target_tensor[:, :, :, :23, :23].mean((-1, -2), keepdim=True)
ps_target_target_tensor[:, :, :, :23, :23] = mu

for i in range(target_input_tensor.shape[0]):
    torch.save(target_input_tensor[i], f"./dataset/target_input/idx{i}_input.pt")
    torch.save(target_target_tensor[i], f"./dataset/target_target/idx{i}_target.pt")
    torch.save(ps_target_target_tensor[i], f"./dataset/ps_target_target/idx{i}_target.pt")
