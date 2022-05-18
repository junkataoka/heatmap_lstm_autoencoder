#%%
import numpy as np
import torch
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
#%%
df = pd.read_csv("temp/BO_test_result.csv")
plt.figure(figsize=(4, 3))
plt.plot(df.Step, df.SATA4, ".-", label=f"Measurement")
plt.plot(df.Step, df.BO, ".-", label=f"BO")
plt.ylabel("Temperature")
plt.xlabel("Time Step")
plt.legend()
plt.savefig(f"Figure/BO_comparison.png", dpi=300, bbox_inches='tight')
plt.close("all")

# plt.fidf.SATA4
#%%
df = pd.read_json("bo_logs.json", lines=True)
df.params.iloc[df.target.argmin()]

#%%
in_paths = glob("bin/INPUT/recipe*")
out_path = "INPUT"

# %%
for p in in_paths:
    recipe_num = int(p.split("_")[-1].split(".")[0]) + 3
    new_file_name = "_".join(p.split("_")[:-1]) + "_" + str(recipe_num) + ".csv"
    new_file_name = new_file_name.split("/")[-1]
    os.rename(p, os.path.join(out_path, new_file_name))

# %%
recipe_num = 81
temps = [120, 150, 180]
sample = pd.read_csv("INPUT/recipe_81_7.csv", header=None)
#%%
for ind in range(recipe_num):
    for t in range(len(temps)):
        file_name = f"INPUT/recipe_{ind+1}_{t+1}.csv"
        sample.iloc[:, :] = temps[t]
        sample.to_csv(file_name, index=False, header=False)
#%%
num_add_recipe = 3
recipe_list = { "M7":[105, 130, 160, 190, 230, 270, 290],
               "M8":[110, 140, 170, 200, 240, 280, 290],
                "M9":[120, 150, 180, 220, 260, 280, 300]}

c = 1
for key, val in recipe_list.items():
    for t in range(len(val)):
        file_name = f"INPUT_Experiment/recipe_{c + recipe_num}_{t+1}.csv"
        sample.iloc[:, :] = val[t]
        sample.to_csv(file_name, index=False, header=False)
    c += 1
# %%

# %%
in_paths = glob("INPUT/recipe*")
# %%
d = defaultdict(list)
recipe_num = 81
area_length = 7
for i in range(recipe_num):
    res = []
    for j in range(area_length):
        fname = f"INPUT/recipe_{i+1}_{j+1}.csv"
        print(fname)
        df_temp = pd.read_csv(fname, index_col=0)
        val = df_temp.iloc[0, 0]
        print(val)
        res.append(val)
    d[i] = res

# %%
# Recipe 13 is close to M9 recipe

pcb_temp = []
for i in range(15):
    fname = f"Output/IMG_9_13_{i+1}.csv"
    df = pd.read_csv(fname, index_col=0)
    val = df.iloc[1, 1]
    pcb_temp.append(val)

len(pcb_temp)
# %%
data = pd.read_csv("experiment_20220323.csv")
# %%
result = []
result = [33, 66, 99, 132, 171, 204, 214, 224,
            234, 244, 254, 264, 274, 284, 294]

result = [i * 2 for i in result]

# p = 0
# for temp in pcb_temp:
#     arr = np.zeros(601) + 1e+6
#     diff = np.abs(data["M9_board"] - temp)
#     arr[p:] = diff[p:]
#     ind = np.argmin(arr)
#     result.append(ind)
#     p = ind

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
#%%
plt.figure(figsize=(4,3))
plt.plot(result, small_data["M9_center"].values, ".-", label="M9")
plt.plot(result, small_data["M8_center"].values,".-", label="M8")
plt.plot(result, small_data["M7_center"].values, ".-", label="M7")
plt.ylabel("Temeperature")
plt.xlabel("Sec")
plt.legend()
plt.savefig("small_data.png", dpi=300, bbox_inches="tight")
# %%
def generate_heatmap(die, subtrate, pcb, recipe_num, board_num):
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
        df.to_csv(f"Output_Experiment/IMG_{board_num}_{recipe_num}_{i+1}.csv", index=False, header=False)

generate_heatmap(die_M9, subtrate_M9, pcb_M9, 84, 1)
generate_heatmap(die_M8, subtrate_M8, pcb_M8, 83, 1)
generate_heatmap(die_M7, subtrate_M7, pcb_M7, 82, 1)


# %%

df = pd.read_csv("Output_Experiment/IMG_1_84_5.csv")
plt.figure(figsize=(4,4))
plt.imshow(df)
plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")

#%%
data = torch.load("dataset/tar_y_train.pt", map_location=torch.device("cpu"))
# %%
data = pd.read_csv("logs_avg_diff_src_src.csv", header=None)
plt.figure(figsize=(4, 3))
plt.plot(data.iloc[:, 2])
plt.xlabel("Epoch")
plt.ylabel("Estimation Error")
plt.savefig("src_loss_train.png", dpi=300, bbox_inches="tight")
# %%

data = pd.read_csv("logs_avg_diff_tar_tar.csv", header=None)
plt.figure(figsize=(4, 3))
plt.plot(data.iloc[:, 2])
plt.xlabel("Epoch")
plt.ylabel("Estimation Error")
plt.savefig("tar_loss_train.png", dpi=300, bbox_inches="tight")
# %%

def create_input(geom_num, recipes):

    root = "INPUT"
    die_path = f"M{geom_num}_DIE.csv"
    pcb_path = f"M{geom_num}_PCB.csv"
    trate_path = f"M{geom_num}_Substrate.csv"
    out = np.empty((1, len(recipes), 4, 50, 50))
    die_img = np.genfromtxt(os.path.join(root, die_path), delimiter=",")
    pcb_img = np.genfromtxt(os.path.join(root, pcb_path), delimiter=",")
    trace_img = np.genfromtxt(os.path.join(root, trate_path), delimiter=",")
    recipe_img = np.zeros_like(die_img)
    for i in range(len(recipes)):
        recipe_img[:, :] = recipes[i]
        arr = np.concatenate([die_img[np.newaxis, ...], pcb_img[np.newaxis, ...],
                        trace_img[np.newaxis, ...], recipe_img[np.newaxis, ...]], axis=0)
        out[0, i,  :, :, :] = arr
    inp = torch.tensor(out)
    return inp

inp_target = torch.load("dataset/tar_x_train.pt", map_location="cuda:0")
inp_target = inp_target[0, :]
inp_target.unsqueeze_(0)
inp_target = inp_target.type(torch.cuda.FloatTensor)
src_mean = torch.load("dataset/source_mean.pt", map_location="cuda:0")
src_sd = torch.load("dataset/source_sd.pt", map_location="cuda:0")

recipes = [120, 150, 180, 210, 240, 280, 300]
steps = [33, 66, 99, 132, 171, 204, 214, 224,
            234, 244, 254, 264, 274, 284, 294]
inp = create_input(7, recipes)
inp = inp.cuda()
inp_normalized = (inp - src_mean + 1e-5)/(src_sd+1e-5)
inp_normalized = inp_normalized.type(torch.cuda.FloatTensor)
#%%
inp_target[:, 1, -1, 1, 1]
# %%
inp_normalized[:, 1, -1, 1, 1]
