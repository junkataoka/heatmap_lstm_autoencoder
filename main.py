# import libraries
from cgi import print_directory
import os
from time import time
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process
from data_module import TSDataModule
from lstm_ae import EncoderDecoderConvLSTM
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from torchvision import transforms
from mmd import MMD
import seaborn as sns
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import pandas  as pd

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--n_hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')
parser.add_argument('--log_images', action='store_true', help='Whether to log images')
parser.add_argument('--is_distributed', action='store_true', help='Whether to used distributeds dataloader')

parser.add_argument('--root', type=str, default="./dataset")
parser.add_argument('--src_input_file', type=str, default="source_input.pt")
parser.add_argument('--src_target_file', type=str, default="source_target.pt")
parser.add_argument('--tar_input_file', type=str, default="target_input.pt")
parser.add_argument('--tar_target_file', type=str, default="sp_target_target.pt")
parser.add_argument('--time_steps', type=int, default=15)

parser.add_argument('--model_path', type=str, default="checkpoints/lstm_ac.ckpt")
parser.add_argument('--out_model_path', type=str, default="checkpoints/lstm_ac.ckpt")
parser.add_argument('--retrain', action='store_true', help='Whether to retrain the model or not')
parser.add_argument('--neptune_logger', action='store_true', help='Whether to use neptune.ai logger')
parser.add_argument('--api_key', type=str,
                    default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOTE0MGFjYy02NzMwLTRkODQtYTU4My1lNjk0YWEzODM3MGIifQ==")

parser.add_argument('--run_type', type=str, default="train")

opt = parser.parse_args()
print(opt)

# SEED = 1234
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

class OvenLightningModule(pl.LightningModule):

    def __init__(self, opt):

        super(OvenLightningModule, self).__init__()

        self.save_hyperparameters()
        self.opt = opt
        self.normalize = False
        self.model1 = EncoderDecoderConvLSTM(nf=self.opt.n_hidden_dim, in_chan=4)
        self.model2 = EncoderDecoderConvLSTM(nf=self.opt.n_hidden_dim, in_chan=4)
        self.log_images = self.opt.log_images
        self.criterion = torch.nn.MSELoss()
        self.dcl_criterion = torch.nn.NLLLoss()
        self.batch_size = self.opt.batch_size
        self.time_steps = self.opt.time_steps
        self.epoch = 0
        self.step = 0
        alpha = torch.FloatTensor(5).fill_(1)
        self.register_parameter(name="alpha", param=torch.nn.Parameter(data=alpha, requires_grad=True))
        beta = torch.FloatTensor(5).fill_(0)
        self.register_parameter(name="beta", param=torch.nn.Parameter(data=beta, requires_grad=True))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 100 , gamma = 0.5)

        # return {"optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler":sch,
        #             "monitor": "loss"}}
        return optimizer

    def load_model(self):
        checkpoint = torch.load(self.opt.model_path)
        self.model1.load_state_dict(checkpoint["model1"])
        self.model2.load_state_dict(checkpoint["model2"])

        print('Model Created!')


    def forward(self, x, model):

        output, f1, f2, f3, f4 = model(x, future_step=self.time_steps)

        return output, f1, f2, f3, f4

    def RegLoss(self, model1, model2, alpha, beta):
        loss = 0
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if 'encoder_1' in name1:
                loss += torch.norm(alpha[0] * param1 + beta[0] - param2)

            if 'encoder_2' in name1:
                loss += torch.norm(alpha[1] * param1 + beta[1] - param2)

            if 'decoder_1' in name1:
                loss += torch.norm(alpha[2] * param1 + beta[2] - param2)

            if 'decoder_2' in name1:
                loss += torch.norm(alpha[3] * param1 + beta[3] - param2)

            if 'decoder_CNN' in name1:
                loss += torch.norm(alpha[4] * param1 + beta[4] - param2)
        return loss

    def training_step(self, batch, batch_idx):

        src_batch = batch[0]
        tar_batch = batch[1]
        src_x, src_y = src_batch
        tar_x, tar_y = tar_batch

        src_y_hat, src_feat1, src_feat2, src_feat3, src_feat4 = self.forward(src_x, self.model1)
        tar_y_hat, tar_feat1, tar_feat2, tar_feat3, tar_feat4 = self.forward(tar_x, self.model2)

        src_label = torch.zeros(src_x.shape[0]).long().cuda()
        tar_label = torch.ones(tar_x.shape[0]).long().cuda()

        src_loss = self.criterion(src_y_hat, src_y)
        tar_loss = self.criterion(tar_y_hat, tar_y)
        b ,t, c, h, w = src_y_hat.shape

        mmd_loss1 = MMD(src_feat1, tar_feat1, kernel="multiscale")
        mmd_loss2 = MMD(src_feat2, tar_feat2, kernel="multiscale")
        mmd_loss3 = MMD(src_feat3, tar_feat3, kernel="multiscale")
        mmd_loss4 = MMD(src_feat4, tar_feat4, kernel="multiscale")


        reg_loss = self.RegLoss(self.model1, self.model2, self.alpha, self.beta)

        avg_diff_src_src = torch.mean(torch.abs(src_y_hat - src_y))
        avg_diff_tar_tar = torch.mean(torch.abs(tar_y_hat - tar_y))

        self.log("src_loss", src_loss.item(), on_step=False, on_epoch=True)
        self.log("tar_loss", tar_loss.item(), on_step=False, on_epoch=True)
        self.log("mmd_loss1", mmd_loss1.item(), on_step=False, on_epoch=True)
        self.log("mmd_loss2", mmd_loss2.item(), on_step=False, on_epoch=True)
        self.log("mmd_loss3", mmd_loss3.item(), on_step=False, on_epoch=True)
        self.log("mmd_loss4", mmd_loss4.item(), on_step=False, on_epoch=True)
        self.log("alpha", self.alpha[0], on_step=False, on_epoch=True)
        self.log("beta", self.beta[0], on_step=False, on_epoch=True)

        self.log("avg_diff_src_src", avg_diff_src_src.item(), on_step=False, on_epoch=True)

        self.log("avg_diff_tar_tar", avg_diff_tar_tar.item(), on_step=False, on_epoch=True)


        loss = src_loss + tar_loss + mmd_loss1 + mmd_loss2 + mmd_loss3 + mmd_loss4 + reg_loss
        # loss = src_loss

        return loss

def test_trainer():
    model =OvenLightningModule(opt).cuda()
    oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.tar_input_file, opt.tar_target_file, batch_size=1)
    oven_data.setup()
    source_loader, target_loader = oven_data.test_dataloader()
    model.load_model()
    model.eval()
    step_err = torch.zeros(15).cuda(0)
    h = 12
    w = 17
    area_err = torch.zeros((h, w)).cuda(0)
    exp_name = "src"
    steps = [33, 66, 99, 132, 171, 204, 214, 224,
                234, 244, 254, 264, 274, 284, 294]
    for i, batch in enumerate(target_loader):
        inp, target = batch
        with torch.no_grad():
            predictions, _, _, _, _ = model(inp, model.model2)
            step_diff = torch.mean(torch.abs(target[:, :, :, :h, :w] - predictions[:, :, :, :h, :w]), (0,  2, 3, 4))
            area_diff = torch.mean(torch.abs(target[:, :, :, :h, :w] - predictions[:, :, :, :h, :w]), (0,  1, 2))

            step_err += step_diff
            area_err += area_diff

        plt.figure(figsize=(4, 3))
        plt.plot(steps, target[0, :, :, :h, :w].mean((-1, -2)).cpu(), ".-", label="Target")
        plt.plot(steps, predictions[0, :, :, :h, :w].mean((-1, -2)).cpu(), ",-", label="Prediction")
        plt.ylabel("Temperature")
        plt.xlabel("Time Step")
        plt.legend()
        plt.savefig(f"Figure/sample{i}_{exp_name}.png", dpi=300, bbox_inches='tight')
        plt.close("all")

        for k in range(inp.shape[1]):

            plt.figure(figsize=(4, 3))
            plt.imshow(inp[0, k, :-1].permute(1,2,0).cpu())
            plt.xticks([]),plt.yticks([])
            plt.savefig(f"Figure/input_sample{i}_t{k}_{exp_name}.png", dpi=300, bbox_inches="tight")
            plt.close("all")

        for j in range(target.shape[1]):

            plt.figure(figsize=(4, 3))
            plt.imshow(target[0, j, :, :, :].reshape((50, 50)).cpu())
            plt.xticks([]),plt.yticks([])
            plt.savefig(f"Figure/target_sample{i}_t{j}_{exp_name}.png", dpi=300, bbox_inches="tight")

            plt.figure(figsize=(4, 3))
            plt.imshow(predictions[0, j, :, :, :].reshape((50, 50)).cpu())
            plt.xticks([]),plt.yticks([])
            plt.savefig(f"Figure/pred_sample{i}_t{j}_{exp_name}.png", dpi=300, bbox_inches="tight")
            plt.close("all")

    step_err = step_err / len(target_loader)
    area_err = area_err / len(target_loader)
    step_err = step_err.cpu().numpy()
    area_err = area_err.cpu().numpy()

    plt.figure(figsize=(4, 3))
    sns.set(font_scale=2)
    sns.heatmap(area_err)
    plt.xticks([]),plt.yticks([])
    plt.savefig(f"Figure/area_error_{exp_name}.png", dpi=300)
    plt.close("all")
    plt.figure(figsize=(12, 8))
    plt.bar(torch.arange(1,16), step_err)
    plt.ylabel("Mean Aboslute Difference", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Time Step", fontsize=16)
    plt.xticks(fontsize=16)
    plt.savefig(f"Figure/step_error_{exp_name}.png", dpi=300)
    plt.close("all")

def val_best_recipes():
    model =OvenLightningModule(opt).cuda()
    model.eval()
    oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.tar_input_file, opt.tar_target_file, batch_size=1)
    oven_data.setup()
    source_loader, target_loader = oven_data.val_dataloader()
    avg_diff = []
    for idx, batch in enumerate(target_loader):
        inp, target = batch
        with torch.no_grad():
            predictions, _ = model(inp)
            avg_diff.append(torch.mean(torch.abs(target - predictions)))
    arr = torch.stack(avg_diff)
    torch.save(arr, f"temp/{opt.tar_input_file}_err.pt")

def run_trainer():
    model =OvenLightningModule(opt).cuda()
    oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.tar_input_file, opt.tar_target_file, opt.batch_size)
    if opt.neptune_logger:
        logger = NeptuneLogger(
                api_key=opt.api_key,
                    project_name='junkataoka/heatmap',
                                )
    else:
        logger = None


    trainer = Trainer(max_epochs=opt.epochs,
                        gpus=opt.n_gpus,
                        logger=logger,
                        accelerator='ddp',
                        num_nodes=opt.num_nodes,
                        # gradient_clip_val=0.5,
                        # multiple_trainloader_mode="min_size"
                      )

    if opt.retrain:
        model.load_model()

    trainer.fit(model, datamodule=oven_data)
    torch.save({"model1":model.model1.state_dict(),
                "model2":model.model2.state_dict()
                }, opt.out_model_path)


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


def bayesian_ops():
    model =OvenLightningModule(opt).cuda()
    model.load_model()
    # model.load_state_dict(torch.load(opt.model_path, map_location='cuda:0'), strict=False)
    model.eval()

    src_mean = torch.load("dataset/source_mean.pt", map_location="cuda:0")
    src_sd = torch.load("dataset/source_sd.pt", map_location="cuda:0")
    plt.figure(figsize=(4, 3))
    geom_num = 1

    def GetSlope(val1, val2, t1, t2):
        slope = (val2 - val1) / (t2 - t1)
        return slope

    def black_box_function(r1, r2, r3, r4, r5, r6, r7):

        steps = [33, 66, 99, 132, 171, 204, 214, 224,
                    234, 244, 254, 264, 274, 284, 294]
        recipes = [r1, r2, r3, r4, r5, r6, r7]
        inp = create_input(geom_num, recipes)
        inp = inp.cuda()
        inp_normalized = (inp - src_mean + 1e-5)/(src_sd+1e-5)
        inp_normalized = inp_normalized.type(torch.cuda.FloatTensor)
        loss = 0

        target_temp = torch.cuda.FloatTensor( [57, 90, 123, 156, 195, 228, 233, 238, 242, 247, 242, 238, 233, 228, 223]).unsqueeze(1)

        with torch.no_grad():
            pred, _, _, _, _ = model(inp_normalized, model.model1)

        pred_temp = pred[:,:,:, :12, :17].mean((-1,-2)).squeeze(0)
        loss += (target_temp - pred_temp).pow(2).sum() / target_temp.size(0)
        # peak_err = max((max(pred_temp) - max(recipes), 0.0))
        # loss += float(peak_err)


        loss = np.array(loss.cpu())


        return -loss

    # pbounds = {"r1": (100, 120), "r2": (120, 170), "r3": (170, 190),
    #            "r4":(190, 210), "r5":(240, 400), "r6": (270, 400), "r7": (290, 400)}


    pbounds = {"r1": (100, 280), "r2": (100, 280), "r3": (100, 280),
               "r4":(100, 280), "r5":(100, 280), "r6": (100, 280), "r7": (100, 280)}

#recipes = [120, 150, 180, 210, 240, 280, 300]

    optimizer = BayesianOptimization(f=black_box_function,
                                     pbounds=pbounds,
                                     random_state=1)
# logger = JSONLogger(path="./bo_logs.json")
# optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        acq="ei",
        xi=0.1,
        init_points=100,
        n_iter=100
    )
    print(optimizer.max)
    r1, r2, r3, r4, r5, r6, r7 = optimizer.max["params"]["r1"], \
                                 optimizer.max["params"]["r2"], \
                                 optimizer.max["params"]["r3"], \
                                 optimizer.max["params"]["r4"], \
                                 optimizer.max["params"]["r5"], \
                                 optimizer.max["params"]["r6"], \
                                 optimizer.max["params"]["r7"],

    recipes = [r1, r2, r3, r4, r5, r6, r7]

    recipe_list = { "Experiment1":[105, 130, 160, 190, 230, 270, 290],
               "Experiment2":[110, 140, 170, 200, 240, 280, 290],
                "Experiment3":[120, 150, 180, 220, 260, 280, 300]}

    recipe_list["BO"] = recipes
    df = pd.DataFrame()
    for k, v in recipe_list.items():
        print(f"", black_box_function(v[0], v[1], v[2], v[3], v[4], v[5], v[6]))

        inp = create_input(geom_num, v)
        inp = inp.cuda()
        inp_normalized = (inp - src_mean + 1e-5)/(src_sd+1e-5)
        inp_normalized = inp_normalized.type(torch.cuda.FloatTensor)
        steps = [33, 66, 99, 132, 171, 204, 214, 224,
                    234, 244, 254, 264, 274, 284, 294]

        with torch.no_grad():
            pred, _, _, _, _ = model(inp_normalized, model.model1)

        y = pred[0, :, :, :5, :5].mean((-1, -2)).cpu()
        df[k] = pd.Series(y.squeeze(1))
        plt.plot(steps, y, ".-", label=f"M{geom_num}_{k}")

    target_temp = torch.cuda.FloatTensor( [57, 90, 123, 156, 195, 228, 233, 238, 242, 247, 242, 238, 233, 228, 223]).unsqueeze(1).cpu()

    plt.plot(steps, target_temp, ".-", label="Target")

    df.to_csv("temp/bo_profile.csv")
    plt.ylabel("Temperature")
    plt.xlabel("Time Step")
    plt.legend()
    plt.savefig(f"Figure/BO_pred.png", dpi=300, bbox_inches='tight')
    plt.close("all")


if __name__ == '__main__':
    if opt.run_type=="test":
        test_trainer()
    elif opt.run_type=="validation":
        val_best_recipes()
    elif opt.run_type=="bo":
        bayesian_ops()
    else:
        run_trainer()
