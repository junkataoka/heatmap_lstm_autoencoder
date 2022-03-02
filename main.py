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
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--retrain', action='store_true', help='Whether to retrain the model or not')
parser.add_argument('--neptune_logger', action='store_true', help='Whether to use neptune.ai logger')
parser.add_argument('--api_key', type=str,
                    default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOTE0MGFjYy02NzMwLTRkODQtYTU4My1lNjk0YWEzODM3MGIifQ==")

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
        self.model = EncoderDecoderConvLSTM(nf=self.opt.n_hidden_dim, in_chan=4)
        self.log_images = self.opt.log_images
        self.criterion = torch.nn.MSELoss()
        self.dcl_criterion = torch.nn.NLLLoss()
        self.batch_size = self.opt.batch_size
        self.time_steps = self.opt.time_steps
        self.epoch = 0
        self.step = 0

    def load_model(self):

        self.model.load_state_dict(torch.load(self.opt.model_path, map_location='cuda:0'), strict=False)
        print('Model Created!')

    def create_video(self, x, y_hat, y):

        b, t, c, h, w = x.shape
        x_t = x.cpu()
        x_t = x_t[1, 1, :, :, :]
        x_grid = torchvision.utils.make_grid(x_t, nrow=t)

        b, t, c, h, w = y.shape
        y_t = y[1, :, :, :, :]
        y_hat_t = y_hat[1, :, :, :, :]
        y_grid = torchvision.utils.make_grid(y_t.cpu(), nrow=t)
        y_hat_grid = torchvision.utils.make_grid(y_hat_t.cpu(), nrow=t)

        return x_grid, y_grid, y_hat_grid

    def forward(self, x):

        output, dcl_output = self.model(x, future_step=self.time_steps)

        return output, dcl_output

    def training_step(self, batch, batch_idx):

        src_batch = batch[0]
        tar_batch = batch[1]
        src_x, src_y = src_batch
        tar_x, tar_y = tar_batch

        src_y_hat, src_dcl_y_hat = self.forward(src_x)
        tar_y_hat, tar_dcl_y_hat = self.forward(tar_x)

        src_label = torch.zeros(src_x.shape[0]).long().cuda()
        tar_label = torch.ones(tar_x.shape[0]).long().cuda()

        src_loss = self.criterion(src_y_hat, src_y)
        tar_loss = self.criterion(tar_y_hat, tar_y)
        b ,t, c, h, w = src_y_hat.shape

        mmd_loss = MMD(src_y_hat.reshape(b, -1), tar_y_hat.reshape(b, -1), kernel="multiscale")

        avg_diff_src_src = torch.mean(torch.abs(src_y_hat - src_y))

        avg_diff_tar_tar = torch.mean(torch.abs(tar_y_hat - tar_y))

        self.log("src_loss", src_loss.item(), on_step=False, on_epoch=True)
        self.log("tar_loss", tar_loss.item(), on_step=False, on_epoch=True)
        self.log("mmd_loss", mmd_loss.item(), on_step=False, on_epoch=True)

        self.log("avg_diff_src_src", avg_diff_src_src.item(), on_step=False, on_epoch=True)

        self.log("avg_diff_tar_tar", avg_diff_tar_tar.item(), on_step=False, on_epoch=True)

        if self.log_images:
            x_grid, y_grid, y_hat_grid = self.create_video(src_x, src_y_hat, src_y)
            fname = 'epoch_' + str(self.current_epoch+1) + '_step' + str(self.global_step)

            figure, ax = plt.subplots(1, 1, figsize=(18, 3))
            ax.imshow(y_hat_grid.permute(1,2,0))
            figure.suptitle("pred_"+fname, fontsize=16)
            self.logger.experiment.log_image("pred", figure)
            plt.clf()
            plt.cla()

            figure, ax = plt.subplots(1, 1, figsize=(18, 3))
            ax.imshow(y_grid.permute(1,2,0))
            figure.suptitle("target_"+fname, fontsize=16)
            self.logger.experiment.log_image("target", figure)
            plt.clf()
            plt.cla()

            figure, ax = plt.subplots(1, 1, figsize=(18, 3))
            ax.imshow(x_grid.permute(1,2,0))
            figure.suptitle("input_"+fname, fontsize=16)
            self.logger.experiment.log_image("input", figure)
            plt.clf()
            plt.cla()

        loss = src_loss  + tar_loss + mmd_loss

        return loss

    # def validation_step(self, batch, batch_idx, dataloader_idx):

        # src_batch = batch[0]
        # tar_batch = batch[1]
        # src_x, src_y = src_batch
        # tar_x, tar_y = tar_batch

        # src_y_hat, src_dcl_y_hat = self.forward(src_x)
        # tar_y_hat, tar_dcl_y_hat = self.forward(tar_x)

        # src_label = torch.zeros(src_x.shape[0]).cuda()
        # tar_label = torch.ones(tar_x.shape[0]).cuda()

        # src_loss = self.criterion(src_y_hat, src_y)
        # tar_loss = self.criterion(tar_y_hat, tar_y)
        # src_dcl_loss =self.dcl_criterion(src_dcl_y_hat, src_label)
        # tar_dcl_loss =self.dcl_criterion(tar_dcl_y_hat, tar_label) 
        # self.log("val_src_loss", src_loss.item(), on_step=False, on_epoch=True)
        # self.log("val_tar_loss", tar_loss.item(), on_step=False, on_epoch=True)
        # self.log("val_src_dcl_loss", src_dcl_loss.item(), on_step=False, on_epoch=True)
        # self.log("val_tar_dcl_loss", tar_dcl_loss.item(), on_step=False, on_epoch=True)

        # self.log("val_avg_diff_src_tar", avg_diff_src_tar.item(), on_step=False, on_epoch=True)
        # self.log("val_avg_diff_src_src", avg_diff_src_src.item(), on_step=False, on_epoch=True)

        # self.log("val_avg_diff_tar_tar", avg_diff_tar_tar.item(), on_step=False, on_epoch=True)
        # self.log("val_avg_diff_tar_src", avg_diff_tar_src.item(), on_step=Falalexnet_resnet_finetune

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 100 , gamma = 0.5)

        # return {"optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler":sch,
        #             "monitor": "loss"}}
        return optimizer
def test_trainer():
    model =OvenLightningModule(opt).cuda()
    oven_data = TSDataModule(opt, opt.root, opt.src_input_file, opt.src_target_file, opt.tar_input_file, opt.tar_target_file, batch_size=1)
    oven_data.setup()
    source_loader, target_loader = oven_data.val_dataloader()
    model.load_model()
    model.eval()
    err = torch.zeros(15).cuda(0)
    for idx, batch in enumerate(target_loader):
        inp, target = batch
        with torch.no_grad():
            predictions, _ = model(inp)
            diff = torch.mean(torch.abs(target - predictions), (0,  2, 3, 4))
            err += diff

            plt.figure(figsize=(12, 8))
            plt.plot(target[0, :, :25, :25].mean((-1, -2)).cpu(), label="Target")
            plt.plot(predictions[0, :, :25, :25].mean((-1, -2)).cpu(), label="Prediction")
            plt.ylabel("Temperature", fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("Time Step", fontsize=16)
            plt.xticks(fontsize=16)
            plt.legend(fontsize=16)
            plt.ylim([220, 280])
            plt.savefig("Figure/sample.png", dpi=300, bbox_inches='tight')

            input()

    step_err = err / len(target_loader)


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
                        multiple_trainloader_mode="min_size"
                      )

    if opt.retrain:
        model.load_model()

    trainer.fit(model, datamodule=oven_data)
    torch.save(model.model.state_dict(), opt.out_model_path)


if __name__ == '__main__':
    if opt.test:
        test_trainer()

    else:
        run_trainer()
