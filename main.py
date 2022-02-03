# import libraries
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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--use_amp', default=False, type=bool, help='mixed-precision training')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--n_hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')

parser.add_argument('--root', type=str, default="./dataset")
parser.add_argument('--input_file', type=str, default="input.pt")
parser.add_argument('--target_file', type=str, default="./target.pt")
parser.add_argument('--time_steps', type=int, default=15)
parser.add_argument('--api_key', type=str, 
                    default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOTE0MGFjYy02NzMwLTRkODQtYTU4My1lNjk0YWEzODM3MGIifQ==")


opt = parser.parse_args()


class OvenLightningModule(pl.LightningModule):

    def __init__(self, opt, hparams=None, model=None):

        super(OvenLightningModule, self).__init__()
        self.save_hyperparameters()
        self.opt = opt

        self.normalize = False
        self.model = model

        # logging config
        self.log_images = True

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = self.opt.batch_size
        self.time_steps = self.opt.time_steps

    # def create_video(self, x, y_hat, y):
    #     # predictions with input for illustration purposes
    #     preds = torch.cat([x.cpu(), y_hat.unsqueeze(2).cpu()], dim=1)[0]

    #     # entire input and ground truth
    #     y_plot = torch.cat([x.cpu(), y.unsqueeze(2).cpu()], dim=1)[0]

    #     # error (l2 norm) plot between pred and ground truth
    #     difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
    #     zeros = torch.zeros(difference.shape)
    #     difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[
    #         0].unsqueeze(1)

    #     # concat all images
    #     final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

    #     # make them into a single grid image file
    #     grid = torchvision.utils.make_grid(final_image, nrow=self.n_steps_past + self.n_steps_ahead)

    #     return grid

    def forward(self, x):

        x = x.to(device='cuda')

        output = self.model(x, future_step=self.time_steps)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)  # is squeeze neccessary?

        loss = self.criterion(y_hat, y)

        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        # save predicted images every 250 global_step
        
        self.log("recon_loss", loss.item(), on_step=True, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.criterion(y_hat, y)}


    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        return avg_loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))




def run_trainer():
    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=4)

    model =OvenLightningModule(opt, model=conv_lstm_model)
    oven_data = TSDataModule(opt.root, opt.input_file, opt.target_file, opt.batch_size)
    neptune_logger = NeptuneLogger(
            api_key=opt.api_key,
                project_name='junkataoka/heatmap',
                        #params={'max_epochs': 10},  # Optional,
                            #tags=['pytorch-lightning', 'mlp']  # Optional,
                            )

    trainer = Trainer(max_epochs=opt.epochs,
                        gpus=opt.n_gpus,
                        logger=neptune_logger,
                        distributed_backend='ddp',
                        num_nodes=opt.num_nodes
                    #   early_stop_callback=False,
                    #    fast_dev_run = True


                      )

    trainer.fit(model, datamodule=oven_data)


if __name__ == '__main__':
    run_trainer()
    # p1 = Process(target=run_trainer)                    # start trainer
    # p1.start()
    # p2 = Process(target=run_tensorboard(new_run=True))  # start tensorboard
    # p2.start()
    # p1.join()
    # p2.join()