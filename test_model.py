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
from main import OvenLightningModule

parser = argparse.ArgumentParser()

parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
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
parser.add_argument('--input_file', type=str, default="input.pt")
parser.add_argument('--target_file', type=str, default="./target.pt")
parser.add_argument('--time_steps', type=int, default=15)
parser.add_argument('--api_key', type=str, 
                    default="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOTE0MGFjYy02NzMwLTRkODQtYTU4My1lNjk0YWEzODM3MGIifQ==")

parser.add_argument('--model_path', type=str, default="checkpoints/lstm_ac.ckpt")

opt = parser.parse_args()

def run_trainer():
    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=4)

    neptune_logger = NeptuneLogger(
            api_key=opt.api_key,
                project_name='junkataoka/heatmap',
                        #params={'max_epochs': 10},  # Optional,
                            #tags=['pytorch-lightning', 'mlp']  # Optional,
                            )
    trainer = Trainer(max_epochs=1,
                        gpus=1,
                        logger=neptune_logger,
                        accelerator='ddp',
                        num_nodes=1

                      )

    if opt.test:
        model =OvenLightningModule.load_from_checkpoint(checkpoint_path=f"{opt.model_path}")

    else:
        model =OvenLightningModule(opt, model=conv_lstm_model)
    oven_data = TSDataModule(opt, opt.root, opt.input_file, opt.target_file, opt.batch_size)


    trainer.fit(model, datamodule=oven_data)
    trainer.save_checkpoint("checkpoints/lstm_ac.ckpt")


if __name__ == '__main__':
    run_trainer()