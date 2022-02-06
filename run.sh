#!/bin/zsh
python preprocess.py
python main.py --n_gpus=1 --num_nodes=1 --batch_size=18 --epochs=300 --log_images
