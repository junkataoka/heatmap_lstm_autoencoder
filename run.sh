#!/bin/zsh
# python preprocess.py
python main.py --n_gpus=1 --num_nodes=1 --batch_size=8 --epochs=100 --neptune_logger
# python postprocess.py
