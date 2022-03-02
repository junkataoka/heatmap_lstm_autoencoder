#!/bin/zsh
# python preprocess.py
python main.py --n_gpus=1 --num_nodes=1 --batch_size=4 --epochs=500 --neptune_logger --lr=0.001 \
	--src_input_file=source_input.pt --src_target_file=source_target.pt \
	--tar_input_file=target_input.pt --tar_target_file=target_target.pt \
	--model_path=checkpoints/lstm_ac.ckpt --out_model_path=checkpoints/lstm_ac.ckpt


