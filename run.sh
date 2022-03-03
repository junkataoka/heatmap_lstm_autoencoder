#!/bin/zsh
# python preprocess.py

for i in {1..10}
	do
		python main.py --n_gpus=1 --num_nodes=1 --batch_size=4 --epochs=200 --neptune_logger --lr=0.001 \
			--src_input_file=train_input_fold$i.pt --src_target_file=train_target_fold$i.pt \
			--tar_input_file=test_input_fold$i.pt --tar_target_file=test_target_fold$i.pt \
			--model_path=checkpoints/lstm_ac_fold$i.ckpt --out_model_path=checkpoints/lstm_ac_fold$i.ckpt
	done

