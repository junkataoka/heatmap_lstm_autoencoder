#!/bin/zsh
# python preprocess.py

# for i in {1..10}
# 	do
# 		python main.py --n_gpus=1 --num_nodes=1 --batch_size=4 --epochs=200 --neptune_logger --lr=0.001 \
# 			--src_input_file=train_input_fold$i.pt --src_target_file=train_target_fold$i.pt \
# 			--tar_input_file=test_input_fold$i.pt --tar_target_file=test_target_fold$i.pt \
# 			--model_path=checkpoints/lstm_ac_fold$i.ckpt --out_model_path=checkpoints/lstm_ac_fold$i.ckpt \
# 			--val_recipe
# 	done


# python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=100 --neptune_logger --lr=0.001 \
# 			--src_input_file=x_train.pt --src_target_file=y_train.pt \
# 			--tar_input_file=x_val.pt --tar_target_file=y_val.pt \
# 			--model_path checkpoints/lstm_ac_attention_mmd.ckpt --out_model_path checkpoints/lstm_ac_attention_mmd.ckpt

python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
			--src_input_file=source_input.pt --src_target_file=source_target.pt \
			--tar_input_file=tar_x_train.pt --tar_target_file=tar_y_train.pt \
			--model_path checkpoints/lstm_ac_reg_mmd_exp4.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_exp4.ckpt --run_type bo