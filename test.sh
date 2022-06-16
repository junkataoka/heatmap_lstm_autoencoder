#!/bin/zsh


python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 --source_only \
			--src_input_file=source_input.pt --src_target_file=source_target.pt \
			--tar_input_file=target_input_cv1_test.pt --tar_target_file=target_target_cv1_test.pt \
			--model_path checkpoints/lstm_ac_reg_mmd_cv1_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv1_noda.ckpt --run_type test

python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
			--src_input_file=source_input.pt --src_target_file=source_target.pt \
			--tar_input_file=target_input_cv1_test.pt --tar_target_file=target_target_cv1_test.pt \
			--model_path checkpoints/lstm_ac_reg_mmd_cv1.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv1.ckpt --run_type test

python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 --source_only \
			--src_input_file=source_input.pt --src_target_file=source_target.pt \
			--tar_input_file=target_input_cv2_test.pt --tar_target_file=target_target_cv2_test.pt \
			--model_path checkpoints/lstm_ac_reg_mmd_cv2_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv2_noda.ckpt --run_type test

python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
			--src_input_file=source_input.pt --src_target_file=source_target.pt \
			--tar_input_file=target_input_cv2_test.pt --tar_target_file=target_target_cv2_test.pt \
			--model_path checkpoints/lstm_ac_reg_mmd_cv2.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv2.ckpt --run_type test

python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 --source_only \
			--src_input_file=source_input.pt --src_target_file=source_target.pt \
			--tar_input_file=target_input_cv3_test.pt --tar_target_file=target_target_cv3_test.pt \
			--model_path checkpoints/lstm_ac_reg_mmd_cv3_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv3_noda.ckpt --run_type test

python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
			--src_input_file=source_input.pt --src_target_file=source_target.pt \
			--tar_input_file=target_input_cv3_test.pt --tar_target_file=target_target_cv3_test.pt \
			--model_path checkpoints/lstm_ac_reg_mmd_cv3.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv3.ckpt --run_type test