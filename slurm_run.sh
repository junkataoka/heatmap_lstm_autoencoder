#!/usr/bin/zsh -l
#SBATCH --job-name=heat_generator
#SBATCH --output=heat_henerator_output.txt
#SBATCH --error=heat_generator_error.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpucompute
#SBATCH --mem=20GB
#SBATCH --gres=gpu:2

module load cuda11.1/toolkit/11.1.1

srun python preprocess.py
# srun python main.py --n_gpus=2 --num_nodes=4 --batch_size=16 --epochs=1000 --log_images --is_distributed --n_hidden_dim=64 --time_steps=15 --retrain --lr=0.001
# srun python preprocess.py
srun python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 --is_distributed \
			--src_input_file=source_input.pt --src_target_file=source_target.pt \
			--tar_input_file=tar_x_train.pt --tar_target_file=tar_y_train.pt \
			--model_path checkpoints/lstm_ac_reg_mmd_exp3.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_exp3.ckpt

# srun python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=tar_x_train.pt --tar_target_file=tar_y_train.pt \
# 			--model_path checkpoints/lstm_ac_reg_mmd_exp2.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_exp2.ckpt --run_type bo

# srun python main.py --n_gpus=1 --num_nodes=1 --batch_size=2 --epochs=100 --neptune_logger --lr=0.001 \
# 			--src_input_file=x_train.pt --src_target_file=y_train.pt \
# 			--tar_input_file=x_test.pt --tar_target_file=y_test.pt \
#  			--model_path checkpoints/lstm_ac_cnn_reg_mmd.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_mmd.ckpt