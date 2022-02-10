#!/usr/bin/zsh -l
#SBATCH --job-name=heat_generator
#SBATCH --output=heat_henerator_output.txt
#SBATCH --error=heat_generator_error.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpucompute
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

module load cuda11.1/toolkit/11.1.1

# srun python preprocess.py
# srun python main.py --n_gpus=2 --num_nodes=4 --batch_size=16 --epochs=1000 --log_images --is_distributed --n_hidden_dim=64 --time_steps=15 --retrain --lr=0.001
srun python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=1 --log_images --test


