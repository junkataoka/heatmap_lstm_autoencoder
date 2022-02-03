#!/usr/bin/zsh -l
#SBATCH --job-name=heat_generator
#SBATCH --output=heat_henerator_output.txt
#SBATCH --error=heat_generator_error.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpucompute
#SBATCH --mem=20GB

conda activate cyclegan
module load cuda11.1/toolkit/11.1.1

srun python preprocess.py
srun python main.py \
	--ngp=2
	--num_nodes=1

