#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --output=output.txt 
#SBATCH --error=error.txt

module load cuda/12.2
conda activate torch24
cd /ibex/user/$USER/data/RegionSegProject
python DeepLabV3.py