#!/bin/bash
#SBATCH --job-name dialog-breakdown-research-gpt
#SBATCH --output ./slurm-log/%x-%j.log
#SBATCH --error ./slurm-log/%x-%j.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8 --gpus tesla:1 
#SBATCH --time 1440:00

make run-gpt $1 $2
