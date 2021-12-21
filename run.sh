#!/bin/bash

#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=main

cd ~/projects/pcr
singularity run --nv ~/.images/pcr_final.sif main.py > train-output.txt 2>&1
