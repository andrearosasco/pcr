#!/bin/bash

srun --partition=main --ntasks=1 --nodes=1 --nodelist=gnode04 --pty --gres=gpu:1 --cpus-per-task=32 --mem=40G^
 singularity shell --nv ~/.images/pcr_final.sif

