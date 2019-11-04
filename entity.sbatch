#!/bin/bash
#SBATCH --job-name=entity
#SBATCH --output=../output/%j.out
#SBATCH --error=../output/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=4:00:00

#SBATCH --mail-user=xuaxuadenpy@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

source ~/.bashrc
conda activate torch
module load cuda/10.0.130
export CUDA_VISIBLE_DEVICES=0
uid=$RANDOM
echo $uid
python main.py --cuda 0 \
    --batch_size 48 \
    --ce_loss_scale 0.5 \
    --lr 3e-5 \
    --info entity0.5ce \
    --latent \
    --mask_mode entity \
    --max_epoch 3

