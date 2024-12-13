#!/bin/bash
#SBATCH -p general ## Partition
#SBATCH -q public  ## QOS
#SBATCH -c 2       ## Number of Cores
#SBATCH --gres=gpu:a:30:0
#SBATCH --time=30   ## 8 hours of compute
#SBATCH --job-name=comp-epi-split
#SBATCH --output=slurm.%j.out  ## job /dev/stdout record (%j expands -> jobid)
#SBATCH --error=slurm.%j.err   ## job /dev/stderr record 
#SBATCH --export=NONE          ## keep environment clean
#SBATCH --mail-type=ALL        ## notify <asurite>@asu.edu for any job state change

module load mamba/latest

source activate test6

cd /home/ccastron/CSE559/ATM-TCR

CUDA_VISIBLE_DEVICES=0 python main.py \
    --infile data/combined_dataset.csv \
    --indepfile data/testing_data/epitope_split_test.csv \
    --min_epoch 1 \
    --epoch 1 \
    --batch_size 64 \
    --results_dir "/home/ccastron/CSE559/ATM-TCR/results/run_2_default_epi.png" \
    --lr 0.0001 \
    --lr_drop_factor 0.1 \
    --split_type epitope \
    --mode test \
    --model cross_attention
