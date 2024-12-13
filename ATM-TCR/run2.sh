cd /home/ccastron/CSE559/ATM-TCR

CUDA_VISIBLE_DEVICES=0 python main.py \
    --infile data/combined_dataset.csv \
    --indepfile data/testing_data/tcr_split_test.csv \
    --min_epoch 1 \
    --epoch 1 \
    --batch_size 64 \
    --results_dir "/home/ccastron/CSE559/ATM-TCR/results/run_2_default_epi.png" \
    --lr 0.0001 \
    --lr_drop_factor 0.1 \
    --split_type tcr \
    --mode test \
    --model cross_attention
