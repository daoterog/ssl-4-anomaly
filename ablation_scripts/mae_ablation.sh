#!/bin/bash

# Check if batch_size argument is provided, else use default value
batch_size=${1:-256}

# Define defect proportions
defect_proportion=(0.45 0.35 0.25 0.15 0.05)

# Loop over each defect proportion
for defect_proportion in "${defect_proportion[@]}"
do
    echo "Running ablation study for $architecture with defect proportion $defect_proportion"

    python launch_job.py \
        --project "ssl_ablation" \
        --group "mae" \
        --model_type "mae-vit-tiny" \
        --batch_size $batch_size \
        --lr "1e-3" \
        --weight_decay "0.05" \
        --warmup_start_lr "5e-4" \
        --optimizer "adamw" \
        --imbalance_dataset "True" \
        --defect_proportion $defect_proportion \
        --use_sample "True" \
        --sample_percentage "0.5" \
        --learning_strategy "ssl" \
        --only_log_on_epoch_end "True" \
        --ssl_method "mae" \
        --decoder_depth "12" \
        --decoder_num_heads "12" \
        --decoder_embed_dim "192" \
        --norm_pix_loss "True"
done

echo "Ablation study completed."
