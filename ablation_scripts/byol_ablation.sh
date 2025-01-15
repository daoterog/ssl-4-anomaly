#!/bin/bash

# Check if batch_size argument is provided, else use default value
batch_size=${1:-256}

# Define architectures and defect proportions
architectures=("resnet18" "vit-tiny")
defect_proportion=(0.45 0.35 0.25 0.15 0.05)

# Loop over each defect proportion
for defect_proportion in "${defect_proportion[@]}"
do
    for architecture in "${architectures[@]}"
    do
        echo "Running ablation study for $architecture with defect proportion $defect_proportion"

        python launch_job.py \
            --project "ssl_ablation" \
            --group "byol" \
            --model_type $architecture \
            --batch_size $batch_size \
            --lr "0.2" \
            --weight_decay "1.5e-6" \
            --warmup_start_lr "0.1" \
            --optimizer "lars" \
            --exclude_bias_and_norm "True" \
            --imbalance_dataset "True" \
            --defect_proportion $defect_proportion \
            --use_sample "True" \
            --sample_percentage "0.5" \
            --learning_strategy "ssl" \
            --only_log_on_epoch_end "True" \
            --ssl_method "byol" \
            --normalize_projector "False"

    done
done

echo "Ablation study completed."
