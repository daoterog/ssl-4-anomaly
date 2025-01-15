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
            --group "simclr" \
            --model_type $architecture \
            --batch_size $batch_size \
            --sync_batchnorm "True" \
            --lr "0.3" \
            --weight_decay "1e-6" \
            --warmup_start_lr "0.15" \
            --optimizer "lars" \
            --imbalance_dataset "True" \
            --defect_proportion $defect_proportion \
            --use_sample "True" \
            --sample_percentage "0.5" \
            --learning_strategy "ssl" \
            --only_log_on_epoch_end "True" \
            --ssl_method "simclr" \

    done
done

echo "Ablation study completed."
