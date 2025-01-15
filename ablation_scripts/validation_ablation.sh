#!/bin/bash

# Check if batch_size argument is provided, else use default value
batch_size=${1:-256}

# Path to JSON file containing wandb_ids
json_file="ablation_scripts/ssl_ablation_run_mapping.json"

# Define SSL methods, architectures, and defect proportions
ssl_methods=("simclr" "barlow_twins" "byol", "supervised")
architectures=("resnet18" "vit-tiny")
defect_proportions=(0.45 0.15 0.05 0.02 0.01)

# Iterate through each SSL method, architecture, and defect proportion
for ssl_method in "${ssl_methods[@]}"
do
    for architecture in "${architectures[@]}"
    do
        for defect_proportion in "${defect_proportions[@]}"
        do
            for test_defect_proportion in "${defect_proportions[@]}"
            do
                # Extract corresponding wandb_id from JSON file
                runid=$(jq -r ".${ssl_method}.${architecture}[\"$defect_proportion\"]" "$json_file")

                # Check if wandb_id was found
                if [ "$runid" != "null" ]; then
                    echo "Running ablation study for $ssl_method with architecture $architecture and defect proportion $defect_proportion (wandb_id: $runid)"

                    # Run the Python script with the extracted wandb_id
                    python launch_job.py \
                        --job_type "inference" \
                        --learning_strategy "supervised" \
                        --project "val_ablation" \
                        --load_model "True" \
                        --wandb_project "ssl_ablation" \
                        --wandb_id "$runid" \
                        --imbalance_dataset "True" \
                        --defect_proportion "$test_defect_proportion" \
                        --group "$ssl_method" \
                        --tags "$ssl_method" "$architecture" "$defect_proportion" \
                        --batch_size "$batch_size" \
                        --only_log_on_epoch_end "True" \
                        --use_color_jitter "False" \
                        --use_random_resized_crop "False"
                else
                    echo "wandb_id not found for $ssl_method with architecture $architecture and defect proportion $defect_proportion"
                fi
            done
        done
    done
done


for defect_proportion in "${defect_proportions[@]}"
do
    for test_defect_proportion in "${defect_proportions[@]}"
    do
        # Extract corresponding wandb_id from JSON file
        runid=$(jq -r ".mae[\"mae-vit-tiny\"][\"$defect_proportion\"]" "$json_file")

        # Check if wandb_id was found
        if [ "$runid" != "null" ]; then
            echo "Running ablation study for $ssl_method with architecture $architecture and defect proportion $defect_proportion (wandb_id: $runid)"

            # Run the Python script with the extracted wandb_id
            python launch_job.py \
                --job_type "inference" \
                --learning_strategy "supervised" \
                --project "val_ablation" \
                --load_model "True" \
                --wandb_project "ssl_ablation" \
                --wandb_id "$runid" \
                --imbalance_dataset "True" \
                --defect_proportion "$test_defect_proportion" \
                --group "mae" \
                --tags "mae" "mae-vit-tiny" "$defect_proportion" \
                --batch_size "$batch_size" \
                --only_log_on_epoch_end "True" \
                --use_color_jitter "False" \
                --use_random_resized_crop "False"
        else
            echo "wandb_id not found for $ssl_method with architecture $architecture and defect proportion $defect_proportion"
        fi
    done
done


echo "Ablation study completed."
