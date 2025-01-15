#!/bin/bash

# Check if batch_size argument is provided, else use default value
batch_size=${1:-256}

# Define architectures and image resolutions
architectures=("resnet18" "vit-tiny")
resolutions=(64 128 224 384 512)

# Loop over each architecture
for architecture in "${architectures[@]}"
do
  # Loop over each resolution
  for resolution in "${resolutions[@]}"
  do
    echo "Running ablation study for $architecture with resolution $resolution"
    
    # Set up the command to run your model training/evaluation
    # Replace the following line with your actual command
    python launch_job.py \
      --project "resolution_ablation" \
      --model_type $architecture \
      --img_size $resolution \
      --batch_size $batch_size \
      --use_random_resized_crop "False" \
      --use_color_jitter "False"

    # You can add more commands here, like saving results, logging, etc.

  done
done

echo "Ablation study completed."
