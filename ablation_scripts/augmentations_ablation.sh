#!/bin/bash

# Check if resolution and batch_size arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <resolution> <batch_size>"
  exit 1
fi

resolution=$1
batch_size=$2

# Define architectures, RandomCrop min_scale, and ColorJitter t_val
architectures=("resnet18" "vit-tiny")
min_scales=(0.08 0.185 0.29 0.395 0.5)
t_vals=(0.1 0.275 0.45 0.625 0.8)

# Loop over each architecture
for architecture in "${architectures[@]}"
do
  # Loop over each RandomCrop min_scale
  for min_scale in "${min_scales[@]}"
  do
    # Loop over each ColorJitter t_val
    for t_val in "${t_vals[@]}"
    do
      echo "Running grid search for $architecture with resolution $resolution, batch size $batch_size, min_scale $min_scale, and t_val $t_val"

      # Calculate hue as t_val/2
      hue=$(echo "$t_val / 2" | bc -l)

      # Set up the command to run your model training/evaluation
      python launch_job.py \
        --project "augmentations_ablation" \
        --model_type $architecture \
        --img_size $resolution \
        --batch_size $batch_size \
        --min_scale $min_scale \
        --brightness $t_val \
        --contrast $t_val \
        --saturation $t_val \
        --hue $hue

      # You can add more commands here, like saving results, logging, etc.

    done
  done
done

echo "Grid search completed."
