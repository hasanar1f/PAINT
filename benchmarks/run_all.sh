#!/bin/bash

# Navigate to the benchmarks directory
cd benchmarks

# List of scripts to run
scripts=(
    "llava_1.6_mistral_7b.sh"
    "llava_1.6_vicuna_7b.sh"
)

# Loop through each script and execute it
for script in "${scripts[@]}"; 
do
  echo "Running $script..."
  bash "$script"
done

echo "All benchmark scripts have been executed."
