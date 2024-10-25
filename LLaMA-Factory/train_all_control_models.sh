#!/bin/bash

script="train_control_model.sh"

# Iterate over each value and run the script
while read task; do
    # Modify the script to set the variable task to the current value
    sed -i "s/task=.*/task=\"$task\"/" "$script"
    
    # run the script
    ./$script
    
    # You can also submit the SLURM script
    # In this case, you need to modify the script 'train_control_model.sh' to be a SLURM script
    # sbatch "$script"
    
    echo $task
    
done < flan_tasks.lst
