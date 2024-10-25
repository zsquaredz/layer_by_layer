#!/bin/bash

# llama2 FLAN
while read task; do
    taskname=${task}_last_token 

    # llama2_sft_flan_all_50k
    for layer in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}
    do
        echo "currently doing llama2-all-50k vs task-${task} layer-${layer}"
        python analysis/cka.py \
            --data_dir1 ./output/llama2_sft_flan_all_50k/predictions/${taskname}/hidden_states_${layer}.npy \
            --data_dir2 ./output/llama2_sft_flan_${task}/predictions/${taskname}/hidden_states_${layer}.npy \
            --do_cka 
    done

    # llama2_no_sft, i.e., vanilla (pre-trained) llama2
    for layer in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}
    do
        echo "currently doing llama2-no-sft vs task-${task} layer-${layer}"
        python analysis/cka.py \
            --data_dir1 ./output/llama2_no_sft_flan/predictions/${taskname}/hidden_states_${layer}.npy \
            --data_dir2 ./output/llama2_sft_flan_${task}/predictions/${taskname}/hidden_states_${layer}.npy \
            --do_cka 
    done
    
done < ./flan_tasks.lst
