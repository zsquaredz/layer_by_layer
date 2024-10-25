#!/bin/bash

task="aeslc"

# vanilla pre-trained experimental model with no instructions tuned
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
   --stage sft \
   --do_predict \
   --model_name_or_path /path/to/llama2/hf/model/Llama-2-7b-hf/ \
   --dataset ${task}_test \
   --template llama2 \
   --finetuning_type full \
   --output_dir ./output/llama2_no_sft_flan/predictions/ \
   --per_device_eval_batch_size 1 \
   --max_samples 10000 \
   --predict_with_generate \
   --output_hidden_states \
   --output_task_name ${task}_last_token \
   --fp16


# experimental model finetuned on all instructions
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
   --stage sft \
   --do_predict \
   --model_name_or_path /path/to/llama2/hf/model/Llama-2-7b-hf/ \
   --adapter_name_or_path ./output/llama2_sft_flan_all_50k \
   --dataset ${task}_test \
   --template llama2 \
   --finetuning_type lora \
   --output_dir ./output/llama2_sft_flan_all_50k/predictions/ \
   --per_device_eval_batch_size 1 \
   --max_samples 10000 \
   --predict_with_generate \
   --output_hidden_states \
   --output_task_name ${task}_last_token \
   --fp16