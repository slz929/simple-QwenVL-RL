export DEBUG_MODE="true"
export LOG_PATH="./debug_log_GRPO.txt"

export CKPT_PATH=./Qwen/Qwen2.5-VL-3B-Instruct
export SAVE_PATH=./save_train/Qwen2.5-VL-3B-grpo-exp

export DATA_PATH=./demo_data/vsr_cot_train_10.jsonl
export IMG_PATH=./demo_data/vsr

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12789" \
    src/virft/src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --train_data_path ${DATA_PATH} \
    --train_image_folder_path ${IMG_PATH} \
    --deepspeed src/virft/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --max_pixels 401408 \
    --num_train_epochs 10 \
    --run_name QwenVL_GRPO_exp \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 8 \
    --max_completion_length 1024 \
    --temperature 1.0 \
    --beta 0.04 

