export MLP_WORKER_GPU=8
export MLP_WORKER_NUM=1
export MLP_WORKER_0_HOST="127.0.0.1"
export MLP_WORKER_0_PORT=12345
export MLP_ROLE_INDEX=0

export LAUNCHER="pytorch"
OUTPUT_DIR='/fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/checkpoints/checkpoint-1.7b-nothink-0505'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

python -m torch.distributed.launch --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
  --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
  /fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/internvl/train/pretrain_mem.py \
  --vision_path "/fs-computility/ai-shen/mllm_safety-shared/models/huggingface/InternViT-300M-448px-V2_5/" \
  --llm_path "/fs-computility/ai-shen/mllm_safety-shared/models/huggingface/Qwen/Qwen3-1.7B/" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/shell/data/pretrain_data.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --min_num_frame 8 \
  --max_num_frame 32 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 8 \
  --bf16 True \
  --max_steps 10000 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 2 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "/fs-computility/ai-shen/mllm_safety-shared/projects/mllm-reasoning/Qwen3-VL/internvl_chat/zero_stage3_config.json" \
  --report_to "tensorboard" \
  --use_packed_ds True \
  --num_images_expected 48 \
  --max_packed_tokens 16384 \
  --max_buffer_size 20 \
  --log_freq 1000 \
  --strict_mode False \
  --replacement False \
  --allow_overflow False \
  --remove_unused_columns False \
  --loss_reduction "square" \
  --loss_reduction_all_gather True \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
