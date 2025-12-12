#!/bin/bash
# Set the following environment variables or define paths in this file
# export DATA_ROOT=/path/to/your/data
# export MODEL_ROOT=/path/to/your/models
# export OUTPUT_ROOT=/path/to/your/output

# Data paths (use environment variables, fallback to relative paths if not set)
DATA_ROOT=${DATA_ROOT:-./data}
MODEL_ROOT=${MODEL_ROOT:-./models}
OUTPUT_ROOT=${OUTPUT_ROOT:-./output}

aime_train_path=${DATA_ROOT}/aime/aime_train.parquet
aime_test_path=${DATA_ROOT}/aime/aime_test.parquet
geo3k_train_path=${DATA_ROOT}/geo3k/geometry3k_train.parquet
geo3k_test_path=${DATA_ROOT}/geo3k/geometry3k_test.parquet

train_text_files="['$aime_train_path']"
train_mm_files="['$geo3k_train_path']"
test_files="['$aime_test_path','$geo3k_test_path']"

policy_path=${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct
rollout_batch_size=480
n_samples_per_prompts=8
episode=5
temperature=0.7
batch_size=16
lr=1e-6
kl_loss_coef=0.0
kl_coef=0.001
entropy_coeff=0
max_gen_length=4096
dataset_name=text_vision_combined
run_name=rl.arm2.vl.3b_${dataset_name}
samples_save_path=${OUTPUT_ROOT}/samples/$run_name

python3 -m verl.trainer.main_ppo \
    ray_init.num_cpus=52 \
    algorithm.adv_estimator=ada_grpo_w_length_penalty \
    algorithm.use_tool_based_gen=True \
    data.train_text_files="$train_text_files" \
    data.train_mm_files="$train_mm_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$rollout_batch_size \
    data.val_batch_size=1312 \
    data.max_prompt_length=2048 \
    data.max_response_length=$max_gen_length \
    actor_rollout_ref.model.path=$policy_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=96 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$n_samples_per_prompts \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=arm2 \
    trainer.experiment_name=${run_name} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.val_before_train=True \
    trainer.default_local_dir=${OUTPUT_ROOT}/checkpoints/${run_name} \
    trainer.resume_mode=auto \
    trainer.samples_save_path=$samples_save_path \
    trainer.total_epochs=$episode $@