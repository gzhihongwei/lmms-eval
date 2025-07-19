# python -m lmms_eval --model qwen2_5_omni --model_args device_map=auto,modality=video \
#                     --tasks mmug_dev --batch_size 1 --log_samples --log_samples_suffix qwen25omni \
#                     --output_path logs/qwen-out-final --limit 100

# python3 -m accelerate.commands.launch \
#         --num_processes=1 \

# export OUTDIR="logs/logs_$MODEL"
export MODEL="ola"
export TASK="audio_only"
export OUTDIR="logs/logs_ola"
CUDA_VISIBLE_DEVICES=5 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29535 --nproc_per_node=1 -m lmms_eval \
        --model $MODEL \
        --tasks maverix_$TASK \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix maverix_${MODEL}_${TASK} \
        --output_path ./$OUTDIR/${TASK}/ \
        # --log_samples_suffix maverix_gpt4omini_${TASK} \
        # --limit 20 \
        # --verbosity=DEBUG

#to do debug ola audio only. all gpt4o and mini done