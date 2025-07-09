# python -m lmms_eval --model qwen2_5_omni --model_args device_map=auto,modality=video \
#                     --tasks mmug_dev --batch_size 1 --log_samples --log_samples_suffix qwen25omni \
#                     --output_path logs/qwen-out-final --limit 100

# python3 -m accelerate.commands.launch \
#         --num_processes=1 \

MODEL="egogpt"

torchrun --nproc_per_node=1 -m lmms_eval \
        --model $MODEL \
        --tasks mmug_all \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix mmug_$MODEL \
        --output_path ./logs/ \
        --limit 2 \
        --verbosity=DEBUG
