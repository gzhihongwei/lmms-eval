python -m lmms_eval --model grok2-vision --model_args device_map=auto,modality=video \
                    --tasks mmug_dev --batch_size 1 --log_samples --log_samples_suffix grok2 \
                    --output_path logs/grok2-out-final --limit 100