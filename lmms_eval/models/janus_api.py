import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import List, Tuple

import numpy as np
import requests as url_requests
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from PIL import Image

@register_model("janus_api")
class NovaAPI(lmms):
    def __init__(
        self,
        model_version: str = "janus-pro-7b",
        # modality: str = "image",
        # We will cache the Gemini API response in this path and use it for future requests
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.model = partial(client.invoke_model, modelId=model_version)
        
        # specify the path to the model
        if model_version == "janus-pro-7b":
            model_path = "deepseek-ai/Janus-Pro-7B"
        elif model_version == "janus-1.3b":
            model_path = "deepseek-ai/Janus-1.3B"
        else:
            raise ValueError(f"Model version {model_version} is not supported")
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        pil_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            pil_frames.append(img)

        return pil_frames

    def convert_modality(self, images):
        for idx, img in enumerate(images):
            if isinstance(img, str):  # video
                try:
                    images[idx] = self.encode_video(img)
                except Exception as e:
                    eval_logger.error(f"Error converting video: {str(e)}")
        return images

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                # This is the lowest value of temperature accepted?
                gen_kwargs["temperature"] = 0.00001
                
            inference_config_allowed_keys = {"max_new_tokens", "temperature", "top_p", "top_k", "stopSequences", "toolConfig"}
            inference_config = {key: value for key, value in gen_kwargs.items() if key in inference_config_allowed_keys}

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            # TODO: need a way to determine if it is an image or video
            visuals = self.convert_modality(visuals)

            messages = [{"role": "user", "content": [], "images": []}]
            
            # FIXME: assuming video for now
            # for visual_format, visual in visuals:
            #     messages[0]["content"].append({"video": {"format": visual_format, "source": {"bytes": visual}}})
            # messages[0]["content"].append({"text": contexts})

            for visual_format, visual in visuals:
                messages[0]["content"].append({"video": {"format": visual_format, "source": {"bytes": visual}}})
            messages[0]["images"].append(visuals)

            prepare_inputs = vl_chat_processor(
                    conversations=messages, images=visuals, force_batchify=True
                ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
                
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # native_request = {
            #     "schemaVersion": "messages-v1",
            #     "messages": messages,
            #     # TODO: add system prompt
            #     # "system": system_list,
            #     "inferenceConfig": inference_config
            # }

            for attempt in range(1):
                try:
                    # # run the model to get the response
                    outputs = vl_gpt.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=512,
                        do_sample=False,
                        use_cache=True,
                    )
                    content = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

                    # response = self.model(body=json.dumps(native_request))
                    # model_response = json.loads(response["body"].read())
                    # content = model_response["output"]["message"]["content"][0]["text"]
                    
                    # TODO: contains an `error` field 
                    break
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    # TODO: add `ValidationException` for capturing feedback
                    if isinstance(e, ValueError):
                        try:
                            eval_logger.info(f"Prompt feed_back: {content.prompt_feedback}")
                            content = ""
                            break
                        except Exception:
                            pass
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 1 attempts failed. Last error message: {str(e)}")
                        content = ""
            res.append(content)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Janus API")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Janus API does not support"