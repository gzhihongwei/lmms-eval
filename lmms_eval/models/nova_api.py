import io
import base64
import json
import os
import pathlib
import re
import time
from functools import partial
from typing import List, Tuple

import datasets
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    import boto3
    
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    NUM_SECONDS_TO_SLEEP = 30
except Exception as e:
    eval_logger.error(f"Error importing boto3: {str(e)}")
    genai = None


@register_model("nova_api")
class NovaAPI(lmms):
    def __init__(
        self,
        model_version: str = "us.amazon.nova-lite-v1:0",
        # modality: str = "image",
        # We will cache the Gemini API response in this path and use it for future requests
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.model = partial(client.invoke_model, modelId=model_version)
        # self.modality = modality

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def encode_video(self, video_path):
        with open(video_path, "rb") as f:
            binary_data = f.read()
            base64_encoded_data = base64.b64encode(binary_data)
            base64_string = base64_encoded_data.decode("utf-8")
            
        video_format = os.path.splitext(video_path)[-1].lower()[1:]                          
            
        # TODO: might need to standardize video_format for images
        return (video_format, base64_string)

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
            
            messages = [{"role": "user", "content": []}]
            
            # FIXME: assuming video for now
            for visual_format, visual in visuals:
                messages[0]["content"].append({"video": {"format": visual_format, "source": {"bytes": visual}}})
            messages[0]["content"].append({"text": contexts})
            
            # system_list = [    {
            #         "text": "You are a video analyst. When the user provides you with a video and instructions, follow the instructions"
            #     }
            # ]
            
            native_request = {
                "schemaVersion": "messages-v1",
                "messages": messages,
                # TODO: add system prompt
                # "system": system_list,
                "inferenceConfig": inference_config
            }

            for attempt in range(5):
                try:
                    response = self.model(body=json.dumps(native_request))
                    model_response = json.loads(response["body"].read())
                    content = model_response["output"]["message"]["content"][0]["text"]
                    
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
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        content = ""
            res.append(content)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Nova API")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Nova API does not support"

    # def get_image_audio_text_interleaved_messsage(self, image_path, audio_path, question):
    #     # image_path for list of image path
    #     # audio_path for list of audio path
    #     # question for question

    #     # fixed image token and no audio in text
    #     for index in range(1, 1 + len(image_path)):
    #         question = question.replace(f"[img{index}]", "<image>")
    #     for index in range(1, 1 + len(audio_path)):
    #         question = question.replace(f"[audio{index}]", "<audio>")

    #     text = question

    #     info_list = []
    #     image_counter = 0
    #     audio_counter = 0
    #     for part in re.split(r"(<image>|<audio>)", text):
    #         if part == "<image>":
    #             info_list.append(Image.open(image_path[image_counter]))
    #             image_counter += 1
    #         elif part == "<audio>":
    #             info_list.append({"mime_type": "audio/wav", "data": pathlib.Path(audio_path[audio_counter]).read_bytes()})
    #             audio_counter += 1
    #         else:
    #             if part == " ":
    #                 continue
    #             info_list.append(part)

    #     return info_list

    # def get_video_audio_text_interleaved_message(self, video_path, audio_path, question):
    #     # image_path for list of image path
    #     # audio_path for list of audio path
    #     # question for question

    #     # fixed video token and no audio in text
    #     for index in range(1, 1 + len(video_path)):
    #         question = question.replace(f"[video{index}]", "<video>")
    #     for index in range(1, 1 + len(audio_path)):
    #         question = question.replace(f"[audio{index}]", "<audio>")

    #     text = question

    #     info_list = []
    #     video_counter = 0
    #     audio_counter = 0
    #     for part in re.split(r"(<video>|<audio>)", text):
    #         if part == "<video>":
    #             current_video_file_name = video_path[video_counter]
    #             current_video_file = genai.upload_file(path=current_video_file_name)
    #             while current_video_file.state.name == "processing":
    #                 print("uploading file")
    #                 time.sleep(5)
    #                 current_video_file = genai.get_file(current_video_file.name)
    #             if current_video_file.state.name == "FAILED":
    #                 print("uploading file failed, next question")
    #                 return 0
    #             info_list.append(current_video_file)
    #             video_counter += 1
    #         elif part == "<audio>":
    #             info_list.append({"mime_type": "audio/wav", "data": pathlib.Path(audio_path[audio_counter]).read_bytes()})
    #             audio_counter += 1
    #         else:
    #             if part == " ":
    #                 continue
    #             info_list.append(part)

    #     return info_list
