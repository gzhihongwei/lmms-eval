import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union, Dict
import PIL.Image

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
# from lmms_eval.models.model_utils.load_video import load_video_decord
import cv2
from decord import VideoReader, cpu

try:
    from deepseek_vl2.models import DeepseekVLV2Processor
except ImportError:
    eval_logger.warning("Failed to import deepseek-vl2 dependencies; Please install them via `pip install deepseek-vl2`")


def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images

def load_video_decord(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        vr_cv = cv2.VideoCapture(video_path)

    else:
        vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=1)
        vr_cv = cv2.VideoCapture(video_path[0])
    total_frame_num = int(vr_cv.get(cv2.CAP_PROP_FRAME_COUNT))
    # total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(1, total_frame_num - 10, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    # print(total_frame_num, frame_idx)
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


def split_model(model_name):
    device_map = {}
    model_splits = {
        'deepseek-ai/deepseek-vl2-small': [13, 14], # 2 GPU for 16b
        # 'deepseek-ai/deepseek-vl2': [6, 6, 6, 6], # 3 GPU for 27b
        'deepseek-ai/deepseek-vl2': [10,10,10], # 3 GPU for 27b
    }
    num_layers_per_gpu = model_splits[model_name]
    num_layers =  sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision'] = 0
    device_map['projector'] = 0
    device_map['image_newline'] = 0
    device_map['view_seperator'] = 0
    device_map['language.model.embed_tokens'] = 0
    device_map['language.model.norm'] = 0
    device_map['language.lm_head'] = 0
    device_map[f'language.model.layers.{num_layers - 1}'] = 0
    return device_map


# accelerate launch --num_processes 3 --multi_gpu --gpu_ids 4,5,6,7 --main_process_port 29501 -m lmms_eval --model deepseek_vl2 --tasks mmug --batch_size 1 --log_samples --log_samples_suffix deepseek --output_path ./logs/deepseek_vl2_full

@register_model("deepseek_vl2")
class DeepSeek_VL2(lmms):
    """
    DeepSeek-VL2 Model
    "https://github.com/deepseek-ai/DeepSeek-VL2"
    """

    def __init__(
        self,
        pretrained: str = "deepseek-ai/deepseek-vl2-tiny",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        max_pixels: int = 12845056,
        max_num_frames: int = 6,
        text_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        # if accelerator.num_processes > 1:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     # self.device_map = f"cuda:{accelerator.local_process_index}"
        #     self.device_map = split_model(pretrained)

        # elif accelerator.num_processes == 1 and device_map == "auto":
        #     self._device = torch.device(device)
        #     self.device_map = device_map
        # else:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"

        self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        if pretrained != "deepseek-ai/deepseek-vl2-tiny":
            self.device_map = split_model(pretrained)
        else:
            if accelerator.num_processes > 1:
                self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                # self.device_map = f"cuda:{accelerator.local_process_index}"
                self.device_map = split_model(pretrained)

            elif accelerator.num_processes == 1 and device_map == "auto":
                self._device = torch.device(device)
                self.device_map = device_map
            else:
                self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                self.device_map = f"cuda:{accelerator.local_process_index}"

        self._model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map
        ).eval()

        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(pretrained)
        self._tokenizer = self.processor.tokenizer
        self.max_num_frames = max_num_frames
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.text_only = text_only

        # if accelerator.num_processes > 1:
        #     assert accelerator.distributed_type in [
        #         DistributedType.FSDP,
        #         DistributedType.MULTI_GPU,
        #     ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
        #     if accelerator.distributed_type == DistributedType.FSDP:
        #         self._model = accelerator.prepare(self._model)
        #     else:
        #         self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
        #     self.accelerator = accelerator
        #     if self.accelerator.is_local_main_process:
        #         eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        #     self._rank = self.accelerator.local_process_index
        #     self._world_size = self.accelerator.num_processes
        # else:
        #     self._rank = 0
        #     self._world_size = 1

    @property
    def config(self):
        return self._model.config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Deepseek_VL2")

    def flatten(self, input):
        return [item for sublist in input for item in sublist]

    def process_visuals(self, visual):
        if isinstance(visual, str):
            if visual.endswith(('.mp4', '.avi', '.mov')):
                frames = load_video_decord(visual, self.max_num_frames)
                return [Image.fromarray(frame) for frame in frames]
            else:
                return [Image.open(visual).convert("RGB")]
        elif isinstance(visual, Image.Image):
            return [visual]
        elif isinstance(visual, (list, tuple)):
            return list(visual)
        return []

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.pop("until", [self.tokenizer.eos_token])
            if isinstance(until, str):
                until = [until]

            all_conversations = []
            processed_images_all = []
            for context, visual in zip(contexts, visuals):
                # print(visual)
                processed_images = self.process_visuals(visual)
                # print(f"len(processed_images) = {len(processed_images)}")

                # Create temporary files for in-memory images
                image_paths = []
                for idx, img in enumerate(processed_images):
                    if isinstance(img, Image.Image):
                        # Save PIL Image to temporary file
                        temp_file = f"/home/liuyuex/benchmark/lmms-eval/multi_image_{idx}.jpg"
                        # img.save(temp_file)
                        image_paths.append(temp_file)
                    elif isinstance(img, str):
                        # Already a file path
                        image_paths.append(img)
                    else:
                        raise ValueError(f"Unsupported image type: {type(img)}")

                # Build content with <image> placeholders
                image_tags = [f"<image>" 
                                    for i in range(len(image_paths))]

                # context = 'Look at this series of image frames, what is going on in this sequence?'
                formatted_content = f"{image_tags}{context}" # {image_tags}\n

                conversation = [
                    {
                        "role": "<|User|>", 
                        "content": formatted_content,
                        "images": image_paths,
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                # processed_images_all.append(processed_images)
                all_conversations.append(conversation)
                # del processed_images
            # print("All conversations processed")  
            # Process EACH conversation individually
            inputs_list = []
            for conv in all_conversations:
                try:
                    # import pdb; pdb.set_trace()
                    # pil_images = load_pil_images(conv)
                    if self.text_only == True:
                        pil_images = []
                    else:
                        pil_images = processed_images
                    inputs = self.processor(
                        conversations=conv,  # Single conversation
                        images=pil_images,
                        force_batchify=True,
                        system_prompt="").to(self.model.device)

                    inputs_list.append(inputs)
                except Exception as e:
                    eval_logger.error(f"Processing error: {e}")

            # print("model generating")
            # Generate responses
                   # Batch processing
            # print(inputs_list)
            # import pdb; pdb.set_trace()
            # batch_inputs = {
            #     "input_ids": torch.cat([x.input_ids for x in inputs_list]),
            #     "attention_mask": torch.cat([x.attention_mask for x in inputs_list]),
            #     "images": torch.cat([x.images for x in inputs_list]),
            #     "images_seq_mask": torch.cat([x.images_seq_mask for x in inputs_list]),
            #     "images_spatial_crop": torch.cat([x.images_spatial_crop for x in inputs_list]),
            # }
            
            inputs_embeds = self.model.prepare_inputs_embeds(**inputs)
            past_key_values = None
            outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=inputs.input_ids,
                    images=inputs.images,
                    images_seq_mask=inputs.images_seq_mask,
                    images_spatial_crop=inputs.images_spatial_crop,
                    attention_mask=inputs.attention_mask,
                    past_key_values=past_key_values,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    use_cache=True,)

            # outputs = self.model.generate(
            #         inputs_embeds=inputs_embeds,
            #         input_ids=batch_inputs['input_ids'],
            #         images=batch_inputs['images'],
            #         images_seq_mask=batch_inputs['images_seq_mask'],
            #         images_spatial_crop=batch_inputs['images_spatial_crop'],
            #         attention_mask=batch_inputs['attention_mask'],
            #         past_key_values=past_key_values,
            #         pad_token_id=self.tokenizer.eos_token_id,
            #         bos_token_id=self.tokenizer.bos_token_id,
            #         eos_token_id=self.tokenizer.eos_token_id,
            #         max_new_tokens=512,
            #         do_sample=True,
            #         temperature=0.4,
            #         top_p=0.9,
            #         repetition_penalty=1.1,
            #         use_cache=True,)

            answers =self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # answers =self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True)
            res.append(answers)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

