import ast
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import requests
import yaml
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.tasks._task_utils import file_utils

# TODO: fix this as short?
VIDEO_TYPE = ["short"]
# TODO: update this
CATEGORIES = ["Social Situations", "Sentiments", "Egocentric Agents", "Information Querying", "Sports", "Gaming", "Shopping"]

# TODO: update this
SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

# TODO: update this
TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]

# Copied, pruned, and modified from VideoMME

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))
    
cache_name = config["dataset_kwargs"]["cache_dir"]

NUM_SECONDS_TO_SLEEP = 5

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def mmug_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, 
                            #   "data", # TODO: need to eventually re-add this back
                              video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path: \"{video_path}\" does not exist, please check")
        
    return [video_path]


def mmug_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    post_prompts = (lmms_eval_specific_kwargs or {}).get("post_prompt", None).split("$")
    question = doc["question"]
    
    if doc["question_id"].endswith("-1"):
        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option."
        options = "\n".join(doc["options"])
        question = question + "\n" + options
        post_prompt = post_prompts[0] or " The best answer is:"
        full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
        return full_prompt
    
    # TODO: fine tune these pre and post prompts
    pre_prompt = (lmms_eval_specific_kwargs or {}).get("pre_prompt", "")
    post_prompt = post_prompts[1] or " The answer is:"
    return f"{pre_prompt}{question}{post_prompt}"


# Frames + Subs
# This video's subtitles are listed below:
# 【subtitles】

# Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:
# Frames / Frames + Audio
# Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.
# 【question】
# The best answer is:


def mmug_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
    # TODO: update so more similar to above
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below:\n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        if "gemini_api_flag" in lmms_eval_specific_kwargs:  # specific for gemini_api
            if lmms_eval_specific_kwargs["gemini_api_flag"] == "full subtitle":
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
        else:
            if "frame_num" in lmms_eval_specific_kwargs:
                frame_num = lmms_eval_specific_kwargs["frame_num"]
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                if frame_num == -1:
                    frame_num = total_frame
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option."
    question = doc["question"]
    options = "\n".join(doc["options"])
    question = question + "\n" + options
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt


def get_eval_generic(question, answer, pred, task, max_tokens: int, retries: int = 5):
    global headers

    if task == "correctness":
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:\n"
                "-----\n"
                "##INSTRUCTIONS:\n"
                "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                "- The predicted answer must be factually accurate and align with the video content.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the factual accuracy of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING. "
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'score': 4}.",
            },
        ]
    elif task == "detailed_orientation":
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:\n"
                "------\n"
                "##INSTRUCTIONS:\n"
                "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING. "
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'score': 4}.",
            },
        ]
    elif task == "context":
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:\n"
                "------\n"
                "##INSTRUCTIONS:\n"
                "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                "- The predicted answer must capture the main themes and sentiments of the video.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Provide your evaluation of the contextual understanding of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'score': 4}.",
            },
        ]
    elif task == "temporal":
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:\n"
                "------\n"
                "##INSTRUCTIONS:\n"
                "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
                "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
                "- Evaluate the temporal accuracy of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of temporal consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'score': 4}.",
            },
        ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            try:
                response_data = response.json()  # Attempt to parse JSON
            except requests.exceptions.JSONDecodeError:
                eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
                continue  # Skip to next retry
            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
        # Handle HTTP errors separately
        except requests.exceptions.HTTPError as e:
            eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
        # Handle other requests-related errors
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

        # Handle other unexpected errors
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
            return "", ""

    return "", ""


def get_eval_consistency(question1, question2, answer, pred1, pred2, max_tokens: int, retries: int = 5):
    global headers

    messages = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
            "You will be given two very similar questions, a common answer common to both the questions and predicted answers for the two questions. "
            "Your task is to compare the predicted answers for two very similar questions, with a common correct answer and determine if they are consistent. Here's how you can accomplish the task:\n"
            "------\n"
            "##INSTRUCTIONS:\n"
            "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or significant differences in the conveyed information.\n"
            "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
            "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
            "- Evaluate the consistency of the two predicted answers compared to the correct answer.",
        },
        {
            "role": "user",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question 1: {question1}\n"
            f"Question 2: {question2}\n"
            f"Correct Answer: {answer}\n"
            f"Predicted Answer to Question 1: {pred1}\n"
            f"Predicted Answer to Question 2: {pred2}\n\n"
            "Provide your evaluation only as a consistency score where the consistency score is an integer value between 0 and 5, with 5 indicating the highest level of consistency. "
            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the consistency score as an INTEGER, not a STRING. "
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {'score': 4}.",
        },
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            try:
                response_data = response.json()  # Attempt to parse JSON
            except requests.exceptions.JSONDecodeError:
                eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
                continue  # Skip to next retry
            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
        # Handle HTTP errors separately
        except requests.exceptions.HTTPError as e:
            eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
        # Handle other requests-related errors
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

        # Handle other unexpected errors
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
            return "", ""

    return "", ""


def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review_dict = ast.literal_eval(review)
        score = review_dict.get("score", 0)
        return int(score)
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return 0
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
        return 0
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return 0


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


matrices = []

for i in VIDEO_TYPE:
    for j in CATEGORIES:
        for k in SUB_CATEGORIES:
            for l in TASK_CATEGORIES:
                matrices.append(f"{i}_{j}_{k}_{l}")


def mmug_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mmug score), value: metric value
    """
    pred = results[0]
    question = doc["question"]
    answer = doc["answer"]
    # TODO: see if we need to do additional filtering here for open-ended questions
    
    correctness_dict = None
    detailed_orientation_dict = None
    context_dict = None
    
    if doc["question_id"].endswith("-1"):
        pred = extract_characters_regex(pred)
       
    elif doc["question_id"].endswith("-2"):
        review_correctness, model_name = get_eval_generic(question, answer, pred, "correctness", 64)
        score_correctness = parse_score(review_correctness)
        
        correctness_dict = {"question_id": doc["question_id"], "Q": doc["question"], "A": doc["answer"], "pred": pred, "score": score_correctness}
        
        review_detailed_orientation, model_name = get_eval_generic(question, answer, pred, "detailed_orientation", 64)
        score_detailed_orientation = parse_score(review_detailed_orientation)
        
        detailed_orientation_dict = {"question_id": doc["question_id"], "Q": doc["question"], "A": doc["answer"], "pred": pred, "score": score_detailed_orientation}
        
        review_context, model_name = get_eval_generic(question, answer, pred, "context", 64)
        score_context = parse_score(review_context)
        
        context_dict = {"question_id": doc["question_id"], "Q": doc["question"], "A": doc["answer"], "pred": pred, "score": score_context}
        
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    category = doc["domain"]
    sub_category = doc["sub_category"]
    task_category = doc["task_type"]
    data_dict = {"question_id": doc["question_id"], "duration": doc["duration"], "category": category, "sub_category": sub_category, "task_category": task_category, "question": question, "pred_answer": pred, "answer": answer}
    
    perception_dict = None
    consistency_dict = None
    
    if doc["question_id"].endswith("-1"):
        perception_dict = data_dict
    else:
        consistency_dict = data_dict

    # return {f"mmug_perception_score": data_dict for metric in matrices}
    return {"mmug_perception_score": perception_dict, "mmug_gpt_eval_score_correctness": correctness_dict, "mmug_gpt_eval_score_detailed_orientation": detailed_orientation_dict, "mmug_gpt_eval_score_context": context_dict, "mmug_gpt_eval_score_consistency": consistency_dict} 


def mmug_multiple_choice_results(results, args):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    # TODO: add a place to dump answers and results
    # TODO: modify so that it is more fine-grained
    result_dict = dict(correct=0, answered=0)
    
    for result in results:
        if result is None:
            continue 
        
        result_dict["answered"] += 1
        result_dict["correct"] += result["pred_answer"] == result["answer"]
        
    total_correct = result_dict["correct"]
    total_answered = result_dict["answered"]

    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0
    
    ## Reference from VideoMME
    # category2score = {}

    # for video_type in VIDEO_TYPE:
    #     for category in CATEGORIES:
    #         for sub_category in SUB_CATEGORIES:
    #             for task_category in TASK_CATEGORIES:
    #                 key = f"{video_type}_{category}_{sub_category}_{task_category}"
    #                 category2score[key] = {"correct": 0, "answered": 0}

    # for result in results:
    #     video_type = result["duration"]
    #     category = result["category"]
    #     sub_category = result["sub_category"]
    #     task_category = result["task_category"]
    #     key = f"{video_type}_{category}_{sub_category}_{task_category}"
    #     category2score[key]["answered"] += 1
    #     category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    # for video_type in VIDEO_TYPE:
    #     total_correct = 0
    #     total_answered = 0
    #     for k, v in category2score.items():
    #         if video_type in k:
    #             total_correct += v["correct"]
    #             total_answered += v["answered"]
    #     eval_logger.info(f"Evaluation on video Type: {video_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    # for category in CATEGORIES:
    #     total_correct = 0
    #     total_answered = 0
    #     for k, v in category2score.items():
    #         if category in k:
    #             total_correct += v["correct"]
    #             total_answered += v["answered"]
    #     eval_logger.info(f"Evaluation on Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    # for sub_cate in SUB_CATEGORIES:
    #     total_correct = 0
    #     total_answered = 0
    #     for k, v in category2score.items():
    #         if sub_cate in k:
    #             total_correct += v["correct"]
    #             total_answered += v["answered"]
    #     eval_logger.info(f"Evaluation on Video Sub Categories: {sub_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    # for task_cate in TASK_CATEGORIES:
    #     total_correct = 0
    #     total_answered = 0
    #     for k, v in category2score.items():
    #         if task_cate in k:
    #             total_correct += v["correct"]
    #             total_answered += v["answered"]
    #     eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    # total_correct = 0
    # total_answered = 0
    # for k, v in category2score.items():
    #     total_correct += v["correct"]
    #     total_answered += v["answered"]
    # eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    # return 100 * total_correct / total_answered if total_answered > 0 else 0
    
    

    
    
def mmug_gpt_eval(result_file_path, args):
    """
    Process the result file containing predictions, score them using GPT,
    and save the results with added scores and correctness fields to a new file.

    Args:
        result_file_path: path to the JSON file with results to be evaluated
    """
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    eval_file_name = f"gpt_eval_result_{get_mmug_task(args)}_{now_date_time}.json"
    eval_file_path = file_utils.generate_submission_file(eval_file_name, args)

    # Load the predictions from the result file
    with open(result_file_path, "r") as file:
        result_list = json.load(file)

    evaluated_results = []

    # Load the predictions from the result file
    with open(result_file_path, "r") as file:
        result_list = json.load(file)

    # Process each result to generate scores
    # If task is consistency (2 questions with 2 answers)
    for data_dict in tqdm(result_list, desc="GPT-Eval-for-Consistency"):
        try:
            question1 = data_dict.get("Q1", "")
            question2 = data_dict.get("Q2", "")
            answer = data_dict.get("A", "")
            pred1 = data_dict.get("pred1", "")
            pred2 = data_dict.get("pred2", "")

            # Assume get_eval returns a review and the model name, and parse_score parses this review
            review, model_name = get_eval_consistency(question1, question2, answer, pred1, pred2, 64)
            score = parse_score(review)
        except Exception as e:
            eval_logger.error(f"Error for Question ID: {data_dict.get('question_id', 'Unknown')}: {e}")
            review = "Failed to Get a Proper Review."
            model_name = "Failed Request"
            score = 0

        # Update the dictionary with the new entries
        updated_dict = {
            "question_id": data_dict["question_id"],
            "model_name": model_name,
            "score": score,
            "Q1": question1,
            "Q2": question2,
            "A": answer,
            "pred1": pred1,
            "pred2": pred2,
        }
        evaluated_results.append(updated_dict)
    
    # Save the evaluated results to a new JSON file
    with open(eval_file_path, "w") as f:
        json.dump(evaluated_results, f, indent=4)

    return eval_file_path


def get_mmug_task(args):
    if isinstance(args.tasks, str):
        return args.tasks
    
    for task in args.tasks:
        if task.startswith("mmug"):
            return task
    

def mmug_aggregate_submissions_consistency(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_{get_mmug_task(args)}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    combined_results = []
    processed_indices = set()

    # Iterate through results to find pairs in order to avoid multiprocessing bugs
    for i in range(len(results)):
        if i in processed_indices:
            continue

        first_dict = results[i]
        
        if first_dict is None:
            processed_indices.add(i)
            continue
        
        question_id = first_dict.get("question_id").split("-")[0]

        for j in range(i + 1, len(results)):
            if j in processed_indices:
                continue

            second_dict = results[j]
            
            if second_dict is None:
                processed_indices.add(j)
                continue
            
            if question_id == second_dict.get("question_id").split("-")[0]:
                combined_dict = dict(question_id=question_id, Q1=first_dict["question"], pred1=first_dict["pred_answer"], Q2=second_dict["question"], pred2=second_dict["pred_answer"], A=first_dict["answer"])
                processed_indices.add(i)
                processed_indices.add(j)
                combined_results.append(combined_dict)
                break

    with open(path, "w") as f:
        json.dump(combined_results, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")

    return path


def mmug_print_scores(eval_file_path, args):
    # Load the predictions from the result file
    with open(eval_file_path, "r") as file:
        evaluated_list = json.load(file)

    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    score_file_name = f"scores_{get_mmug_task(args)}_{now_date_time}.json"
    path = file_utils.generate_submission_file(score_file_name, args)

    # Compute average score
    total_score = 0

    # Iterate over the results to sum scores
    for result_dict in evaluated_list:
        total_score += result_dict["score"]

    # Calculate accuracy and average score
    average_score = total_score / len(evaluated_list) if evaluated_list else 0

    # Write the processed data to the scores file
    with open(path, "w") as f:
        json.dump({"average_score": average_score}, f, indent=4)

    eval_logger.info(f"Score file saved to {path}")

    return average_score


def mmug_gpt_score_results(results, args):
    result_file_path = mmug_aggregate_submissions_consistency(results, args)
    eval_file_path = mmug_gpt_eval(result_file_path, args)
    average_score = mmug_print_scores(eval_file_path, args)
    return average_score


def mmug_aggregate_score(results, args):
    total_score = 0
    num_scores = 0

    # Iterate over the results to sum scores
    for result_dict in results:
        if result_dict is None:
            continue
        
        total_score += result_dict["score"]
        num_scores += 1

    average_score = total_score / num_scores if num_scores > 0 else 0
    eval_logger.info(f"Average Score: {average_score}")
    return average_score
