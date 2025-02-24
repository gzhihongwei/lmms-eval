import pandas as pd
import os
from glob import glob
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import subprocess

VID_ROOT = "/home/liuyuex/.cache/huggingface/mmug/vid_full"

df = pd.read_csv("data.csv")  # Top row is not header


approved_df = df[~df["source_link_if_any"].isna() & (df["review status"] == "Approved")]
approved_df["id"] = approved_df["id"].apply(lambda x: f"{x:04}")
approved_df["domain"] = approved_df["category"]
approved_df["sub_category"] = approved_df["sub_category"]
approved_df["videoID"] = (
    approved_df["source_link_if_any"]
    .str.replace("https://youtu.be/", "https://www.youtube.com/watch?v=")
    .str.split("=")
    .apply(lambda x: x[1] if len(x) > 1 else x[0])
    .str.replace("&t", "")
    .str.replace("?si", "")
    .str.replace("&ab_channel", "")
    .str.replace("https://visualize.ego4d-data.org/", "")
    .str.replace(".mp4", "")
    .tolist()
)

# problematic_video = ["5VeZfX-AVXA", "8TOcGnJE9P4"]


def get_length(filename):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)


cache = defaultdict(int)

reformatted = pd.DataFrame(
    columns=[
        "video_id",
        "duration",
        "domain",
        "sub_category",
        "audio_category",
        "videoID",
        "question_id",
        "task_type",
        "question",
        "options",
        "answer",
        "annotator_assigned_difficulty",
    ]
)
num_rows = 0
cache = defaultdict(int)

for _, row in tqdm(approved_df.iterrows(), total=len(approved_df)):
    # In case we have multiple segments from the same video
    row["videoID"] = f"{row['videoID']}" # _{cache[row['videoID']]}
    cache[row["videoID"]] += 1

    try:
        duration = get_length(f"{VID_ROOT}/{row['videoID']}.mp4")
    except:
        continue
    common = [
        row["id"],
        duration,
        row["domain"].split(", "),
        row["sub_category"],
        row["audio_category"],
        row["videoID"],
        None,
        # "placeholder",
        row["q1_sub_task"],
        row["annotator_assigned_difficulty"],
        None,
        None,
    ]

    common2 = [
        row["id"],
        duration,
        row["domain"].split(", "),
        row["sub_category"],
        row["audio_category"],
        row["videoID"],
        None,
        # "placeholder",
        row["q2_sub_task"],
        row["annotator_assigned_difficulty"],
        None,
        None,
    ]

    # Want complete annotations only
    if pd.isna(row["open_q1_2"]):
        continue

    # First multiple choice
    mcq1 = common.copy()
    mcq1[6] = f"{row['id']}_1-1"
    mcq1[8] = row["mcq_q1_prompt"]
    options = []
    for i, raw_option in enumerate(row.loc["q1_opt_a":"q1_opt_h"]):
        try:
            option = raw_option.rstrip(".") + "."
        except AttributeError:
            option = "Unsure."

        letter = chr(i + ord("A"))
        option = f"{letter}. {option}"
        options.append(option)

    mcq1[9] = options
    mcq1[10] = row["q1_correct_option"].upper()
    reformatted.loc[num_rows] = mcq1
    num_rows += 1
    # First open question
    open1_1 = common.copy()
    open1_1[6] = f"{row['id']}_1-2"
    open1_1[8] = row["open_q1_1"]

    try:
        open1_1[10] = row["open_a1_1"].rstrip(".") + "."
    except AttributeError:
        open1_1[10] = row[f"q1_opt_{row['q1_correct_option']}"].rstrip(".") + "."

    open1_1[10] = row["open_a1_1"].rstrip(".") + "."
    reformatted.loc[num_rows] = open1_1
    num_rows += 1

    # First rephrased open question
    open1_2 = common.copy()
    open1_2[6] = f"{row['id']}_1-3"
    open1_2[8] = row["open_q1_2"]

    try:
        open1_2[10] = row["open_a1_1"].rstrip(".") + "."
    except AttributeError:
        open1_2[10] = row[f"q1_opt_{row['q1_correct_option']}"].rstrip(".") + "."

    reformatted.loc[num_rows] = open1_2
    num_rows += 1

    ## Check for second mcq
    if pd.isna(row["mcq_q2_prompt"]):
        continue

    # Want complete annotations only
    if pd.isna(row["open_q2_2"]):
        continue

    # Second multiple choice
    mcq2 = common2.copy()
    mcq2[6] = f"{row['id']}_2-1"
    mcq2[8] = row["mcq_q2_prompt"]
    options = []
    for i, raw_option in enumerate(row.loc["q2_opt_a":"q2_opt_h"]):
        try:
            option = raw_option.rstrip(".") + "."
        except AttributeError:
            option = "Unsure."

        letter = chr(i + ord("A"))
        option = f"{letter}. {raw_option}"
        options.append(option)

    mcq2[9] = options
    try:
        mcq2[10] = row["q2_correct_option"].upper()
    except AttributeError:  # Not completed
        continue
    reformatted.loc[num_rows] = mcq2
    num_rows += 1
    # import pdb; pdb.set_trace()
    # Second open question
    open2_1 = common2.copy()
    open2_1[6] = f"{row['id']}_2-2"
    open2_1[8] = row["open_q1_1"]

    try:
        open2_1[10] = row["open_a2_1"].rstrip(".") + "."
    except AttributeError:
        open2_1[10] = row[f"q2_opt_{row['q2_correct_option']}"].rstrip(".") + "."

    reformatted.loc[num_rows] = open2_1
    num_rows += 1

    # Second rephrased open question
    open2_2 = common.copy()
    open2_2[6] = f"{row['id']}_2-3"
    open2_2[8] = row["open_q2_2"]

    try:
        open2_2[10] = row["open_a2_1"].rstrip(".") + "."
    except AttributeError:
        open2_2[10] = row[f"q2_opt_{row['q2_correct_option']}"].rstrip(".") + "."

    reformatted.loc[num_rows] = open2_2
    num_rows += 1
reformatted.to_json("test_full.jsonl", orient="records", lines=True)
