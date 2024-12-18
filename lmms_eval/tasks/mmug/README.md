# Adding examples

The data is being stored at https://huggingface.co/datasets/CUBE-CMU/MMUG. Request access to the organization to commit [here](https://huggingface.co/organizations/CUBE-CMU/share/LJpoUeShoLLxMLxvMkZDoRmoLyTCKDZPLP).

To keep organization simple for now, I simply borrowed the format from VideoMME. That is, we'll have a directory structure like

```
MMUG/
├── videomme # will be changed to mmug/
│   ├── test.jsonl
│   └── *.jsonl
├── .gitattributes
└── *.zip
```

## Data format

### Zip files

The zip files will store something like

```
*.zip
├── data/
│   └── *.mp4
└── subtitle/
    └── *.srt
```

For right now, I don't have the videos stored under a data subdirectory, but this should be changed later. We can also use Whisper to generate some pseudo-subtitles, but again the video is more important for now. We could also consider using the structure that Avik and I were using previously, namely

```
videoID/
├── audio.mp3
├── muted.mp4
└── video.mp4
```

this would require more changes to the code, but shouldn't be a super big deal.

### jsonl format

The hackiest thing I did so far was copy the exact columns of VideoMME, namely

```
video_id: (str) Unique identifier as formatted int (e.g. "001")
duration: (str) "short"
domain: (str) {"Social Situations", "Sentiments", "Egocentric Agents", "Information Querying", "Sports", "Gaming", "Shopping"}
sub_category: (str) Just copy first one for now
videoID: (str) Unique identifier to video (i.e. filename)
question_id: (str) f"{video_id}_{qnum}-{i}" for i in {1, 2, 3} (qnum is if you have multiple different queries with different answers for the same video)
task_type: (str) Just copy first one for now
question: (str) the question being asked
options: (List[str]) list of multiple choice answers ["A. ...", "B. ...", "C. ...", "D. ...", ...] or "None" for `question_id`s ending in `-2` or `-3`
answer: (str) {A, B, C, D, ...} or open-ended answer
difficulty: {"easy", "medium", "hard"}
```

We need to create more splits by creating additional .jsonl files, probably

```
test_easy.jsonl
test_hard.jsonl
test_medium.jsonl
```

We'll need to make some modifications so that we can also allow for open-ended generation, possibly with the following changes

- `question_id`s ending in `-2` or `-3` is to change `options` to `None` and `answer` to an open-ended response.

# TODOs:

- [ ] Add more examples (in progress)
- [ ] Update the VIDEO_TYPE, CATEGORIES, SUB_CATEGORIES, and TASK_CATEGORIES
- [ ] Figure out how to update `mmug_doc_to_text` and `mmug_doc_to_text_subtitle` to support open-ended generation (in progress)
- [x] Implement GPT as a Judge
- [x] Add `mmug_gpt_score` to `mmug_process_results`
- [ ] Update `mmug_multiple_choice_results` so that it is more fine-grained
- [x] Implement `mmug_gpt_score_results`