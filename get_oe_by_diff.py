import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--responses', type=str, default='./logs/logs_grok2-vision/sub_only/submissions/gpt_eval_result_maverix_sub_only_2025-07-23-14-51-15.json')
parser.add_argument('--questions', type=str, default='./lmms_eval/tasks/maverix/processing/test_rematch.jsonl')
args = parser.parse_args()

with open(args.responses) as f:
    responses = json.load(f)

qid2diff = {}
with open(args.questions) as f:
    for line in f:
        d = json.loads(line)
        qid = d['question_id'].split('-')[0]
        qid2diff[qid] = d['task_type'].lower()

diff_scores = defaultdict(list)
for r in responses:
    diff = qid2diff.get(r['question_id'], 'Unknown')
    diff_scores[diff].append(r['score'])

all_scores = []
for diff, scores in diff_scores.items():
    avg = sum(scores) / len(scores)
    print(f'{diff}: {avg:.2f}')
    all_scores.extend(scores)

print(f'Overall: {sum(all_scores)/len(all_scores):.2f}')