import re
import json
import copy
import argparse
import sys
import os
from post_process import post_process_code, build_test_method_for_apps
from execution import evaluate_with_test_code
from evaluation import pass_at_K
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='codeforces')
parser.add_argument('--lang', type=str, default='python')
parser.add_argument('--input_path', type=str, default='path_to_input_file.json')
parser.add_argument('--output_path', type=str, default='path_to_output_file.json')
args = parser.parse_args()

INPUT_PATH = args.input_path
OUTPUT_PATH = args.output_path

if args.dataset == 'codeforces':
    dataset = load_from_disk("datasets/codeforces")
    dataset_key = ["test"]

with open(INPUT_PATH, 'r') as f:

    data_dict = {}
    handled_solutions = []
    for key in dataset_key:
        for idx, task in enumerate(dataset[key]):
            data_dict[task['task_id']] = task

    for line in f:
        line = json.loads(line)
        if args.dataset == 'codeforces':
            line["test"] =  build_test_method_for_apps(line["test"], test_case_limit = 5)
            line["entry_point"] = 'solution'
        line["completion"] = post_process_code(prompt=line['prompt'], code=line['completion'], func_name=line['entry_point'], m_indent='    ')
        
        line["prompt"] = ""
        handled_solutions.append(line)

exec_result = evaluate_with_test_code(handled_solutions, timeout=1)

print('pass rates of solutions')
print(len(exec_result))
print(pass_at_K(exec_result, k=[1,5,10]))
# # --------------------------------------------------