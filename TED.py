import re, os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from evaluate.execution import evaluate_with_test_code
from evaluate.evaluation import pass_at_K
from datasets import load_dataset, load_from_disk
import numpy as np
from CDD import levenshtein_distance

def truncate(d, method_name, tokenizer):
    d = d.replace("'''", '"""')
    d = d[d.find('def '+method_name):]
    d = d[d.find('"""')+3:]
    d = d[d.find('"""\n')+4:]
    return tokenizer.encode(d, add_special_tokens=False)

def read_jsonl_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            linedata = json.loads(line)
            if linedata['task_id'] not in data.keys():
                data[linedata['task_id']] = []
            data[linedata['task_id']].append(linedata['completion'])
    return data

def load_tokenizer(model):
    from transformers import AutoTokenizer
    if model == 'CodeLlama':
        model_path = "codellama/CodeLlama-7b-hf"
    else:
        model_path = "Salesforce/codegen-6B-multi"
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def get_edit_distance_distribution_dict(path, benchmark, max_length=100, model="CodeLlama-7B"):

    gendata =read_jsonl_file(path)
    gready_path = os.path.join(f"{path.split('.jsonl')[0]}_gready.jsonl")
    gendata_gready = read_jsonl_file(gready_path)
    tokenizer = load_tokenizer(model)
    
    num = {}
    for i in range(len(benchmark)):
        task_id = benchmark[i]['task_id']
        if  task_id in gendata.keys():
            num[task_id] = []
            name = benchmark[i]['entry_point']
            greadysample  = gendata_gready[benchmark[i]['task_id']][0]
            comparison = truncate(greadysample.strip(), name, tokenizer)[:max_length]
            length = len(comparison)
            for j in gendata[task_id]:
                codesample = truncate(j.strip(), name, tokenizer)[:length]
                edit_distance = levenshtein_distance(comparison, codesample)
                num[task_id].append(edit_distance)

    return num

def evaluate_pass_at_k(Occurrence=0, path = '', duplicates = True, top_percent_exclusion = None, dataset_name = 'humaneval'):
    
    if dataset_name == 'humaneval':
        humaneval_data = load_from_disk('datasets/humaneval')['test']#load_dataset("openai_humaneval")['test']
    
    INPUT_PATH = os.path.join(path, f'Occurrence_{Occurrence}.jsonl')
    assert os.path.exists(INPUT_PATH)

    with open(INPUT_PATH, 'r') as f:
        data_dict = {}
        handled_solutions = []
        edges = {}
        task_index = {}
        for idx, task in enumerate(humaneval_data):
            data_dict[task['task_id']] = task
            task_index[task['task_id']] = 0
        if top_percent_exclusion != None:
            edit_distance_distribution_dict = get_edit_distance_distribution_dict(INPUT_PATH, humaneval_data)
        for line in f:
            line = json.loads(line)
            task_id = line["task_id"]
            line["prompt"] = ""
            line["test"] = data_dict[task_id]["test"]
            line["entry_point"] = data_dict[task_id]["entry_point"]
            task_index[task_id] += 1
            if not duplicates and line in handled_solutions:
                continue
            if top_percent_exclusion != None and edit_distance_distribution_dict[task_id][task_index[task_id]-1]<= top_percent_exclusion:
                continue
            handled_solutions.append(line)
    exec_result = evaluate_with_test_code(handled_solutions, timeout=10)
    final_result = exec_result
        
    result = pass_at_K(final_result, k=[1,5,10])
    handled_task = list(set([i["task_id"] for i in handled_solutions]))
    result['pass@1'] = result['pass@1'] * len(handled_task) / len(data_dict.keys())
    
    return result['pass@1']


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tau', type=float, default=2)
parser.add_argument('--duplicates', type=bool, default=False)
parser.add_argument('--Occurrences', type=int, default=20)
parser.add_argument('--input_path', type=str, default='./Sample_Results/CodeLlama-7B')

args = parser.parse_args()

if __name__ == '__main__':
    pass_at_k_TED = evaluate_pass_at_k(Occurrence= args.Occurrences, path=args.input_path, duplicates =args.duplicates, top_percent_exclusion = args.tau)
    print('Occurrences:', args.Occurrences)
    print('pass_at_k_TED:', pass_at_k_TED)