import json
from datasets import load_from_disk

def levenshtein_distance(str1, str2):
    if len(str1) > len(str2):
        str1, str2 = str2, str1

    distances = range(len(str1) + 1)
    for index2, char2 in enumerate(str2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(str1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]

def strip_code(sample):
    return sample.strip().split('\n\n\n')[0] if '\n\n\n' in sample else sample.strip().split('```')[0]

def tokenize_code(sample, tokenizer, length):
    return tokenizer.encode(sample)[:length] if length else tokenizer.encode(sample)

def get_edit_distance_distribution_star(samples, gready_sample, tokenizer, length = 100):
    gready_sample = strip_code(gready_sample)
    gs = tokenize_code(gready_sample, tokenizer, length)
    num = []
    max_length = len(gs)
    for sample in samples:
        sample = strip_code(sample)
        s = tokenize_code(sample, tokenizer, length)
        num.append(levenshtein_distance(gs, s))
        max_length = max(max_length, len(s))
    return num, max_length

def calculate_ratio(numbers, alpha=1):
    count = sum(1 for num in numbers if num <= alpha)
    total = len(numbers)
    ratio = count / total if total > 0 else 0
    return ratio

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

def evaluate_classification(y_true, y_pred, y_pred_prob=None):
    metrics = {
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    
    if y_pred_prob is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_pred_prob)
    
    return metrics

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default='0.05')
parser.add_argument('--xi', type=float, default='0.01')
parser.add_argument('--input_path', type=str, default='./DETCON')
parser.add_argument('--real_world_applications', action='store_true', default=False)
parser.add_argument('--model', type=str, default='DETCON')

args = parser.parse_args()

if __name__ == '__main__':
    if 'gpt' in args.model:
        import tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-4")
    else:
        from transformers import AutoTokenizer
        codellama_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        codegen_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-6B-multi")

    if not args.real_world_applications:
        dataset = load_from_disk(args.input_path)['code generation original']
    
        Results=[]
        Labels = []
        for task in dataset:
            if task['model_name'] == "CodeLlama-7b":
                tokenizer = codellama_tokenizer
            elif task['model_name'] == "codegen-6B-multi":
                tokenizer = codegen_tokenizer
                
            dist, ml = get_edit_distance_distribution_star(task['samples'], task['greedy_sample'], tokenizer)
            peak = calculate_ratio(dist, args.alpha*ml) 
            Results.append(peak)
            Labels.append(task['label'])
        
        metric = evaluate_classification(Labels, [i>args.xi for i in Results], Results)
        print(f'Accuracy = {metric["Accuracy"]}')
        print(f'Precision = {metric["Precision"]}')
        print(f'Recall = {metric["Recall"]}')
        print(f'F1Score = {metric["F1 Score"]}')
        print(f'AUC = {metric["AUC"]}')
    else:
        tasks = json.load(open(args.input_path, 'r'))
        # enhance the detection precision
        args.alpha = 0
        args.xi = 0.2

        Peaks=[]
        for task in tasks:
            # task['samples'] temperature = 0.8 num = 50
            # task['gready_sample'] temperature = 0 num = 1
            dist, ml = get_edit_distance_distribution_star(task['samples'], task['greedy_sample'], tokenizer)
            peak = calculate_ratio(dist, args.alpha*ml) 
            Peaks.append(peak)
        Results = [i>args.xi for i in Peaks]
        
        print("Contamination Ratio:", sum(Results)/len(Results))
        print("Avg. Peak:", sum(Peaks)/len(Peaks))



