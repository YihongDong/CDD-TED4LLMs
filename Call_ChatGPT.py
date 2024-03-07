# pip install --upgrade openai
# pip install datasets

import openai
from datasets import load_dataset, load_from_disk
import os, json, time
import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='codeforces')
parser.add_argument('--output_path', type=str, default='./')

parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106')

parser.add_argument('--max_tokens', type=int, default=512)
parser.add_argument('--num', type=int, default=50)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top_p', type=float, default=1)
args = parser.parse_args()


def call_chatgpt(prompt, model='gpt-3.5-turbo', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, majority_at=None):
    from openai import OpenAI

    api_key = "sk-xxxxxx"
    base_url = "https://api.openai.com/v1"
    

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    message = [{
            "role": "user",
            "content": prompt
            }]
    
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 10

    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))

            response = client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=requested_completions
            )
            completions.extend([str(choice.message.content) for choice in response.choices])
        
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except:
            time.sleep(1)
    raise RuntimeError('Failed to call GPT API')

if __name__ == '__main__':
    OUTPUT_PATH =  f'{args.output_path}/result/{args.model}_{args.temperature}_{args.num}_{args.dataset}.json'
    if not os.path.exists(f'{args.output_path}/result'):
        os.makedirs(f'{args.output_path}/result')
    
    if args.dataset == 'codeforces':
        dataset = load_from_disk("./CodeForces2305")
        dataset_key = ['test']
    else:
        raise NotImplementedError
    
    handled_solutions = []
    
    with open(OUTPUT_PATH, 'a+') as f:
        for key in dataset_key:
            pbar = tqdm.tqdm(dataset[key], total=len(dataset[key]))
            for idx, task in enumerate(pbar):
                try:
                    completions = call_chatgpt(prompt=task['prompt'], model=args.model, stop=None, \
                                    temperature=args.temperature, top_p=args.top_p, \
                                    max_tokens=args.max_tokens, majority_at=args.num)
                except RuntimeError as e:
                    print(str(e))
                    print("task-%d fail"%(task['task_id']))
                    continue
                for completion in completions:
                    solution = {
                            'task_id': task['task_id'],
                            'prompt': task['prompt'],
                            'test': task['test'],
                            'entry_point': task['entry_point'],
                            'completion': completion,
                        }
                    f.write(json.dumps(solution, ensure_ascii=False) + '\n')
                    f.flush()
