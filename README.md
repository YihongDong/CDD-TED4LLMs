# Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models
[**Paper**](https://arxiv.org/abs/2402.15938) (Accepted to ACL), [**Big Code Models Leaderboard**](https://huggingface.co/spaces/wzxii/Memorization-or-Generation-of-Big-Code-Models-Leaderboard)

## Evaluate CDD
```bash
python CDD.py
```

## Evaluate TED
```bash
python TED.py
```

## Load Dataset
```Python
# pip install datasets
from datasets import load_from_disk
dataset = load_from_disk('./CodeForces2305')
```

## Evaluate CodeForces2305
```bash
# generate
python Call_ChatGPT.py --output_path path_to_generation_file.json 
# evaluate
python evaluate.evaluate_codeforces.py --input_path path_to_generation_file.json
```

## Dataset Information
**CodeForces2305** comprises 90 of the easiest level programming problems collected from the CodeForces website since May 2023, as well as 10 problems of the same type from March 2023 to April 2023 for optional few-shot prompting. We report the zero-shot results of ChatGPTs on CodeForces2305 dataset in the paper.

## Contaminated Models
Given that many researchers are interested in the models that we finetuned to simulate data contamination, we have preliminarily organized the model weights and outputs, totaling about 300GB. Considering the large amount of data, we have selected about 26GB of data to package it so that it can cover the settings in the paper. We have divided the data into two parts, each stored in Google Drive, the links are as follows: 

https://drive.google.com/file/d/1FTakU4HIXz00rg8GQQjM7jRw2hBWMsHu/view

https://drive.google.com/file/d/1uBsnVoVCqT8VXA-4REClR6FS8AOab2X9/view

```bash
cat datacontamination_share_26G.tar.gz.* > datacontamination_share_26G_combined.tar.gz
tar -xzf datacontamination_share_26G_combined.tar.gz
```
 
## Citation
```
@article{dong2024generalization,
  title={Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models},
  author={Dong, Yihong and Jiang, Xue and Liu, Huanyu and Jin, Zhi and Li, Ge},
  journal={arXiv preprint arXiv:2402.15938},
  year={2024}
}
```
