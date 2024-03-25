# Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models
[**Paper**](https://arxiv.org/abs/2402.15938)

*The code and datasets are organized for public release in two weeks...*

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
 
## Citation
```
@article{dong2024generalization,
  title={Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models},
  author={Dong, Yihong and Jiang, Xue and Liu, Huanyu and Jin, Zhi and Li, Ge},
  journal={arXiv preprint arXiv:2402.15938},
  year={2024}
}
```
