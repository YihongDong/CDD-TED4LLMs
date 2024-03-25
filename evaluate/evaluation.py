import numpy as np
from collections import defaultdict
import logging
from typing import List, Union
import itertools

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def pass_at_K_by_task(results, k):
    result_dict = defaultdict(list)
    for line in results:
        result_dict[line['task_id']].append(line['passed'])
    result = dict()
    for task_id in result_dict.keys():
        total = len(result_dict[task_id])
        correct = sum(result_dict[task_id])
        score = _estimate_pass_at_k(total, [correct], k)[0]
        result[task_id] = score
    return result

def pass_at_K(results, k = [1, 10, 100]):
    def _turn_list_into_dict(result_lines):
        result_dict = defaultdict(list)
        for line in result_lines:
            result_dict[line['task_id']].append(line['passed'])
        return result_dict

    # Calculate pass@k.
    total, correct = [], []
    for passed in _turn_list_into_dict(results).values():
        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": round(_estimate_pass_at_k(total, correct, k).mean(), 4)
                 for k in ks if (total >= k).all()}
    # logger.info(pass_at_k)
    return pass_at_k

def _estimator(n: int, c: int, k: int) -> float:
    """
    Calculates comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 0
    return np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def _estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([1.0 - _estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def AvgPassRatio(handled_solutions):
    total = len(handled_solutions)
    correct = sum([1 for s in handled_solutions if s['passed']])
    return correct/total
    

