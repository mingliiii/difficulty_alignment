import os
import glob

import re
import json
import numpy as np

from tqdm import tqdm

from scipy.stats import spearmanr

CLASSIFICATION_TASKS = {'SAT_math', 'SAT_reading'}

TASK_CONDITIONS = {
    'USMLE': 'USMLE',
    'Cambridge': 'Cambridge',
    'SAT_reading': 'SAT_reading',
    'SAT_math': 'SAT_math'
}

CONDITION_LABELS = {
    'USMLE': 'USMLE',
    'Cambridge': 'Cambridge',
    'SAT_reading': 'SAT Reading',
    'SAT_math': 'SAT Math'
}

MODEL_GROUPS = [
    # Group 1: OpenAI
    [
        ('gpt35', 'GPT-3.5-Turbo'),
        ('gpt4o', 'GPT-4o'),
        ('gpt4omini', 'GPT-4o-mini'),
        ('gpt41', 'GPT-4.1'),
        ('gpt41mini', 'GPT-4.1-mini'),
        ('o4mini', 'GPT-o4-mini'),
        ('gpt5', 'GPT-5'),
    ],
    # Group 2: Standard Open Source
    [
        ('Llama2_7B', 'Llama2-7B'),
        ('Llama2_13B', 'Llama2-13B'),
        ('Phi3_4k', 'Phi3'),
        ('Phi35_4k', 'Phi3.5'),
        ('Llama3_1_8B', 'Llama3.1-8B'),
        ('Qwen25_7B', 'Qwen2.5-7B'),
        ('Qwen25_32B', 'Qwen2.5-32B'),
        ('Phi4', 'Phi4'),
        ('Qwen3_8B_NR', 'Qwen3-8B'),
        ('Qwen3_32B_NR', 'Qwen3-32B'),
    ],
    # Group 3: Reasoning
    [
        ('deepseekR1', 'DeepSeek-R1'), 
        ('QwQ32B', 'QWQ-32B'),
        ('R1_Distill_Qwen_32B', 'R1-Qwen32B'),
        ('Qwen3_32B_R', 'Qwen3-32B (R)'), 
    ]
]

def extract_predicted_difficulty_classification(model_response):
    if not model_response: return None
    matches = list(re.finditer(r'\b(easy|medium|hard)\b', model_response, re.IGNORECASE))
    return matches[-1].group(1).lower() if matches else None

def convert_difficulty_to_numeric(difficulty_str):
    if not isinstance(difficulty_str, str): return None
    d = difficulty_str.lower().strip()
    return 0 if d == 'easy' else 1 if d == 'medium' else 2 if d == 'hard' else None

def extract_predicted_difficulty(model_response):
    if not model_response: return None
    match = re.search(r'\\boxed\s*\{\s*(\d+(?:\.\d+)?)\s*\}', model_response)
    if match: return float(match.group(1))
    matches = list(re.finditer(r'\*\*\s*(\d+(?:\.\d+)?)\s*\*\*', model_response))
    if matches: return float(matches[-1].group(1))
    patterns = [
        r'Final\s+(?:Difficulty\s+)?Value[:\s]+(\d+(?:\.\d+)?)',
        r'Final\s+Difficulty[:\s]+(\d+(?:\.\d+)?)',
        r'difficulty\s+(?:value|level|is|of)\s+(?:is\s+)?(?:around\s+)?(\d+(?:\.\d+)?)',
        r'score[:\s]+(\d+(?:\.\d+)?)'
    ]
    for p in patterns:
        m = re.search(p, model_response, re.IGNORECASE)
        if m: 
            val = float(m.group(1))
            if 0 <= val <= 100: return val
    nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', model_response)
    if nums:
        for n_str in reversed(nums):
            try:
                val = float(n_str)
                if 0 <= val <= 100: return val
            except: continue
    return None


def calculate_correlation_from_jsonl(jsonl_file, task=None):
    if not os.path.exists(jsonl_file): return None
    is_cls = task in CLASSIFICATION_TASKS if task else False
    true_diffs, pred_diffs = [], []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    td = data.get('difficulty')
                    if td is None: td = data.get('Difficulty')
                    if td is None: continue
                    resp = data.get('model_response', '')
                    if is_cls:
                        pd_str = extract_predicted_difficulty_classification(resp)
                        if pd_str is None: continue
                        pd = convert_difficulty_to_numeric(pd_str)
                        td_num = convert_difficulty_to_numeric(str(td))
                        if td_num is not None:
                            true_diffs.append(td_num)
                            pred_diffs.append(pd)
                    else:
                        try: td_float = float(td)
                        except: continue
                        pd = extract_predicted_difficulty(resp)
                        if pd is not None:
                            true_diffs.append(td_float)
                            pred_diffs.append(pd)
                except: continue
        if len(true_diffs) < 2: return None
        ta, pa = np.array(true_diffs), np.array(pred_diffs)
        if np.std(ta) == 0 or np.std(pa) == 0: return None
        corr, p = spearmanr(ta, pa)
        if np.isnan(corr): return None
        return {'spearman_correlation': corr, 'p_value': p}
    except: return None

def load_all_task_results(role_condition='diff'):
    all_results = {}
    result_dir = f'model_results/{role_condition}'
    if not os.path.exists(result_dir): 
        print(f"Warning: Directory does not exist: {result_dir}")
        return all_results
    
    for task in TASK_CONDITIONS.keys():
        pattern = os.path.join(result_dir, f"{task}_*_results.jsonl")
        files = glob.glob(pattern)
        for fpath in tqdm(files, desc=f"Loading {task}", leave=False):
            base = os.path.basename(fpath).replace('_results.jsonl', '')
            if not base.startswith(f"{task}_"): continue
            raw_model_name = base[len(task)+1:]
            if 'gemma' in raw_model_name.lower(): continue
            res = calculate_correlation_from_jsonl(fpath, task)
            if res:
                if raw_model_name not in all_results: all_results[raw_model_name] = {}
                all_results[raw_model_name][task] = res
    return all_results