#!/usr/bin/env python3
import json
import os
import argparse
import glob
from collections import defaultdict
import csv

ROLE_DIRS = {
    'direct': 'model_results/direct',
    'direct_role_weak': 'model_results/direct_role_weak_result',
    'direct_role_medium': 'model_results/direct_role_medium_result',
    'direct_role_strong': 'model_results/direct_role_strong_result'
}

ROLE_LABELS = {
    'direct': 'Baseline',
    'direct_role_weak': 'Weak',
    'direct_role_medium': 'Medium',
    'direct_role_strong': 'Strong'
}

ALL_TASKS = ['USMLE', 'Cambridge', 'SAT_reading', 'SAT_math']

ROLE_ORDER = ['direct', 'direct_role_weak', 'direct_role_medium', 'direct_role_strong']

MODEL_GROUPS = [
    [('gpt35', 'GPT-3.5-Turbo'), ('gpt4o', 'GPT-4o'), ('gpt4omini', 'GPT-4o-mini'), ('gpt41', 'GPT-4.1'), ('gpt41mini', 'GPT-4.1-mini'), ('o4mini', 'GPT-o4-mini'), ('gpt5', 'GPT-5')],
    [('Llama2_7B', 'Llama2-7B'), ('Llama2_13B', 'Llama2-13B'), ('Phi3_4k', 'Phi3'), ('Phi35_4k', 'Phi3.5'), ('Llama3_1_8B', 'Llama3.1-8B'), ('Qwen25_7B', 'Qwen2.5-7B'), ('Qwen25_32B', 'Qwen2.5-32B'), ('Phi4', 'Phi4'), ('Qwen3_8B_NR', 'Qwen3-8B'), ('Qwen3_32B_NR', 'Qwen3-32B')],
    [('deepseekR1', 'DeepSeek-R1'), ('QwQ32B', 'QWQ-32B'), ('R1_Distill_Qwen_32B', 'R1-Qwen32B'), ('Qwen3_32B_R', 'Qwen3-32B (R)')]
]

MODEL_NAME_MAP = {}
for group in MODEL_GROUPS:
    for raw, display in group: MODEL_NAME_MAP[raw] = display

def calculate_accuracy(jsonl_file):
    stats = {'total': 0, 'correct': 0, 'accuracy': 0.0}
    if not os.path.exists(jsonl_file): return stats
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                stats['total'] += 1
                
                if 'Correct' in data:
                    val = data['Correct']
                    is_correct = False
                    if isinstance(val, bool):
                        is_correct = val
                    elif isinstance(val, str):
                        is_correct = (val.lower() == 'true')
                    elif isinstance(val, int):
                        is_correct = (val == 1)
                    
                    if is_correct:
                        stats['correct'] += 1
            except: continue
            
    if stats['total'] > 0:
        stats['accuracy'] = stats['correct'] / stats['total']
    return stats

def analyze_role_task(role_dir, task):
    pattern = os.path.join(role_dir, f"{task}_*_results.jsonl")
    jsonl_files = glob.glob(pattern)
    model_stats = {}
    
    for jsonl_file in jsonl_files:
        basename = os.path.basename(jsonl_file)
        if not basename.startswith(f"{task}_"): continue
        model_name = basename[len(task)+1 : -14]
        
        if 'Gemma' in model_name: continue
        
        model_stats[model_name] = calculate_accuracy(jsonl_file)
    
    return model_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', '-o', type=str, default="capability_result.csv")
    args = parser.parse_args()
    
    print("Calculating Accuracy across all Tasks and Roles...")
    
    data = defaultdict(dict)
    all_models = set()
    
    ordered_cols = []
    
    for task in ALL_TASKS:
        for role in ROLE_ORDER:
            if role not in ROLE_DIRS: continue
            
            col_name = f"{task}_{ROLE_LABELS[role]}"
            ordered_cols.append(col_name)
            
            role_dir = ROLE_DIRS[role]
            if not os.path.exists(role_dir): continue
            
            results = analyze_role_task(role_dir, task)
            
            for model, stats in results.items():
                all_models.add(model)
                data[model][col_name] = stats['accuracy']

    model_sort_list = []
    for group in MODEL_GROUPS:
        for raw, display in group:
            if raw in all_models: model_sort_list.append(raw)
    for m in all_models:
        if m not in model_sort_list: model_sort_list.append(m)
        
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    
    with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
        headers = ['Model'] + ordered_cols + ['Average_Acc']
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for m in model_sort_list:
            row = {'Model': MODEL_NAME_MAP.get(m, m)}
            acc_list = []
            for col in ordered_cols:
                if col in data[m]:
                    val = data[m][col]
                    row[col] = f"{val:.4f}"
                    acc_list.append(val)
                else:
                    row[col] = ""
            
            if acc_list:
                row['Average_Acc'] = f"{sum(acc_list)/len(acc_list):.4f}"
            else:
                row['Average_Acc'] = "0.0000"
                
            writer.writerow(row)
            
    print(f"Done! Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
