#!/usr/bin/env python3
import json
import os
import argparse
import glob
import re
import numpy as np
import pandas as pd

from utils import load_all_task_results, TASK_CONDITIONS, CONDITION_LABELS, MODEL_GROUPS

def analyze_by_model(all_results):
    model_stats = {}
    
    for model_name, task_results in all_results.items():
        model_stats[model_name] = {}
        values = []
        
        for task in TASK_CONDITIONS.keys():
            if task in task_results:
                corr = task_results[task]['spearman_correlation']
                model_stats[model_name][task] = corr
                values.append(corr)
            else:
                model_stats[model_name][task] = None
        
        if values:
            model_stats[model_name]['Average'] = np.mean(values)
        else:
            model_stats[model_name]['Average'] = None
    
    return model_stats

def print_results(model_stats, role_condition):
    print("\n" + "="*80)
    print(f"Analysis Results for Role Condition: {role_condition}")
    print("="*80)
    
    print("\n1. Correlation by Model (each model across all tasks):")
    print("-"*80)
    
    rows = []
    for group in MODEL_GROUPS:
        for raw_name, display_name in group:
            if raw_name in model_stats:
                row = {'Model': display_name}
                for task in TASK_CONDITIONS.keys():
                    val = model_stats[raw_name].get(task)
                    row[CONDITION_LABELS[task]] = f"{val:.4f}" if val is not None else "N/A"
                avg = model_stats[raw_name].get('Average')
                row['Average'] = f"{avg:.4f}" if avg is not None else "N/A"
                rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Analyze correlation results by task")
    parser.add_argument("--role_condition", "-r", type=str, default="diff", 
                        choices=["diff", "diff_role_weak", "diff_role_medium", "diff_role_strong"],
                        help="Role condition to use (default: diff)")
    parser.add_argument("--output_file", "-o", type=str, default=None)
    args = parser.parse_args()
    
    results = load_all_task_results(args.role_condition)
    
    if not results:
        print("Error: No results found.")
        return
    
    print("Analyzing results...")
    
    model_stats = analyze_by_model(results)
    
    print_results(model_stats, args.role_condition)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
