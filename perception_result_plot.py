#!/usr/bin/env python3
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image

from utils import load_all_task_results, TASK_CONDITIONS

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['figure.titleweight'] = 'bold'

matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['xtick.major.width'] = 1.2
matplotlib.rcParams['ytick.major.width'] = 1.2

COLORS = {
    'USMLE':         '#1f77b4',
    'Cambridge':     '#ff7f0e',
    'SAT_reading':   '#2ca02c',
    'SAT_math':      '#d62728'
}

CONDITION_LABELS = {
    'USMLE': 'USMLE',
    'Cambridge': 'Cambridge',
    'SAT_reading': 'SAT Reading',
    'SAT_math': 'SAT Math'
}

TASK_X_OFFSET_MAP = {
    'USMLE': -0.5,
    'Cambridge': 0.5,
    'SAT_reading': -0.5,
    'SAT_math': 0.5
}

MODEL_GROUPS = [
    [
        ('gpt35', 'GPT-3.5-Turbo'),
        ('gpt4o', 'GPT-4o'),
        ('gpt4omini', 'GPT-4o-mini'),
        ('gpt41', 'GPT-4.1'),
        ('gpt41mini', 'GPT-4.1-mini'),
        ('o4mini', 'GPT-o4-mini'),
        ('gpt5', 'GPT-5'),
    ],
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
    [
        ('deepseekR1', 'DeepSeek-R1'), 
        ('QwQ32B', 'QWQ-32B'),
        ('R1_Distill_Qwen_32B', 'R1-Qwen32B'),
        ('Qwen3_32B_R', 'Qwen3-32B (R)'), 
    ]
]

DOWNWARD_TASKS = {'SAT_math', 'SAT_reading'}

def plot_grouped_bar_chart(all_results, role_condition, output_file, figsize):
    models_to_plot = []
    for g_idx, group in enumerate(MODEL_GROUPS):
        for raw_name, display_name in group:
            if raw_name in all_results:
                models_to_plot.append({'raw': raw_name, 'display': display_name, 'group': g_idx})

    if not models_to_plot:
        print("Error: No matching model data found.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    
    tasks = list(TASK_CONDITIONS.keys())
    n_tasks = len(tasks)
    
    bar_width = 0.1
    standard_step = 0.4
    extra_group_gap = 0.2
    
    x_centers = []
    group_boundaries = []
    current_x = 0
    
    for i, item in enumerate(models_to_plot):
        if i == 0:
            current_x = 0
        else:
            prev_group = models_to_plot[i-1]['group']
            curr_group = item['group']
            
            if curr_group != prev_group:
                step = standard_step + extra_group_gap
                current_x += step
                boundary_pos = current_x - (step / 2)
                group_boundaries.append(boundary_pos)
            else:
                current_x += standard_step
        x_centers.append(current_x)
    
    for t_idx, task in enumerate(tasks):
        vals = []
        xs = []
        for i, item in enumerate(models_to_plot):
            raw = item['raw']
            if task in all_results[raw]:
                vals.append(all_results[raw][task]['spearman_correlation'])
            else:
                vals.append(np.nan)
            
            offset = TASK_X_OFFSET_MAP[task] * bar_width
            xs.append(x_centers[i] + offset)
            
        xs = np.array(xs)
        vals = np.array(vals)
        mask = ~np.isnan(vals)
        
        is_downward = task in DOWNWARD_TASKS
        if is_downward:
            plot_vals = -vals[mask]
            bottom = 0
        else:
            plot_vals = vals[mask]
            bottom = None
        
        ax.bar(xs[mask], plot_vals, width=bar_width, bottom=bottom,
               label=CONDITION_LABELS[task], 
               color=COLORS[task], 
               alpha=0.85,
               edgecolor='black', 
               linewidth=0.8, 
               zorder=3)

        for x, v, pv in zip(xs[mask], vals[mask], plot_vals):
            if is_downward:
                ax.text(x, pv - 0.02, f'{v:.2f}', 
                        ha='center', va='top', 
                        fontsize=7, fontweight='bold', rotation=90)
            else:
                ax.text(x, pv + 0.02, f'{v:.2f}', 
                        ha='center', va='bottom', 
                        fontsize=7, fontweight='bold', rotation=90)

    for boundary in group_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1.0, zorder=1)

    ax.set_xticks(x_centers)
    display_names = [m['display'] for m in models_to_plot]
    ax.set_xticklabels(display_names, rotation=40, ha='right', fontsize=11, fontweight='bold')
    
    total_group_width = bar_width * n_tasks
    left_limit = x_centers[0] - (total_group_width / 2)
    right_limit = x_centers[-1] + (total_group_width / 2)
    ax.set_xlim(left_limit, right_limit)

    ax.set_ylabel('Spearman Correlation', fontsize=14, fontweight='bold', labelpad=10)
    
    upward_max = 0
    downward_max = 0
    
    for m in all_results.values():
        for task, d in m.items():
            val = d['spearman_correlation']
            if task in DOWNWARD_TASKS:
                downward_max = max(downward_max, val)
            else:
                upward_max = max(upward_max, val)
    
    if upward_max > 0 or downward_max > 0:
        max_val = max(upward_max, downward_max)
        ax.set_ylim([-max_val - 0.15, max_val + 0.15])
        
        ticks = ax.get_yticks()
        labels = [f'{abs(t):.2f}' if t < 0 else f'{t:.2f}' for t in ticks]
        ax.set_yticklabels(labels)

    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=10, frameon=False)
    for text in leg.get_texts():
        text.set_fontweight('bold')

    ax.grid(axis='y', linestyle=':', color='gray', alpha=0.4, linewidth=1.0, zorder=0)
    
    plt.tight_layout()
    if output_file:
        temp_file = output_file + '.temp.png'
        plt.savefig(temp_file, dpi=300, bbox_inches='tight')
        
        img = Image.open(temp_file)
        img_array = np.array(img)
        
        if len(img_array.shape) == 3:
            non_white = np.any(img_array < 250, axis=2)
        else:
            non_white = img_array < 250
        
        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)
        
        if np.any(rows) and np.any(cols):
            top = np.argmax(rows)
            bottom = len(rows) - np.argmax(rows[::-1])
            left = np.argmax(cols)
            right = len(cols) - np.argmax(cols[::-1])
            
            padding = 5
            width, height = img.size
            top = max(0, top - padding)
            bottom = min(height, bottom + padding)
            left = max(0, left - padding)
            right = min(width, right + padding)
            
            img_cropped = img.crop((left, top, right, bottom))
            img_cropped.save(output_file, dpi=(300, 300))
            os.remove(temp_file)
            print(f"Saved to {output_file}")
        else:
            os.rename(temp_file, output_file)
            print(f"Saved to {output_file}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role_condition", "-r", type=str, default="diff", 
                        choices=["diff", "diff_role_weak", "diff_role_medium", "diff_role_strong"],
                        help="Role condition to use (default: diff)")
    parser.add_argument("--output_file", "-o", type=str, default=None)
    parser.add_argument("--figsize", type=str, default="12,6") 
    args = parser.parse_args()
    
    try: figsize = tuple(map(float, args.figsize.split(',')))
    except: figsize = (12, 6)
    
    if not args.output_file:
        args.output_file = f"perception_result_plot{args.role_condition}.png"
    
    print(f"Loading results for role condition: {args.role_condition}...")
    results = load_all_task_results(args.role_condition)
    
    print("Generating bold plot with default colors...")
    plot_grouped_bar_chart(results, args.role_condition, args.output_file, figsize)
    print("Done.")

if __name__ == "__main__":
    main()
