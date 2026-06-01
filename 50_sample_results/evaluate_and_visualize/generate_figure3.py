#!/usr/bin/env python3
"""
Generate Figure 3-style visualization from FUR (Faithfulness through Unlearning
Reasoning Steps) experimental results.

Reproduces the style of Figure 3 from the paper:
  "Measuring Chain of Thought Faithfulness by UnLearning Reasoning Steps"

Usage:
  python generate_figure3.py                          # default instance
  python generate_figure3.py --instance 9-29 --step 6 # specific instance
  python generate_figure3.py --list                   # list available instances
  python generate_figure3.py --list-flips             # list instances with prediction flips
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ─── Paths ───────────────────────────────────────────────────────
BASE_DIR = "measuring Cot/reproduce of measuring Cot1/parametric-faithfulness"
COT_FILE = os.path.join(
    BASE_DIR,
    'final_cot/openbook/Llama-3.2-3B-Instruct_s=1001_t=0.0_cots.jsonl'
)
RESULTS_FILE = os.path.join(
    BASE_DIR,
    'final_results/openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_pos=True_ff2=True.out'
)

# ─── Visual constants ─────────────────────────────────────────────
BAR_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
LETTERS = ['A', 'B', 'C', 'D', 'E']

TARGET_BG = '#FFE0E0'
TARGET_BORDER = '#CC5555'
NORMAL_BG = '#FFFFFF'
NORMAL_BORDER = '#DDDDDD'
POST_COT_BG = '#FFE8E8'
POST_COT_BORDER = '#DDAAAA'
ITER1_COT_BG = '#E8F0FE'
ITER1_COT_BORDER = '#AACCDD'
QUESTION_BG = '#F0F0F0'
QUESTION_BORDER = '#CCCCCC'
OPTIONS_BG = '#FFF9E6'
OPTIONS_BORDER = '#E6D580'


# ─── Data loading utilities ───────────────────────────────────────
def load_jsonl(filepath):
    """Load a JSONL file, skipping malformed lines."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def extract_instance_data(target_instance_id, target_step_idx):
    """Extract COT + unlearning results for a specific instance & step."""
    cots = {}
    for item in load_jsonl(COT_FILE):
        cots[item['id']] = item

    all_results = {}
    for item in load_jsonl(RESULTS_FILE):
        rid = item['id']
        if rid not in all_results:
            all_results[rid] = {}
        all_results[rid][item['step_idx']] = item

    cot_data = cots.get(target_instance_id)
    if cot_data is None:
        raise ValueError(f"Instance '{target_instance_id}' not found in COT data")

    step_result = all_results.get(target_instance_id, {}).get(target_step_idx)
    if step_result is None:
        available = sorted(all_results.get(target_instance_id, {}).keys())
        raise ValueError(
            f"Step {target_step_idx} not found for instance '{target_instance_id}'. "
            f"Available steps: {available}"
        )
    return cot_data, step_result


def build_probability_iterations(step_result):
    """Return list of probability vectors: [Base, Iter1, Iter2, ..., IterN]."""
    initial_probs = step_result['initial_probs']
    unlearning_results = step_result.get('unlearning_results', {})
    iterations = [initial_probs]
    for ik in sorted(unlearning_results.keys(), key=int):
        v = unlearning_results[ik]
        it_probs = v.get('probs', [])
        if it_probs:
            iterations.append(it_probs)
    return iterations


def clean_text(text, max_len=110):
    """Remove EOT tokens, strip whitespace, optionally truncate."""
    text = text.replace('<|eot_id|>', '').strip()
    if len(text) > max_len:
        text = text[:max_len - 3] + '...'
    return text


# ─── Helper: list available instances ──────────────────────────────
def list_instances(only_flips=False):
    """Print all available instances with step counts and flip info."""
    cots = {}
    for item in load_jsonl(COT_FILE):
        cots[item['id']] = item

    results_by_id = {}
    for item in load_jsonl(RESULTS_FILE):
        rid = item['id']
        if rid not in results_by_id:
            results_by_id[rid] = {}
        results_by_id[rid][item['step_idx']] = item

    for qid in sorted(cots.keys(), key=lambda x: int(x.split('-')[0])):
        cot_data = cots[qid]
        if qid not in results_by_id:
            continue

        num_steps = len(cot_data['segmented_cot'])
        steps_data = results_by_id[qid]
        flipped_steps = []

        for step_idx in sorted(steps_data.keys()):
            r = steps_data[step_idx]
            initial_pred = r.get('prediction', -1)
            ur = r.get('unlearning_results', {})
            for ik in sorted(ur.keys(), key=int):
                v = ur[ik]
                probs = v.get('probs', [])
                if probs:
                    curr_pred = np.argmax(probs)
                    if curr_pred != initial_pred and int(ik) >= 1:
                        flipped_steps.append(step_idx)
                        break

        if only_flips and not flipped_steps:
            continue

        initial_probs = list(steps_data.values())[0].get('initial_probs', [])
        initial_pred = np.argmax(initial_probs) if initial_probs else -1
        pred_letter = LETTERS[initial_pred] if initial_pred >= 0 else '?'
        correct = cot_data.get('correct_letter', '?')

        print(f"[{qid}] {cot_data['question'][:70]}")
        print(f"       Steps: {num_steps} | Pred: {pred_letter} | Correct: {correct} | "
              f"Flip steps: {flipped_steps if flipped_steps else 'none'}")
        print()


# ─── Core visualization ────────────────────────────────────────────
def draw_probability_bars(fig, gs_parent, prob_data, num_options):
    """Draw the top panel: 6 bar charts showing probability evolution."""
    num_bars = min(6, len(prob_data))
    option_labels = LETTERS[:num_options]

    bar_labels = ['Base model']
    for i in range(1, num_bars):
        bar_labels.append(f'Iter {i}')

    gs_bars = gridspec.GridSpecFromSubplotSpec(
        1, num_bars, subplot_spec=gs_parent, wspace=0.35
    )

    for i in range(num_bars):
        ax = fig.add_subplot(gs_bars[0, i])
        probs = list(prob_data[i])
        if len(probs) < num_options:
            probs += [0.0] * (num_options - len(probs))
        probs = probs[:num_options]

        x_pos = np.arange(num_options)
        bars = ax.bar(x_pos, probs, width=0.5,
                      color=BAR_COLORS[:num_options], alpha=0.85, edgecolor='white')

        max_idx = np.argmax(probs)
        bars[max_idx].set_edgecolor('black')
        bars[max_idx].set_linewidth(2.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(option_labels, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_title(bar_labels[i], fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, p in zip(bars, probs):
            if p > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f'{p:.2f}', ha='center', va='bottom', fontsize=9,
                        fontweight='bold')


def draw_cot_text_panel(ax, cot_data, step_result, y_start=97):
    """Draw the bottom panel: question, options, base CoT, post-unlearning CoT."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    y = y_start

    question = cot_data['question']
    options = cot_data['options']
    segmented_cot = cot_data['segmented_cot']
    step_idx = step_result['step_idx']
    unlearning_results = step_result.get('unlearning_results', {})
    initial_probs = step_result['initial_probs']
    num_options = len(options)
    option_labels = LETTERS[:num_options]

    initial_pred_idx = np.argmax(initial_probs)

    # ── Helper to add a text box that consumes vertical space ──
    def add_box(x, y, text, fontsize=10, fontweight='normal', color='#333333',
                bg=NORMAL_BG, border=NORMAL_BORDER, pad=0.3, indent=0):
        ax.text(x + indent, y, text, fontsize=fontsize, fontweight=fontweight,
                color=color, va='top', ha='left',
                bbox=dict(boxstyle=f'round,pad={pad}', facecolor=bg, edgecolor=border,
                          linewidth=1.0))
        return y - max(2, fontsize * 0.35 * (1 + text.count('\n')))

    def add_multiline(x, y, lines, fontsize=9, bg=NORMAL_BG, border=NORMAL_BORDER,
                      indent=2, max_lines=15):
        for i, line in enumerate(lines):
            if i >= max_lines:
                ax.text(x + indent, y, '  ... (truncated)', fontsize=fontsize,
                        color='#888888', va='top', family='monospace')
                y -= 2.5
                break
            line_clean = clean_text(line, max_len=115)
            if not line_clean:
                continue
            ax.text(x + indent, y, f'  {line_clean}', fontsize=fontsize,
                    va='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=bg,
                              edgecolor=border, linewidth=0.5))
            y -= 2.8
        return y

    # Question
    y = add_box(1, y, f'Question: {question}', fontsize=11, fontweight='bold',
                bg=QUESTION_BG, border=QUESTION_BORDER, pad=0.4)

    # Options
    opts_text = '  |  '.join(options)
    y = add_box(1, y, f'Options: {opts_text}', fontsize=10,
                bg=OPTIONS_BG, border=OPTIONS_BORDER, pad=0.35)

    # Model prediction
    initial_pred_letter = option_labels[initial_pred_idx]
    initial_pred_opt = options[initial_pred_idx] if initial_pred_idx < len(options) else '?'
    y = add_box(1, y, f'Base no-CoT prediction: {initial_pred_letter}) {initial_pred_opt}  '
                f'(prob={initial_probs[initial_pred_idx]:.3f})',
                fontsize=10, color='#666666')

    # ── Base CoT ──
    y -= 1
    y = add_box(1, y, 'Base model CoT:', fontsize=10, fontweight='bold')
    for si, step_text in enumerate(segmented_cot):
        step_display = clean_text(step_text, max_len=110)
        is_target = (si == step_idx)
        bg = TARGET_BG if is_target else NORMAL_BG
        border = TARGET_BORDER if is_target else NORMAL_BORDER
        prefix = '[Target] ' if is_target else ''
        marker = f'  {si + 1}. {prefix}{step_display}'
        y = add_box(1, y, marker, fontsize=9, bg=bg, border=border, indent=2)
        if y < 20:
            break

    # ── Post-unlearning CoT: Iter 1 ──
    if '0' in unlearning_results:
        iter1_data = unlearning_results['0']
        iter1_cot = iter1_data.get('new_cot', '')
        iter1_probs = iter1_data.get('probs', [])
        if iter1_probs:
            pred1 = np.argmax(iter1_probs)
            changed = (pred1 != initial_pred_idx)
        else:
            changed = False

        if iter1_cot and y > 15:
            y -= 2
            flag = ' [PREDICTION CHANGED]' if changed else ''
            y = add_box(1, y, f'Post-unlearning CoT (Iter 1){flag}:',
                        fontsize=10, fontweight='bold', color='#2980B9')
            cot_lines = iter1_cot.split('\n')
            y = add_multiline(1, y, cot_lines, fontsize=9,
                              bg=ITER1_COT_BG, border=ITER1_COT_BORDER)

    # ── Post-unlearning CoT: last iteration ──
    last_iter_key = sorted(unlearning_results.keys(), key=int)[-1]
    last_data = unlearning_results.get(last_iter_key, {})
    last_cot = last_data.get('new_cot', '')
    last_probs = last_data.get('probs', [])
    if last_probs:
        last_pred_idx = np.argmax(last_probs)
        flipped = (last_pred_idx != initial_pred_idx)
    else:
        last_pred_idx = initial_pred_idx
        flipped = False

    if last_cot and y > 15 and last_iter_key != '0':
        y -= 2
        flip_text = ' [FLIPPED]' if flipped else ''
        y = add_box(1, y, f'Post-unlearning CoT (Iter {last_iter_key}){flip_text}:',
                    fontsize=10, fontweight='bold', color='#C0392B')
        cot_lines = last_cot.split('\n')
        y = add_multiline(1, y, cot_lines, fontsize=9,
                          bg=POST_COT_BG, border=POST_COT_BORDER)

    # ── Summary ──
    if y > 3:
        y -= 2
        summary = (f'Summary: Initial = {option_labels[initial_pred_idx]} '
                   f'→ Final (Iter {last_iter_key}) = {option_labels[last_pred_idx]}')
        if flipped:
            summary += '  ◆ Prediction FLIPPED'
        y = add_box(1, y, summary, fontsize=10, fontweight='bold',
                    color='#8E44AD', pad=0.3)


def create_figure3(cot_data, step_result, output_path='figure3_output.pdf'):
    """Main function: create the full Figure 3-style visualization."""
    question = cot_data['question']
    options = cot_data['options']
    step_idx = step_result['step_idx']
    num_options = len(options)

    prob_data = build_probability_iterations(step_result)

    # ── Build figure ──
    fig = plt.figure(figsize=(16, 13))
    title = (f'FUR Unlearning Visualization — LLaMA-3-3B · OpenBookQA\n'
             f'Instance: {cot_data["id"]} | Unlearned Step: {step_idx} | '
             f'Question: {question}')
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.99)

    # Grid layout: top = bar charts (30%), bottom = text (70%)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 2.2],
                           top=0.94, bottom=0.02, left=0.04, right=0.98, hspace=0.28)

    # Top: probability bar charts
    draw_probability_bars(fig, gs[0, 0], prob_data, num_options)

    # Bottom: CoT text
    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.axis('off')
    draw_cot_text_panel(ax_text, cot_data, step_result)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'Figure saved to: {output_path}')
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Figure 3-style FUR visualization'
    )
    parser.add_argument('--instance', '-i', type=str, default='7-391',
                        help='Instance ID to visualize (default: 7-391)')
    parser.add_argument('--step', '-s', type=int, default=0,
                        help='CoT step index to unlearn (default: 0)')
    parser.add_argument('--list', action='store_true',
                        help='List all available instances')
    parser.add_argument('--list-flips', action='store_true',
                        help='List instances where unlearning causes a prediction flip')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path (default: auto-generated)')

    args = parser.parse_args()

    if args.list:
        list_instances(only_flips=False)
        sys.exit(0)

    if args.list_flips:
        list_instances(only_flips=True)
        sys.exit(0)

    # Build output filename
    if args.output:
        output_file = args.output
    else:
        output_file = f'figure3_{args.instance.replace("/", "_")}_step{args.step}.pdf'

    print(f'{"="*60}')
    print(f'Instance: {args.instance}  |  Step: {args.step}')
    print(f'Model: LLaMA-3-3B  |  Dataset: OpenBookQA')
    print(f'{"="*60}\n')

    cot_data, step_result = extract_instance_data(args.instance, args.step)

    print(f'Question: {cot_data["question"]}')
    print(f'Options: {cot_data["options"]}')
    print(f'Correct answer: {cot_data["correct_letter"]}')
    print(f'\nSegmented CoT ({len(cot_data["segmented_cot"])} steps):')
    for i, step_text in enumerate(cot_data['segmented_cot']):
        marker = ' <-- TARGET (UNLEARNED)' if i == args.step else ''
        print(f'  [{i}] {clean_text(step_text, max_len=85)}{marker}')

    initial_probs = step_result['initial_probs']
    initial_pred = np.argmax(initial_probs)
    print(f'\nInitial no-CoT probs: {[f"{p:.4f}" for p in initial_probs]}')
    print(f'Initial prediction: {LETTERS[initial_pred]}')

    ur = step_result.get('unlearning_results', {})
    print(f'\nUnlearning iterations: {len(ur)}')
    for ik in sorted(ur.keys(), key=int):
        v = ur[ik]
        probs = v.get('probs', [])
        if not probs:
            continue
        pred = np.argmax(probs)
        flip = ' *** FLIP ***' if pred != initial_pred else ''
        print(f'  Iter {ik}: pred={LETTERS[pred]}, probs={[f"{p:.4f}" for p in probs]}{flip}')
        new_cot = v.get('new_cot', '')
        if new_cot:
            print(f'           CoT preview: {new_cot[:120]}...')
            print()

    print(f'\nGenerating visualization → {output_file}')
    create_figure3(cot_data, step_result, output_path=output_file)
    print('Done!')