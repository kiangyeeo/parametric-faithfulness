#!/usr/bin/env python3
"""
Generate Figure 4-style single-instance heatmap from FUR experimental results.

Reproduces the heatmap visualization from Figure 4 of the paper:
  "Measuring Chain of Thought Faithfulness by UnLearning Reasoning Steps"

Figure 4 shows a heatmap produced by unlearning individual reasoning steps,
with FF-SOFT (Δp) values indicating how much probability mass shifted from
the model's initial prediction when each step was unlearned:
  - Positive Δp (red): unlearning removes probability from the initial answer
  - Negative Δp (green): unlearning adds probability to the initial answer

Usage:
  python generate_heatmap.py
  python generate_heatmap.py --dataset openbook --model LLaMA-3-3B --instance 9-368
  python generate_heatmap.py -d sqa -m LLaMA-3-3B -i 7-391 -o my_heatmap.pdf
"""

import os
import sys
import json
import argparse
import textwrap

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ─── Constants ────────────────────────────────────────────────────
LETTERS = ['A', 'B', 'C', 'D', 'E']

RESULTS_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'parametric-faithfulness', 'final_results'
)

DATASET_CONFIG = {
    'openbook': {
        'LLaMA-3-3B': 'openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_pos=True_ff2=True.out',
        'Phi-3': 'openbook/Phi-3/npo_KL_sentencize_s=True_lr=0.0001_rs=1001_pos=True_ff2=True.out',
    },
    'sqa': {
        'LLaMA-3-3B': 'sqa/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_pos=True_ff2=True.out',
        'Phi-3': 'sqa/Phi-3/npo_KL_sentencize_s=True_lr=5e-05_rs=1001_pos=True_ff2=True.out',
    },
}

DEFAULT_MODEL = 'LLaMA-3-3B'
DEFAULT_DATASET = 'openbook'
DEFAULT_INSTANCE = '9-368'
STRIP_PREFIXES = ['Step 1: ', 'Step 2: ', 'Step 3: ', 'Step 4: ', 'Step 5: ',
                   'Step 6: ', 'Step 7: ', 'Step 8: ', 'Step 9: ', 'Step 10: ',
                   'Step 11: ', 'Step 12: ', 'Step 13: ', 'Step 14: ', 'Step 15: ',
                   'Step 16: ']

# Colormap: green → white → red, matching Figure 4 of the paper
GREEN_WHITE_RED = LinearSegmentedColormap.from_list(
    'green_white_red',
    [(0.2, 0.7, 0.3), (1.0, 1.0, 1.0), (0.9, 0.3, 0.3)]
)


# ─── Data loading ─────────────────────────────────────────────────
def load_jsonl(filepath):
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


def load_instance_results(dataset, model, instance_id):
    """Load all unlearning step results for a single instance."""
    config_key = f'{dataset}/{model}'
    relative_path = DATASET_CONFIG[dataset][model]
    filepath = os.path.join(RESULTS_BASE, relative_path)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Results file not found: {filepath}')

    results = load_jsonl(filepath)

    instance_steps = [r for r in results if r['id'] == instance_id]
    if not instance_steps:
        raise ValueError(
            f'Instance "{instance_id}" not found in {filepath}. '
            f'Available IDs: {sorted(set(r["id"] for r in results))[:10]}...'
        )

    instance_steps.sort(key=lambda x: x['step_idx'])
    return instance_steps


def renorm(probs):
    """Renormalize probability vector to sum to 1."""
    total = sum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


def compute_ff_soft(step_result):
    """
    Compute FF-SOFT for a single unlearned step.

    FF-SOFT = mean change in probability of the initial prediction
    across all unlearning iterations.

    Positive: probability removed from initial answer (step was helpful)
    Negative: probability added to initial answer (step was counterproductive)
    """
    unlearning_results = step_result['unlearning_results']
    sorted_keys = sorted(unlearning_results.keys(), key=int)
    probs = [renorm(unlearning_results[k]['probs']) for k in sorted_keys]

    initial_pred = np.argmax(probs[0])
    initial_mass = probs[0][initial_pred]

    dmasses = [initial_mass - m[initial_pred] for m in probs[1:]]
    return float(np.mean(dmasses))


def extract_instance_profile(instance_steps):
    """Build complete profile for a single instance."""
    first = instance_steps[0]
    profile = {
        'id': first['id'],
        'question': first['question'],
        'options': first['options'],
        'correct_letter': first['correct'],
        'prediction': first['prediction'],
        'segmented_cot': first['segmented_cot'],
    }

    ff_soft_values = []
    for step_result in instance_steps:
        ff_value = compute_ff_soft(step_result)
        ff_soft_values.append(ff_value)

    profile['ff_soft_values'] = ff_soft_values
    profile['num_steps'] = len(instance_steps)
    profile['initial_probs'] = renorm(first['initial_probs'])

    return profile


# ─── Visualization ────────────────────────────────────────────────
def clean_step_text(text, max_len=None):
    """Clean CoT step text: strip prefixes and EOT tokens."""
    text = text.replace('<|eot_id|>', '').strip()
    for prefix in STRIP_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.strip()
    if max_len and len(text) > max_len:
        text = text[:max_len - 3] + '...'
    return text


def wrap_text(text, width=80):
    """Wrap long text for display in the figure."""
    return '\n'.join(textwrap.wrap(text, width=width))


def create_heatmap(profile, output_path='figure4_heatmap.pdf', line_width=80):
    """
    Create Figure 4-style heatmap visualization for a single instance.

    Layout (top to bottom):
      1. Question + answer options
      2. Initial model prediction (with probability)
      3. Reasoning steps highlighted by FF-SOFT values
      4. Colorbar legend
    """
    question = profile['question']
    options = profile['options']
    segmented_cot = profile['segmented_cot']
    ff_values = profile['ff_soft_values']
    initial_probs = profile['initial_probs']
    prediction = profile['prediction']
    correct_letter = profile['correct_letter']
    num_steps = profile['num_steps']
    num_options = len(options)
    option_labels = LETTERS[:num_options]
    pred_letter = option_labels[prediction]
    pred_prob = initial_probs[prediction]

    ff_min = min(ff_values)
    ff_max = max(ff_values)
    ff_abs_max = max(abs(ff_min), abs(ff_max), 0.01)

    norm = mcolors.TwoSlopeNorm(vmin=-ff_abs_max, vcenter=0, vmax=ff_abs_max)

    # Compute layout dimensions
    char_width = 1.0 / line_width * 8.5
    max_text_chars = max(
        max(len(clean_step_text(s)) for s in segmented_cot),
        len(question),
        max(len(opt) for opt in options) if options else 0
    )
    fig_width = max(12, char_width * min(max_text_chars, 120))
    step_height_per_line = 1.0
    total_step_height = sum(
        max(1, len(wrap_text(clean_step_text(s, max_len=line_width),
                             width=line_width).split('\n')))
        for s in segmented_cot
    ) * step_height_per_line
    fig_height = max(7, total_step_height * 1.2 + 4)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)
    ax = fig.add_axes([0.05, 0.02, 0.90, 0.94])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    y = 98
    x_left = 2
    x_center = 50

    # ── Title ──
    ax.text(x_center, y,
            'FF-SOFT Heatmap — Single Instance Analysis',
            fontsize=13, fontweight='bold', ha='center', va='top',
            color='#2C3E50')
    y -= 5.5

    info_line = (f'Dataset: OpenBookQA  |  Model: LLaMA-3.2-3B-Instruct  |  '
                 f'Instance: {profile["id"]}')
    ax.text(x_center, y, info_line, fontsize=9, ha='center', va='top',
            color='#7F8C8D')
    y -= 5.5

    # ── Question box ──
    wrapped_q = wrap_text(question, width=line_width)
    q_lines = wrapped_q.count('\n') + 1
    q_height = max(2, q_lines * 1.8)

    ax.add_patch(FancyBboxPatch(
        (x_left - 1, y - q_height * 1.6), 96, q_height * 1.8,
        boxstyle='round,pad=0.3',
        facecolor='#EBF5FB', edgecolor='#AED6F1', linewidth=1.2
    ))
    ax.text(x_left, y - 0.5,
            f'Question: {wrapped_q}',
            fontsize=11, fontweight='bold', va='top', ha='left',
            color='#2C3E50')
    y -= q_height * 2.4

    # ── Options ──
    opt_lines_per = [wrap_text(opt, width=line_width // 2).count('\n') + 1
                     for opt in options]
    max_opt_lines = max(opt_lines_per)

    line_ys = []
    for i, opt in enumerate(options):
        cleaned = clean_step_text(opt, max_len=120)
        opt_lines = wrap_text(cleaned, width=line_width // 2)
        display = f'{option_labels[i]}) {opt_lines}'
        opt_lines_count = display.count('\n') + 1

        is_predicted = (i == prediction)
        is_correct = (option_labels[i] == correct_letter)
        bg = '#FFFFFF'
        border = '#CCCCCC'
        label = ''

        if is_predicted and is_correct:
            bg = '#D5F5E3'
            border = '#82E0AA'
            label = ' [Predicted ✓]'
        elif is_predicted:
            bg = '#FADBD8'
            border = '#F1948A'
            label = ' [Predicted]'
        elif option_labels[i] == correct_letter:
            bg = '#D5F5E3'
            border = '#82E0AA'
            label = ' [Correct]'

        display_wrapped = display.split('\n')
        for li, dl in enumerate(display_wrapped):
            ax.text(x_left + 1, y - li * 1.8, f'{dl}{label if li == 0 else ""}',
                    fontsize=10, va='top', ha='left', color='#2C3E50',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=bg,
                              edgecolor=border, linewidth=0.8) if li == 0 else None)
        line_ys.append(y - (opt_lines_count - 1) * 1.8)
        y -= opt_lines_count * 1.8 + 1.2

    ymin_after_opts = y

    # Reset y to align with last option
    y = min(line_ys) - 3

    # ── Initial prediction display ──
    nopt_label = option_labels[prediction]
    nopt_text = options[prediction] if prediction < len(options) else '?'
    ax.text(x_left, y,
            f'Initial no-CoT prediction: {nopt_label}) {clean_step_text(nopt_text, max_len=100)}  '
            f'(probability = {pred_prob:.4f})',
            fontsize=10, va='top', ha='left', color='#555555')
    y -= 3.5

    # ── Divider ──
    ax.axhline(y=y + 1, xmin=0.02, xmax=0.98, color='#BDC3C7',
               linewidth=0.8, linestyle='-')
    y -= 2.5

    # ── Reasoning steps with heatmap highlighting ──
    ax.text(x_left, y, 'Reasoning Steps:', fontsize=11,
            fontweight='bold', color='#2C3E50')
    y -= 4.0

    for i, (step_text, ff_val) in enumerate(zip(segmented_cot, ff_values)):
        if y < 8:
            break

        cleaned = clean_step_text(step_text, max_len=120)
        wrapped = wrap_text(cleaned, width=line_width)
        lines = wrapped.split('\n')
        n_lines = len(lines)

        bg_color = GREEN_WHITE_RED(norm(ff_val), alpha=0.65)
        bg_rgb = mcolors.to_rgb(bg_color)
        luminance = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        tc = 'white' if luminance < 0.5 else 'black'

        header = f'Step {i + 1}'
        delta_text = f'  Δp = {ff_val:+.3f}'
        if ff_val > 0.01:
            delta_note = '  (supports prediction)'
        elif ff_val < -0.01:
            delta_note = '  (counteracts prediction)'
        else:
            delta_note = '  (no effect)'

        header_full = f'{header}:{delta_text}{delta_note}'
        n_header_lines = len(wrap_text(header_full, width=line_width).split('\n'))
        total_lines = n_lines + n_header_lines

        block_height = total_lines * 2.0 + 1.0
        y_bottom = y - block_height

        rect = FancyBboxPatch(
            (x_left - 1, y_bottom), 96, block_height,
            boxstyle='round,pad=0.3',
            facecolor=bg_color, edgecolor='#AAAAAA', linewidth=1.0
        )
        ax.add_patch(rect)

        y_text = y - 0.6
        ax.text(x_left, y_text, header_full, fontsize=9.5,
                fontweight='bold', va='top', ha='left', color=tc)
        y_text -= 2.0

        for line in lines:
            ax.text(x_left + 2, y_text, line, fontsize=9.5,
                    va='top', ha='left', color=tc)
            y_text -= 1.8

        y = y_bottom - 1.2

    # ── Colorbar ──
    y -= 1.5
    cbar_ax = fig.add_axes([0.15, 0.01, 0.70, 0.025])
    cbar = matplotlib.colorbar.ColorbarBase(
        cbar_ax, cmap=GREEN_WHITE_RED, norm=norm,
        orientation='horizontal'
    )
    cbar.set_label(
        'FF-SOFT (Δp)  ← green: probability added to initial answer  |  '
        'white: no effect  |  red: probability removed →',
        fontsize=9, labelpad=2
    )
    cbar.ax.tick_params(labelsize=8)

    # Annotate range endpoints
    cbar_ax.text(-0.05, -0.5, f'−{ff_abs_max:.2f}', transform=cbar_ax.transAxes,
                 fontsize=8, ha='center', va='top')
    cbar_ax.text(0.5, -0.5, '0', transform=cbar_ax.transAxes,
                 fontsize=8, ha='center', va='top')
    cbar_ax.text(1.05, -0.5, f'+{ff_abs_max:.2f}', transform=cbar_ax.transAxes,
                 fontsize=8, ha='center', va='top')

    # ── Footer ──
    ax.text(x_center, 0.5,
            f'Correct answer: {correct_letter}  |  Model predicted: {pred_letter}  '
            f'(initial prob: {pred_prob:.4f})',
            fontsize=9, ha='center', va='center', color='#888888')

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'Heatmap saved to: {output_path}')
    plt.close()


# ─── Command-line interface ───────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Figure 4-style FF-SOFT heatmap'
    )
    parser.add_argument('--dataset', '-d', type=str, default=DEFAULT_DATASET,
                        choices=['openbook', 'sqa'],
                        help=f'Dataset name (default: {DEFAULT_DATASET})')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL,
                        choices=['LLaMA-3-3B', 'Phi-3'],
                        help=f'Model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--instance', '-i', type=str, default=DEFAULT_INSTANCE,
                        help=f'Instance ID (default: {DEFAULT_INSTANCE})')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename (default: auto-generated)')
    parser.add_argument('--list', action='store_true',
                        help='List available instances in the selected dataset')

    args = parser.parse_args()

    # Build output filename
    if args.output:
        output_file = args.output
    else:
        output_file = (f'figure4_heatmap_{args.dataset}_{args.model}_'
                       f'{args.instance.replace("/", "_")}.pdf')

    print(f'{"=" * 60}')
    print(f'Dataset: {args.dataset}  |  Model: {args.model}')
    print(f'Instance: {args.instance}')
    print(f'{"=" * 60}\n')

    if args.list:
        results_path = os.path.join(RESULTS_BASE,
                                    DATASET_CONFIG[args.dataset][args.model])
        all_results = load_jsonl(results_path)
        unique_ids = sorted(set(r['id'] for r in all_results),
                            key=lambda x: int(x.split('-')[0]) if '-' in x else 0)
        print(f'Available instances ({len(unique_ids)} total):')
        for uid in unique_ids[:30]:
            q = next(r['question'] for r in all_results if r['id'] == uid)
            print(f'  [{uid}] {q[:80]}')
        if len(unique_ids) > 30:
            print(f'  ... and {len(unique_ids) - 30} more')
        sys.exit(0)

    # Load data
    instance_steps = load_instance_results(args.dataset, args.model,
                                           args.instance)
    num_found = len(instance_steps)
    print(f'Loaded {num_found} unlearned step results')

    # Build profile
    profile = extract_instance_profile(instance_steps)

    print(f'\nQuestion: {profile["question"]}')
    print(f'Options: {profile["options"]}')
    print(f'Correct: {profile["correct_letter"]}  |  '
          f'Predicted: {LETTERS[profile["prediction"]]}  '
          f'(prob={profile["initial_probs"][profile["prediction"]]:.4f})')
    print(f'\nFF-SOFT values per step:')
    for i, (step_text, ff_val) in enumerate(
            zip(profile['segmented_cot'], profile['ff_soft_values'])
    ):
        truncated = clean_step_text(step_text, max_len=70)
        color = 'red' if ff_val > 0.02 else ('green' if ff_val < -0.02 else 'gray')
        print(f'  Step {i}: Δp = {ff_val:+.4f} [{color}]  {truncated}')

    # Generate heatmap
    print(f'\nGenerating heatmap → {output_file}')
    create_heatmap(profile, output_path=output_file)
    print('Done!')