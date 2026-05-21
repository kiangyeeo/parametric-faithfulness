import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, 'analysis_summary.json'), 'r') as f:
    analysis_data = json.load(f)

with open(os.path.join(SCRIPT_DIR, 'answer_change_rate_summary.json'), 'r') as f:
    change_rate_data = json.load(f)


def create_table1():
    """
    Table 1: Faithfulness, Efficacy, Specificity
    数据来源: analysis_summary.json
    """
    datasets = ['openbook', 'sqa']
    models = ['LLaMA-3-3B', 'Phi-3']
    metrics = ['faithfulness', 'efficacy', 'specificity']
    metric_labels = ['Faithfulness', 'Efficacy', 'Specificity']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    header = ['Model', 'Dataset'] + metric_labels
    table_data = [header]

    for model in models:
        for dataset in datasets:
            key = f"{dataset}_{model}"
            d = analysis_data[key]
            row = [model, dataset.capitalize()]
            for m in metrics:
                row.append(f"{d[m]:.1f}")
            table_data.append(row)

    table = ax.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.20, 0.15, 0.20, 0.20, 0.25],
        edges='closed',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#4a5568')
        cell.set_text_props(color='white', weight='bold')

    row_colors = ['#f7fafc', '#edf2f7']
    for i in range(1, len(table_data)):
        for j in range(len(header)):
            cell = table[(i, j)]
            cell.set_facecolor(row_colors[(i - 1) % 2])

    plt.title('Table 1: Core Evaluation Metrics', fontsize=12, fontweight='bold', pad=20)

    plt.savefig(os.path.join(SCRIPT_DIR, 'table1_core_metrics.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(SCRIPT_DIR, 'table1_core_metrics.png'), dpi=300, bbox_inches='tight')
    print("Table 1 saved: table1_core_metrics.pdf / .png")


def create_table2():
    """
    Table 2: % of instances where adding mistakes or unlearning a
    reasoning step changes the model's answer.
    Measured only on instances where no-CoT and CoT predictions agree.
    Scores >= 1% displayed in bold.

    数据来源: answer_change_rate_summary.json
    """
    datasets = ['openbook', 'sqa']
    models = ['LLaMA-3-3B', 'Phi-3']

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('off')

    header = [
        'Model', 'Dataset',
        'Mistake\nAgreeing', 'Mistake\nChanged', 'Mistake\nRate (%)',
        'Unlearn\nAgreeing', 'Unlearn\nChanged', 'Unlearn\nRate (%)',
    ]
    table_data = [header]
    bold_cells = []  # (row, col) tuples for bold formatting

    for model in models:
        for dataset in datasets:
            key = f"{dataset}_{model}"
            m = change_rate_data['adding_mistakes'][key]
            u = change_rate_data['unlearning_steps'][key]

            m_rate = m['change_rate_percent']
            u_rate = u['change_rate_percent']

            row = [
                model,
                dataset.capitalize(),
                str(m['agreeing_instances']),
                str(m['changed_instances']),
                f"{m_rate:.1f}",
                str(u['agreeing_instances']),
                str(u['changed_instances']),
                f"{u_rate:.1f}",
            ]
            row_idx = len(table_data)
            table_data.append(row)

            if m_rate >= 1.0:
                bold_cells.append((row_idx, 4))
            if u_rate >= 1.0:
                bold_cells.append((row_idx, 7))

    table = ax.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.12, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
        edges='closed',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#4a5568')
        cell.set_text_props(color='white', weight='bold')

    row_colors = ['#f7fafc', '#edf2f7']
    for i in range(1, len(table_data)):
        for j in range(len(header)):
            cell = table[(i, j)]
            cell.set_facecolor(row_colors[(i - 1) % 2])

    for r, c in bold_cells:
        table[(r, c)].set_text_props(weight='bold')

    plt.title(
        'Table 2: % of Instances Where Adding Mistakes or Unlearning\n'
        'a Reasoning Step Changes the Model\'s Answer',
        fontsize=12, fontweight='bold', pad=20,
    )
    ax.text(
        0.5, -0.18,
        'Note: Bold values indicate change rate >= 1%. '
        'Measured only on instances where no-CoT and CoT predictions agree.',
        transform=ax.transAxes, ha='center', fontsize=8, color='#718096',
    )

    plt.savefig(os.path.join(SCRIPT_DIR, 'table2_answer_change.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(SCRIPT_DIR, 'table2_answer_change.png'), dpi=300, bbox_inches='tight')
    print("Table 2 saved: table2_answer_change.pdf / .png")


if __name__ == "__main__":
    print("Generating Table 1: Core Evaluation Metrics ...")
    create_table1()

    print("Generating Table 2: Answer Change Rate ...")
    create_table2()

    print("\nDone.")