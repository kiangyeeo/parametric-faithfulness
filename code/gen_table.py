"""Generate a PNG table image from table_all_word_types.csv."""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/wanyizhou/measuring cot/parametric-faithfulness/table_all_word_types.csv"
OUT_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/wanyizhou/measuring cot/parametric-faithfulness/table_all_word_types.png"

# Read CSV
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    data = list(reader)

headers = data[0]
rows = data[1:]

# Display columns: dataset, word_type, ff_hard(%), ff_soft, specificity(%), efficacy(%)
# (model/method/lr/seed are the same for all rows, so we skip them for a cleaner table)
col_indices = [0, 3, 6, 7, 8, 9]
display_headers = [headers[i] for i in col_indices]
display_rows = [[row[i] for i in col_indices] for row in rows]

# Add a separator row between datasets
final_rows = []
prev_dataset = None
for row in display_rows:
    if prev_dataset is not None and row[0] != prev_dataset:
        final_rows.append([""] * len(display_headers))
    final_rows.append(row)
    prev_dataset = row[0]

n_cols = len(display_headers)
n_rows = len(final_rows)

fig, ax = plt.subplots(figsize=(12, 0.5 + 0.45 * (n_rows + 1)))
ax.axis("off")

# Build cell text
cell_text = []
cell_colors = []
for i, row in enumerate(final_rows):
    is_sep = all(c == "" for c in row)
    cell_text.append(row)
    if is_sep:
        cell_colors.append(["#FFFFFF"] * n_cols)
    else:
        bg = "#F5F5F5" if i % 2 == 0 else "#FFFFFF"
        cell_colors.append([bg] * n_cols)

table = ax.table(
    cellText=cell_text,
    colLabels=display_headers,
    cellColours=cell_colors,
    colColours=["#4472C4"] * n_cols,
    cellLoc="center",
    loc="center",
)

# Style header
for j in range(n_cols):
    cell = table[0, j]
    cell.set_text_props(color="white", fontweight="bold", fontsize=11)
    cell.set_height(0.06)

# Style data cells
for i in range(1, n_rows + 1):
    for j in range(n_cols):
        cell = table[i, j]
        cell.set_text_props(fontsize=10)
        cell.set_height(0.05)
        # Bold the dataset name column
        if j == 0 and cell.get_text().get_text():
            cell.set_text_props(fontweight="bold", fontsize=10)

# Column widths
col_widths = [0.14, 0.16, 0.14, 0.14, 0.18, 0.14]
for j, w in enumerate(col_widths):
    for i in range(n_rows + 1):
        table[i, j].set_width(w)

plt.title("Unlearning Results by Word Type (LLaMA-3-3B, npo_KL, lr=3e-05, seed=1001)",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {OUT_PATH}")
