"""Metrics wrapper for ``new_lmf`` result files."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lmf.compute_metrics as metrics


metrics.DEFAULT_RESULTS_DIR = ROOT / "new_lmf" / "final_results"
metrics.DEFAULT_OUTPUT_CSV = ROOT / "new_lmf" / "metrics_summary.csv"
metrics.DEFAULT_OUTPUT_JSON = ROOT / "new_lmf" / "metrics_summary.json"
metrics.KNOWN_METHOD_PREFIXES = ["npo_KL_lmf"] + metrics.KNOWN_METHOD_PREFIXES

_old_parse_metadata = metrics.parse_metadata
_old_write_csv = metrics.write_csv
_old_print_table = metrics.print_table


def parse_metadata(path: Path, results_dir: Path) -> dict[str, str | None]:
    meta = _old_parse_metadata(path, results_dir)
    stem = path.name.removesuffix(".out")
    for token in stem.split("_"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in {"lmf", "lambda"}:
            meta[key] = value
    return meta


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "group", "dataset", "model", "method", "lr", "lmf", "lambda",
        "filename", "n_step_results", "n_agree_step_results", "n_instances",
        "n_agree_instances", "ff_hard", "faithfulness_paper_agree",
        "ff_soft", "specificity", "efficacy", "path",
    ]
    import csv

    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=base_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows):
    headers = ["dataset", "model", "lr", "lmf", "ff-hard", "paper-ff", "ff-soft", "specificity", "efficacy"]
    table = []
    for row in rows:
        table.append([
            row.get("dataset") or "NA",
            row.get("model") or "NA",
            row.get("lr") or "NA",
            row.get("lmf") or "NA",
            metrics.fmt(row.get("ff_hard")),
            metrics.fmt(row.get("faithfulness_paper_agree")),
            metrics.fmt(row.get("ff_soft"), 4),
            metrics.fmt(row.get("specificity")),
            metrics.fmt(row.get("efficacy")),
        ])
    widths = [max(len(str(v)) for v in [h] + [r[i] for r in table]) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print("  ".join(v.ljust(widths[i]) for i, v in enumerate(row)))


metrics.parse_metadata = parse_metadata
metrics.write_csv = write_csv
metrics.print_table = print_table


if __name__ == "__main__":
    raise SystemExit(metrics.main())
