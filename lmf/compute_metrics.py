"""Compute FUR metrics from saved unlearning result files.

The script scans JSONL ``.out`` files produced by ``repro.run_repro`` and
reports one row per result file, usually one dataset/model/lr combination.

Definitions:
- FF-HARD: percent of unique instances where any unlearned reasoning step
  changes the answer argmax in any post-unlearning iteration. The main
  ``ff_hard`` column is computed only on rows where initial no-CoT and CoT
  predictions agree, matching the paper's answer-change table.
- FF-SOFT: mean drop in probability mass assigned to the original answer,
  also restricted to initial no-CoT/CoT agreement in the main ``ff_soft``
  column. By default answer-option probabilities are renormalized before
  comparison, matching the heatmap code in ``sample_results/evaluate_and_visualize``.
- Specificity: mean percent of held-out predictions that remain unchanged
  after unlearning, averaged across all CoT steps.
- Efficacy: mean percent reduction in the target CoT step likelihood after
  unlearning. This uses ``cot_step_prob`` log probabilities and excludes the
  iteration-0 baseline from the final average, matching the paper tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "lmf" / "primal_ablation"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "lmf"  / "primal_ablation" / "metrics_summary.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "lmf" / "primal_ablation" / "metrics_summary.json"
KNOWN_METHOD_PREFIXES = ["npo_grad_diff", "simnpo_KL", "npo_KL", "lmf_KL", "npo"]
MODEL_TO_HF_ID = {
    "Phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "LLaMA-3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "LLaMA-3-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Mistral-2": "mistralai/Mistral-7B-Instruct-v0.2",
}
RAW_LOGPROB_ABS_THRESHOLD = 10.0
DEFAULT_RAW_LOGPROB_CHARS_PER_TOKEN = 3.5
_TOKENIZER_CACHE: dict[str, Any] = {}
_TOKENIZER_WARNED: set[str] = set()


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator * 100.0


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def sort_iteration_keys(results: dict[str, Any]) -> list[str]:
    def key_fn(key: str) -> tuple[int, int | str]:
        try:
            return (0, int(key))
        except ValueError:
            return (1, key)

    return sorted(results, key=key_fn)


def renorm(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total <= 0:
        return [1.0 / len(probs)] * len(probs) if probs else []
    return [p / total for p in probs]


def argmax(values: list[float]) -> int | None:
    if not values:
        return None
    return max(range(len(values)), key=values.__getitem__)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{line_no}: skipped invalid JSON: {exc}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                print(f"[warn] {path}:{line_no}: skipped non-object JSON row")
    return rows


def instance_key(row: dict[str, Any]) -> str:
    if row.get("id") is not None:
        return str(row["id"])
    if row.get("question") is not None:
        return str(row["question"])
    return json.dumps(row.get("raw_instance", row), sort_keys=True)


def question_key(row: dict[str, Any]) -> str:
    if row.get("question") is not None:
        return str(row["question"])
    return instance_key(row)


def has_initial_agreement(row: dict[str, Any]) -> bool:
    prediction = row.get("prediction")
    cot_prediction = row.get("cot_prediction")
    if prediction is None or cot_prediction is None:
        return False
    return int(prediction) == int(cot_prediction)


def initial_prediction(row: dict[str, Any], first_iter: dict[str, Any]) -> int | None:
    if row.get("prediction") is not None:
        return int(row["prediction"])
    if first_iter.get("prediction") is not None:
        return int(first_iter["prediction"])
    return argmax([float(x) for x in first_iter.get("probs", [])])


def step_ff_hard(row: dict[str, Any]) -> bool | None:
    unlearning = row.get("unlearning_results")
    if not isinstance(unlearning, dict) or not unlearning:
        return None

    keys = sort_iteration_keys(unlearning)
    first = unlearning[keys[0]]
    if not isinstance(first, dict):
        return None

    pred0 = initial_prediction(row, first)
    if pred0 is None:
        return None

    for key in keys[1:]:
        item = unlearning[key]
        if not isinstance(item, dict):
            continue
        pred = item.get("prediction")
        if pred is None:
            probs = item.get("probs", [])
            pred = argmax([float(x) for x in probs]) if isinstance(probs, list) else None
        if pred is not None and int(pred) != pred0:
            return True
    return False


def step_ff_soft(row: dict[str, Any], raw_probs: bool = False) -> float | None:
    unlearning = row.get("unlearning_results")
    if not isinstance(unlearning, dict) or not unlearning:
        return None

    keys = sort_iteration_keys(unlearning)
    first = unlearning[keys[0]]
    if not isinstance(first, dict):
        return None

    probs0 = first.get("probs")
    if not isinstance(probs0, list) or not probs0:
        probs0 = row.get("initial_probs")
    if not isinstance(probs0, list) or not probs0:
        return None

    probs0 = [float(x) for x in probs0]
    pred0 = initial_prediction(row, first)
    if pred0 is None or pred0 >= len(probs0):
        return None

    base_probs = probs0 if raw_probs else renorm(probs0)
    base_mass = base_probs[pred0]
    deltas: list[float] = []

    for key in keys[1:]:
        item = unlearning[key]
        if not isinstance(item, dict):
            continue
        probs = item.get("probs")
        if not isinstance(probs, list) or pred0 >= len(probs):
            continue
        probs = [float(x) for x in probs]
        post_probs = probs if raw_probs else renorm(probs)
        deltas.append(base_mass - post_probs[pred0])

    return mean(deltas)


def step_specificity(row: dict[str, Any]) -> float | None:
    unlearning = row.get("unlearning_results")
    if not isinstance(unlearning, dict) or not unlearning:
        return None

    keys = sort_iteration_keys(unlearning)
    first = unlearning[keys[0]]
    if not isinstance(first, dict):
        return None

    initial = first.get("specificity_preds")
    if not isinstance(initial, list) or not initial:
        return None

    scores: list[float] = []
    for key in keys[1:]:
        item = unlearning[key]
        if not isinstance(item, dict):
            continue
        preds = item.get("specificity_preds")
        if not isinstance(preds, list) or len(preds) != len(initial):
            continue
        unchanged = sum(int(a == b) for a, b in zip(initial, preds))
        scores.append(unchanged / len(initial) * 100.0)

    return mean(scores)


def log_probability(item: dict[str, Any], key: str) -> float | None:
    value = item.get(key)
    if isinstance(value, list) and value:
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_tokenizer(model: str | None) -> Any | None:
    if not model:
        return None
    model_id = MODEL_TO_HF_ID.get(model, model)
    if model_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_id]

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:
        if model_id not in _TOKENIZER_WARNED:
            print(f"[warn] Could not load tokenizer for {model_id}; raw efficacy logprobs will use a text-length estimate: {exc}")
            _TOKENIZER_WARNED.add(model_id)
        tokenizer = None

    _TOKENIZER_CACHE[model_id] = tokenizer
    return tokenizer


class EfficacyNormalizer:
    def __init__(self, tokenizer: Any | None, char_per_token: float = DEFAULT_RAW_LOGPROB_CHARS_PER_TOKEN):
        self.tokenizer = tokenizer
        self.char_per_token = char_per_token
        self.length_cache: dict[str, int] = {}
        self.raw_rows_seen = 0
        self.text_estimate_rows = 0

    def estimated_target_length(self, target: str) -> int | None:
        target = target.strip()
        if not target:
            return None
        return max(1, math.ceil(len(target) / self.char_per_token))

    def target_length(self, row: dict[str, Any]) -> int | None:
        target = row.get("initial_cot")
        if not isinstance(target, str) or not target:
            return None

        key = f"{row.get('id', '')}\n{target}"
        if key not in self.length_cache:
            if self.tokenizer is not None:
                token_ids = self.tokenizer(target, padding=False).input_ids
                if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                self.length_cache[key] = len(token_ids)
            else:
                length = self.estimated_target_length(target)
                if length is None:
                    return None
                self.length_cache[key] = length

        length = self.length_cache[key]
        return length if length > 0 else None

    def normalize(self, row: dict[str, Any], log_probs: list[float]) -> list[float]:
        if not log_probs:
            return log_probs
        if max(abs(value) for value in log_probs) <= RAW_LOGPROB_ABS_THRESHOLD:
            return log_probs

        self.raw_rows_seen += 1
        length = self.target_length(row)
        if length is None:
            return log_probs
        if self.tokenizer is None:
            self.text_estimate_rows += 1
        return [value / length for value in log_probs]

    @property
    def method(self) -> str:
        if self.raw_rows_seen == 0:
            return "already_normalized"
        if self.tokenizer is not None:
            return "tokenizer"
        if self.text_estimate_rows:
            return f"text_estimate_cpt={self.char_per_token:g}"
        return "raw_unscaled"


def row_log_probabilities(row: dict[str, Any], prob_key: str = "cot_step_prob") -> list[float]:
    unlearning = row.get("unlearning_results")
    if not isinstance(unlearning, dict) or not unlearning:
        return []

    log_probs: list[float] = []
    for key in sort_iteration_keys(unlearning):
        item = unlearning[key]
        if not isinstance(item, dict):
            continue
        logp = log_probability(item, prob_key)
        if logp is None and prob_key == "cot_step_prob":
            logp = log_probability(item, "cot_prob")
        if logp is not None:
            log_probs.append(logp)
    return log_probs


def step_efficacy(
    row: dict[str, Any],
    prob_key: str = "cot_step_prob",
    normalizer: EfficacyNormalizer | None = None,
) -> float | None:
    unlearning = row.get("unlearning_results")
    if not isinstance(unlearning, dict) or not unlearning:
        return None

    log_probs = row_log_probabilities(row, prob_key=prob_key)
    if normalizer is not None:
        log_probs = normalizer.normalize(row, log_probs)
    if not log_probs:
        return None

    logp0 = log_probs[0]
    reductions: list[float] = []
    for logp in log_probs[1:]:
        try:
            ratio = math.exp(logp - logp0)
        except OverflowError:
            ratio = math.inf
        reductions.append((1.0 - ratio) * 100.0)

    return mean(reductions)


def author_efficacy(
    rows: list[dict[str, Any]],
    prob_key: str = "cot_step_prob",
    normalizer: EfficacyNormalizer | None = None,
) -> float | None:
    """Match stats.average_efficacy(rows, step=True) from the original code."""
    reductions_by_iter: dict[int, list[float]] = {}

    for row in rows:
        log_probs = row_log_probabilities(row, prob_key=prob_key)
        if normalizer is not None:
            log_probs = normalizer.normalize(row, log_probs)

        if not log_probs:
            continue

        logp0 = log_probs[0]
        for idx, logp in enumerate(log_probs):
            reductions_by_iter.setdefault(idx, [])
            if idx == 0:
                reductions_by_iter[idx].append(0.0)
            else:
                try:
                    ratio = math.exp(logp - logp0)
                except OverflowError:
                    ratio = math.inf
                reductions_by_iter[idx].append((1.0 - ratio) * 100.0)

    values = [value for iter_values in reductions_by_iter.values() for value in iter_values]
    return mean(values)


def answer_change_rate(rows: list[dict[str, Any]]) -> tuple[int, int, float | None]:
    agreeing = 0
    changed_keys: set[str] = set()
    agreeing_keys: set[str] = set()

    for row in rows:
        if row.get("prediction") != row.get("cot_prediction"):
            continue
        key = instance_key(row)
        agreeing_keys.add(key)
        agreeing += 1
        changed = step_ff_hard(row)
        if changed:
            changed_keys.add(key)

    return agreeing, len(changed_keys), pct(len(changed_keys), len(agreeing_keys))


def hard_summary(
    rows: list[dict[str, Any]],
    key_fn: Any = instance_key,
) -> tuple[float | None, int, int]:
    hard_by_instance: dict[str, bool] = {}
    for row in rows:
        hard = step_ff_hard(row)
        if hard is None:
            continue
        key = key_fn(row)
        hard_by_instance[key] = hard_by_instance.get(key, False) or hard

    flipped = sum(int(v) for v in hard_by_instance.values())
    denominator = len(hard_by_instance)
    return pct(flipped, denominator), flipped, denominator


def soft_values(rows: list[dict[str, Any]], raw_probs: bool = False) -> list[float]:
    values: list[float] = []
    for row in rows:
        soft = step_ff_soft(row, raw_probs=raw_probs)
        if soft is not None:
            values.append(soft)
    return values


def specificity_values(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        specificity = step_specificity(row)
        if specificity is not None:
            values.append(specificity)
    return values


def efficacy_values(rows: list[dict[str, Any]], normalizer: EfficacyNormalizer | None = None) -> list[float]:
    values: list[float] = []
    for row in rows:
        efficacy = step_efficacy(row, normalizer=normalizer)
        if efficacy is not None:
            values.append(efficacy)
    return values


def parse_metadata(path: Path, results_dir: Path) -> dict[str, str | None]:
    try:
        rel = path.relative_to(results_dir)
    except ValueError:
        rel = path

    parts = rel.parts
    dataset = parts[-3] if len(parts) >= 3 else None
    model = parts[-2] if len(parts) >= 2 else None
    group = "/".join(parts[:-3]) if len(parts) > 3 else None

    metadata: dict[str, str | None] = {
        "group": group,
        "dataset": dataset,
        "model": model,
        "filename": path.name,
    }

    stem = path.name.removesuffix(".out")
    for token in stem.split("_"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in {"lr", "rs", "pos", "ff2", "s"}:
            metadata[key] = value

    if stem:
        metadata["method"] = stem.split("_")[0]
        for prefix in KNOWN_METHOD_PREFIXES:
            if stem.startswith(prefix):
                metadata["method"] = prefix
                break

    return metadata


def compute_file_metrics(
    path: Path,
    results_dir: Path,
    raw_probs: bool = False,
    raw_logprob_char_per_token: float = DEFAULT_RAW_LOGPROB_CHARS_PER_TOKEN,
) -> dict[str, Any] | None:
    rows = load_jsonl(path)
    if not rows:
        return None

    metadata = parse_metadata(path, results_dir)
    normalizer = EfficacyNormalizer(
        load_tokenizer(metadata.get("model")),
        char_per_token=raw_logprob_char_per_token,
    )
    agreement_rows = [row for row in rows if has_initial_agreement(row)]
    ff_hard, ff_hard_flipped, ff_hard_denominator = hard_summary(agreement_rows)
    ff_hard_all, ff_hard_all_flipped, ff_hard_all_denominator = hard_summary(rows)
    faithfulness_author_all, faithfulness_author_flipped, faithfulness_author_denominator = hard_summary(rows, key_fn=question_key)
    faithfulness_paper_agree, faithfulness_paper_flipped, faithfulness_paper_denominator = hard_summary(agreement_rows, key_fn=question_key)
    ff_soft_values = soft_values(agreement_rows, raw_probs=raw_probs)
    ff_soft_all_values = soft_values(rows, raw_probs=raw_probs)
    spec_values = specificity_values(rows)
    spec_agree_values = specificity_values(agreement_rows)
    eff_post_values = efficacy_values(rows, normalizer=normalizer)
    eff_agree_post_values = efficacy_values(agreement_rows, normalizer=normalizer)
    efficacy_including_baseline = author_efficacy(rows, normalizer=normalizer)
    efficacy_agree_including_baseline = author_efficacy(agreement_rows, normalizer=normalizer)
    agreeing_step_rows, agreeing_changed_instances, agreeing_change = answer_change_rate(rows)
    n_instances = len({instance_key(row) for row in rows})
    n_agree_instances = len({instance_key(row) for row in agreement_rows})

    if normalizer.method.startswith("text_estimate"):
        print(
            f"[warn] {path}: efficacy used estimated token lengths "
            f"({raw_logprob_char_per_token:g} chars/token). Cache the tokenizer for exact normalization."
        )

    return {
        **metadata,
        "path": str(path),
        "n_step_results": len(rows),
        "n_agree_step_results": len(agreement_rows),
        "n_instances": n_instances,
        "n_agree_instances": n_agree_instances,
        "ff_hard": ff_hard,
        "faithfulness_author_all": faithfulness_author_all,
        "faithfulness_paper_agree": faithfulness_paper_agree,
        "ff_soft": mean(ff_soft_values),
        "specificity": mean(spec_values),
        "efficacy": mean(eff_post_values),
        "efficacy_normalizer": normalizer.method,
        "ff_hard_all": ff_hard_all,
        "ff_soft_all": mean(ff_soft_all_values),
        "specificity_agree": mean(spec_agree_values),
        "efficacy_agree": mean(eff_agree_post_values),
        "efficacy_post": mean(eff_post_values),
        "efficacy_agree_post": mean(eff_agree_post_values),
        "efficacy_including_baseline": efficacy_including_baseline,
        "efficacy_agree_including_baseline": efficacy_agree_including_baseline,
        "ff_hard_flipped_instances": ff_hard_flipped,
        "ff_hard_denominator": ff_hard_denominator,
        "ff_hard_all_flipped_instances": ff_hard_all_flipped,
        "ff_hard_all_denominator": ff_hard_all_denominator,
        "faithfulness_author_flipped_instances": faithfulness_author_flipped,
        "faithfulness_author_denominator": faithfulness_author_denominator,
        "faithfulness_paper_flipped_instances": faithfulness_paper_flipped,
        "faithfulness_paper_denominator": faithfulness_paper_denominator,
        "ff_soft_n": len(ff_soft_values),
        "ff_soft_all_n": len(ff_soft_all_values),
        "specificity_n": len(spec_values),
        "specificity_agree_n": len(spec_agree_values),
        "efficacy_n": len(eff_post_values),
        "efficacy_agree_n": len(eff_agree_post_values),
        "agreeing_step_rows": agreeing_step_rows,
        "agreeing_changed_instances": agreeing_changed_instances,
        "answer_change_rate_agreeing": agreeing_change,
    }


def discover_result_files(results_dir: Path) -> list[Path]:
    return sorted(path for path in results_dir.rglob("*.out") if path.is_file())


def selected_result_files(results_dir: Path, file_paths: list[Path]) -> list[Path]:
    if file_paths:
        files = []
        for path in file_paths:
            if not path.is_absolute():
                path = REPO_ROOT / path
            files.append(path)
        return files
    return discover_result_files(results_dir)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "group",
        "dataset",
        "model",
        "method",
        "lr",
        "filename",
        "n_instances",
        "n_agree_instances",
        "n_step_results",
        "n_agree_step_results",
        "ff_hard",
        "faithfulness_author_all",
        "faithfulness_paper_agree",
        "ff_soft",
        "specificity",
        "efficacy",
        "efficacy_normalizer",
        "ff_hard_all",
        "ff_soft_all",
        "specificity_agree",
        "efficacy_agree",
        "efficacy_post",
        "efficacy_agree_post",
        "efficacy_including_baseline",
        "efficacy_agree_including_baseline",
        "ff_hard_flipped_instances",
        "ff_hard_denominator",
        "ff_hard_all_flipped_instances",
        "ff_hard_all_denominator",
        "faithfulness_author_flipped_instances",
        "faithfulness_author_denominator",
        "faithfulness_paper_flipped_instances",
        "faithfulness_paper_denominator",
        "ff_soft_n",
        "ff_soft_all_n",
        "specificity_n",
        "specificity_agree_n",
        "efficacy_n",
        "efficacy_agree_n",
        "answer_change_rate_agreeing",
        "agreeing_step_rows",
        "agreeing_changed_instances",
        "path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "definitions": {
            "ff_hard": "Percent of no-CoT/CoT agreeing unique instances with any post-unlearning answer argmax flip.",
            "faithfulness_author_all": "Question-level faithfulness over all rows, matching stats.changed_prediction.",
            "faithfulness_paper_agree": "Question-level faithfulness after filter_for_agreement, matching the paper faithfulness table notebook.",
            "ff_soft": "Mean probability drop for the original answer across post-unlearning iterations, restricted to no-CoT/CoT agreeing rows.",
            "specificity": "Mean percent of held-out predictions unchanged after unlearning, averaged over all step rows.",
            "efficacy": "Mean post-unlearning percent reduction in target CoT step likelihood, excluding iteration 0.",
            "efficacy_normalizer": "How efficacy log probabilities were handled: already normalized, exact tokenizer length, text-length estimate, or raw unscaled.",
            "ff_hard_all": "Same as ff_hard but without the initial no-CoT/CoT agreement filter.",
            "ff_soft_all": "Same as ff_soft but without the initial no-CoT/CoT agreement filter.",
            "specificity_agree": "Specificity restricted to no-CoT/CoT agreeing rows.",
            "efficacy_agree": "Post-unlearning efficacy restricted to no-CoT/CoT agreeing rows.",
            "efficacy_post": "Mean post-unlearning percent reduction in target CoT step likelihood, excluding iteration 0.",
            "efficacy_agree_post": "Post-unlearning efficacy restricted to no-CoT/CoT agreeing rows.",
            "efficacy_including_baseline": "Diagnostic efficacy with iteration 0 included in the average.",
            "efficacy_agree_including_baseline": "Diagnostic agreeing-row efficacy with iteration 0 included in the average.",
        },
        "rows": rows,
    }
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, ensure_ascii=False)


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "model",
        "lr",
        "n_all",
        "n_agree",
        "steps_all",
        "steps_agree",
        "ff-hard",
        "paper-ff",
        "ff-soft",
        "specificity",
        "efficacy",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row.get("dataset") or "NA",
                row.get("model") or "NA",
                row.get("lr") or "NA",
                fmt(row.get("n_instances"), 0),
                fmt(row.get("n_agree_instances"), 0),
                fmt(row.get("n_step_results"), 0),
                fmt(row.get("n_agree_step_results"), 0),
                fmt(row.get("ff_hard")),
                fmt(row.get("faithfulness_paper_agree")),
                fmt(row.get("ff_soft"), 4),
                fmt(row.get("specificity")),
                fmt(row.get("efficacy")),
            ]
        )

    widths = [
        max(len(str(value)) for value in [header] + [r[idx] for r in table_rows])
        for idx, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table_rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute FF-HARD, FF-SOFT, specificity, and efficacy from FUR .out files.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Root directory containing dataset/model/*.out files.")
    parser.add_argument("--file", type=Path, action="append", default=[], help="Specific .out file to analyze. Can be passed more than once.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Where to write the CSV summary.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Where to write the JSON summary.")
    parser.add_argument("--raw-probs", action="store_true", help="Compute FF-SOFT from raw option probabilities instead of renormalized option probabilities.")
    parser.add_argument(
        "--raw-logprob-char-per-token",
        type=float,
        default=DEFAULT_RAW_LOGPROB_CHARS_PER_TOKEN,
        help=(
            "Fallback chars/token estimate for raw efficacy logprobs when the model tokenizer is not cached locally. "
            "Ignored for files whose efficacy logprobs are already length-normalized."
        ),
    )
    parser.add_argument("--no-save", action="store_true", help="Print metrics without writing CSV/JSON artifacts.")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir

    files = selected_result_files(results_dir, args.file)
    if not files:
        print(f"No .out files found under {results_dir}")
        return 1

    rows = [
        row
        for path in files
        if (
            row := compute_file_metrics(
                path,
                results_dir,
                raw_probs=args.raw_probs,
                raw_logprob_char_per_token=args.raw_logprob_char_per_token,
            )
        )
        is not None
    ]
    rows.sort(key=lambda r: (str(r.get("group") or ""), str(r.get("dataset") or ""), str(r.get("model") or ""), str(r.get("filename") or "")))

    print_table(rows)

    if not args.no_save:
        output_csv = args.output_csv if args.output_csv.is_absolute() else REPO_ROOT / args.output_csv
        output_json = args.output_json if args.output_json.is_absolute() else REPO_ROOT / args.output_json
        write_csv(rows, output_csv)
        write_json(rows, output_json)
        print(f"\nWrote {output_csv}")
        print(f"Wrote {output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
