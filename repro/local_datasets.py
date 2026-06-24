"""Reproduction-only dataset handlers.

These subclass the upstream OpenQA / SQA handlers and only override
``get_dataset_splits`` to read the local 50-instance subsets under ``data/``,
so the 2x2 reproduction is pinned to a fixed set of instances and does not
depend on network / HuggingFace access. Replace the upstream registry with::

    from repro.local_datasets import LOCAL_DATASETS
"""
import json
import os

from dataload import OpenQA, SQA  # reuse upstream prompt templates / answer logic


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBQA_SAMPLE = os.path.join(REPO_ROOT, "data", "openbookqa", "openbook_sample.json")
SQA_SAMPLE  = os.path.join(REPO_ROOT, "data", "strategyqa", "strategyqa_train.json")


def _load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"{path} must be a JSON list"
    return data


class LocalOpenQA(OpenQA):
    """OpenBookQA from the local 50-instance subset.

    Upstream OpenQA fetches via load_dataset('allenai/openbookqa'); the local
    schema matches (id / question_stem / choices / answerKey), so every other
    method (make_cot_prompt etc.) is inherited unchanged.
    """

    def get_dataset_splits(self):
        all_data = _load_json_list(OBQA_SAMPLE)
        # Use all 50 as test; upstream generate_dataset_cots only reads the
        # test split. train/valid stay empty to avoid downstream None errors.
        return [], [], all_data


class LocalSQA(SQA):
    """StrategyQA from the local 50-instance subset.

    Upstream SQA falls back to ``test = valid`` when test is empty, so we put
    the data in valid (first 8 reserved as demonstrations) and leave test empty.
    """

    def get_dataset_splits(self):
        all_data = _load_json_list(SQA_SAMPLE)
        sqa_train  = all_data[:8]   # demonstrations, matching upstream
        sqa_valid  = all_data[8:]
        sqa_test   = []             # empty -> evaluate.py uses test = valid
        return sqa_train, sqa_valid, sqa_test


LOCAL_DATASETS = {
    "openbook": LocalOpenQA(),
    "sqa":      LocalSQA(),
}
