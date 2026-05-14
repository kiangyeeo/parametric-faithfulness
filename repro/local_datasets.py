"""
repro/local_datasets.py

复现专用的数据 handler。继承源仓库的 OpenQA / SQA，
只把 get_dataset_splits 改成读 data/ 下的本地 50 条子集。

为什么这么做：源代码默认走 HuggingFace load_dataset(),
而我们 2x2 复现要 (a) 限定在挑好的 50 条 ID 上，
(b) 不依赖联网/HF 缓存，结果可复现。

用法：在 entry script 里
    from repro.local_datasets import LOCAL_DATASETS as DATASETS
就能整个替换掉源仓库的 DATASETS 字典。
"""
import json
import os

from dataload import OpenQA, SQA  # 复用源仓库的 prompt 模板/答题逻辑


# 数据子集路径 —— 与仓库现有 data/ 目录对齐
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBQA_SAMPLE = os.path.join(REPO_ROOT, "data", "openbookqa", "openbook_sample.json")
SQA_SAMPLE  = os.path.join(REPO_ROOT, "data", "strategyqa", "strategyqa_train.json")


def _load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"{path} 应该是 JSON 列表"
    return data


class LocalOpenQA(OpenQA):
    """复现用：读本地 openbook_sample.json (50 条)。

    源 OpenQA 通过 load_dataset('allenai/openbookqa') 联网取数。
    instance schema 一致（id / question_stem / choices / answerKey），
    所以除了 splits 之外其它方法（make_cot_prompt 等）都直接继承。
    """

    def get_dataset_splits(self):
        all_data = _load_json_list(OBQA_SAMPLE)
        # 50 条全部当 test 用（生成 CoT + unlearn 都从 test 走）；
        # train / valid 留空 list，避免后续逻辑空报错。
        # 注：源 generate_dataset_cots 只看 test 这一个 split。
        return [], [], all_data


class LocalSQA(SQA):
    """复现用：读本地 strategyqa_train.json (50 条)。

    源 SQA 已经支持本地路径，但默认 test_path 指向的文件是空 []。
    源代码用 `if dataset_id == 'sqa': test = valid` 来兜底，
    所以我们把 50 条整体放到 valid 里，再让 test=valid 即可。
    """

    def get_dataset_splits(self):
        all_data = _load_json_list(SQA_SAMPLE)
        # 前 8 条留作 demonstration（与源代码保持一致），
        # 剩下的当 valid。test 仍然为空，evaluate.py 会自动 fallback 到 valid。
        sqa_train  = all_data[:8]
        sqa_valid  = all_data[8:]
        sqa_test   = []  # 保持空，让 evaluate.py 的 `test = valid` 生效
        return sqa_train, sqa_valid, sqa_test


# entry script 里这样 import：from repro.local_datasets import LOCAL_DATASETS
LOCAL_DATASETS = {
    "openbook": LocalOpenQA(),
    "sqa":      LocalSQA(),
}
