import nltk
import spacy
import torch
import random

import numpy as np

from dataclasses import dataclass, field

# ============================================================
# 词汇类型分组定义 (Vocabulary Type Groups)
# ============================================================
# 基于 Universal Dependencies POS 标签体系的词汇类型分类
# 每个分组对应一类语义/语法角色，用于按词汇类型选择性遗忘
WORD_TYPE_GROUPS = {
    'entity':      {'NOUN', 'PROPN'},                          # 实体词：名词、专有名词 — 承载事实性知识
    'attribute':   {'ADJ', 'NUM'},                             # 属性词：形容词、数词 — 承载量化/描述信息
    'action':      {'VERB'},                                   # 动作词：动词 — 承载关系/过程信息
    'function':    {'ADP', 'AUX', 'CCONJ', 'DET',              # 虚词：介词、助动词、连词、限定词、
                    'PART', 'PRON', 'SCONJ'},                  #       小品词、代词、从属连词
    'modifier':    {'ADV'},                                    # 修饰词：副词 — 承载程度/方式修饰
    'all_content': {'NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM',     # 全部实词（FUR原文配置）
                    'ADV'},
    'random':      None,                                       # 随机对照组（特殊处理，不在POS体系中）
}

# 向后兼容：保留原始 TARGET_TAGS 定义
TARGET_TAGS = set([
    'VERB',
    'NUM',
    'ADJ',
    'NOUN',
    'PROPN',
])

# 词汇类型组的中文显示名称，用于验证输出
WORD_TYPE_DISPLAY_NAMES = {
    'entity':      '实体词',
    'attribute':   '属性词',
    'action':      '动作词',
    'function':    '虚词',
    'modifier':    '修饰词',
    'all_content': '全部实词',
    'random':      '随机词',
}

# 词汇类型组的颜色标记（ANSI），用于终端验证输出
WORD_TYPE_COLORS = {
    'entity':      '\033[94m',   # 蓝色
    'attribute':   '\033[93m',   # 黄色
    'action':      '\033[91m',   # 红色
    'function':    '\033[90m',   # 灰色
    'modifier':    '\033[95m',   # 紫色
    'all_content': '\033[92m',   # 绿色
    'random':      '\033[96m',   # 青色
    'unknown':     '\033[0m',    # 默认
}
COLOR_RESET = '\033[0m'


def get_word_type_group(pos_tag):
    """根据POS标签确定词汇类型分组名称。
    
    Args:
        pos_tag: Universal Dependencies POS 标签字符串
        
    Returns:
        词汇类型分组名称字符串，若无法分类则返回 'unknown'
    """
    for group_name, tag_set in WORD_TYPE_GROUPS.items():
        if tag_set is not None and pos_tag in tag_set:
            return group_name
    return 'unknown'


def get_available_word_type_groups():
    """返回所有可用的词汇类型分组名称列表（排除 'random' 因其为特殊处理组）。"""
    return [g for g in WORD_TYPE_GROUPS if g != 'random']


def validate_word_type_group(group_name):
    """验证给定的词汇类型分组名称是否有效。
    
    Args:
        group_name: 待验证的分组名称
        
    Returns:
        有效则返回 True，否则返回 False
    """
    return group_name in WORD_TYPE_GROUPS


@dataclass
class Word:
    word: str
    pos: str
    span_start: int
    span_end: int

    def is_content(self):
      return self.pos in TARGET_TAGS

    def get_group(self):
        """获取该词所属的词汇类型分组名称。"""
        return get_word_type_group(self.pos)

    def belongs_to_group(self, group_name):
        """判断该词是否属于指定的词汇类型分组。
        
        Args:
            group_name: 词汇类型分组名称，如 'entity', 'action', 'function' 等
            
        Returns:
            True 如果该词属于指定分组，否则 False
        """
        if group_name not in WORD_TYPE_GROUPS:
            return False
        tag_set = WORD_TYPE_GROUPS[group_name]
        if tag_set is None:
            return False
        return self.pos in tag_set

    def get_display_name(self):
        """获取该词所属词汇类型的中文显示名称。"""
        group = self.get_group()
        return WORD_TYPE_DISPLAY_NAMES.get(group, '未知')

WHITESPACE_CHARS = {
    'meta-llama/Meta-Llama-3-8B-Instruct': 'Ġ',
    'microsoft/Phi-3-mini-4k-instruct': '▁',
    'mistralai/Mistral-7B-Instruct-v0.2': '▁',
    'meta-llama/Llama-3.2-3B-Instruct': 'Ġ',
    # 本地路径支持
    '/inspire/hdd/project/fdu-aidake-cfff/public/.huggingface/.hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95': 'Ġ',
}

def sentencize(text):
    return nltk.sent_tokenize(text)

def pos_tag(text, nlp):
    doc = nlp(text)
    return [(w.text, w.pos_) for w in doc]

def words_to_token_spans(wpos, tokens, W):
    # Filter out space tokens
    toks_pos = [(t, p) for t, p in wpos if p != "SPACE"]

    # Iterate over words
    i = 0
    cur_word, cur_pos = toks_pos[i]
    
    word_start = 0
    words = []

    for j, subword in enumerate(tokens):
        # if W == subword: continue
    
        if W in subword: # new word
            word_start = j 
    
        # Convert span to string, filter out whitespace
        span = tokens[word_start:j+1]
        # print(span[0], dir(span[0]))
        span = [e.replace(W, "") for e in span]
        cur = ''.join(span)
    
        # equality check
        if cur == cur_word:
    
            # Store span
            w = Word(cur_word, cur_pos, word_start, j+1)
            words.append(w)
    
            # Goto next
            i += 1
            if i >= len(toks_pos): break
            
            cur_word, cur_pos = toks_pos[i]
            word_start = j
    
    
    if not len(words) == len(toks_pos):
      print("Length mismatch")  
      print(words)
      print("-"*30)
      print([t for t,p in toks_pos])

    return words

def align_cot_to_pos(cot_step_text, tokenizer, model_id, nlp):    
    W = WHITESPACE_CHARS[model_id]
    w_p = pos_tag(cot_step_text, nlp)
    pretokenized_text = [f" {w}" for w,_ in w_p] # Take words, prefix whitespace
    tokens = tokenizer.tokenize(pretokenized_text, is_split_into_words=True, add_special_tokens=False)
    indices = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

    return indices, words_to_token_spans(w_p, tokens, W)


# ============================================================
# 词汇类型验证与可视化函数
# ============================================================

def classify_vocab_types(text, nlp):
    """对输入文本进行词汇类型分类，返回每个词的类型标注结果。
    
    Args:
        text: 输入文本字符串
        nlp: 已加载的 spaCy 模型
        
    Returns:
        list of dict: 每个词的类型标注信息，包含 word, pos, group, display_name 字段
    """
    doc = nlp(text)
    classified = []
    for token in doc:
        if token.pos_ == "SPACE":
            continue
        group = get_word_type_group(token.pos_)
        classified.append({
            'word': token.text,
            'pos': token.pos_,
            'group': group,
            'display_name': WORD_TYPE_DISPLAY_NAMES.get(group, '未知'),
        })
    return classified


def build_word_type_visualization(text, nlp, use_color=True, group_by_type=True):
    """生成带词汇类型标注的可视化文本，用于验证分组正确性。
    
    支持两种输出模式：
    - group_by_type=True: 按词汇类型分组输出，每个类型列出所有词汇
    - group_by_type=False: 原始文本顺序，每个词用标注包裹
    
    Args:
        text: 输入文本字符串
        nlp: 已加载的 spaCy 模型
        use_color: 是否使用 ANSI 颜色代码（终端输出时为 True）
        group_by_type: 是否按类型分组输出
        
    Returns:
        str: 格式化后的可视化文本
    """
    classified = classify_vocab_types(text, nlp)
    
    if not classified:
        return "[空文本，无词汇可分类]"
    
    lines = []
    lines.append("=" * 70)
    lines.append("  词汇类型分组验证输出 (Vocabulary Type Group Validation)")
    lines.append("=" * 70)
    lines.append(f"  输入文本: \"{text}\"")
    lines.append(f"  总词汇数: {len(classified)}")
    lines.append("=" * 70)
    
    if group_by_type:
        # 按词汇类型分组输出
        grouped = {}
        for item in classified:
            group = item['group']
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(item)
        
        # 定义输出顺序
        group_order = ['entity', 'attribute', 'action', 'modifier', 'function', 'unknown']
        
        for group in group_order:
            if group not in grouped:
                continue
            items = grouped[group]
            display_name = WORD_TYPE_DISPLAY_NAMES.get(group, '未知')
            pos_tags = sorted(set(item['pos'] for item in items))
            words = [item['word'] for item in items]
            
            if use_color:
                color = WORD_TYPE_COLORS.get(group, WORD_TYPE_COLORS['unknown'])
                header = f"{color}[{display_name} | {group} | POS: {', '.join(pos_tags)}]{COLOR_RESET}"
                word_str = ' '.join(words)
                lines.append(f"\n  {header}")
                lines.append(f"  词汇: {word_str}")
            else:
                lines.append(f"\n  [{display_name} | {group} | POS: {', '.join(pos_tags)}]")
                lines.append(f"  词汇: {' '.join(words)}")
            lines.append(f"  数量: {len(items)}")
    else:
        # 按原始文本顺序输出，每个词用类型标注包裹
        lines.append("\n  原始顺序输出（每个词标注其词汇类型）:")
        lines.append("  " + "-" * 60)
        annotated_parts = []
        for item in classified:
            if use_color:
                color = WORD_TYPE_COLORS.get(item['group'], WORD_TYPE_COLORS['unknown'])
                annotated = f"{color}{item['word']}[{item['display_name']}]{COLOR_RESET}"
            else:
                annotated = f"{item['word']}[{item['display_name']}]"
            annotated_parts.append(annotated)
        lines.append("  " + ' '.join(annotated_parts))
    
    lines.append("\n" + "=" * 70)
    
    # 统计摘要
    group_counts = {}
    for item in classified:
        group = item['group']
        group_counts[group] = group_counts.get(group, 0) + 1
    
    lines.append("  统计摘要 (Summary):")
    for group, count in group_counts.items():
        display_name = WORD_TYPE_DISPLAY_NAMES.get(group, '未知')
        percentage = count / len(classified) * 100
        indicator = "█" * max(1, int(percentage / 5))
        if use_color:
            color = WORD_TYPE_COLORS.get(group, WORD_TYPE_COLORS['unknown'])
            lines.append(f"    {color}{display_name:6s} ({group:12s}): {count:3d} ({percentage:5.1f}%) {indicator}{COLOR_RESET}")
        else:
            lines.append(f"    {display_name:6s} ({group:12s}): {count:3d} ({percentage:5.1f}%) {indicator}")
    
    lines.append("=" * 70)
    return '\n'.join(lines)


def build_word_type_visualization_plain(text, nlp):
    """生成纯文本（无颜色代码）的词汇类型可视化，适用于文件输出。
    
    Args:
        text: 输入文本字符串
        nlp: 已加载的 spaCy 模型
        
    Returns:
        str: 纯文本格式的可视化输出
    """
    return build_word_type_visualization(text, nlp, use_color=False, group_by_type=True)


def classify_and_filter_cot_step(cot_step_text, tokenizer, model_id, nlp, word_type_group=None):
    """对CoT推理步骤进行词汇类型分类，并根据指定的类型组过滤词汇。
    
    该函数是 align_cot_to_pos 的增强版，在原有对齐功能基础上增加了
    词汇类型过滤能力。当 word_type_group 为 None 时，行为与原始函数完全一致。
    
    Args:
        cot_step_text: CoT推理步骤文本
        tokenizer: HuggingFace tokenizer
        model_id: 模型ID（用于查找空白字符）
        nlp: spaCy 模型
        word_type_group: 词汇类型分组名称，如 'entity', 'action', 'function' 等。
                        为 None 时返回所有词汇（向后兼容）。
        
    Returns:
        tuple: (token_indices, word_spans)
            - token_indices: token ID 张量
            - word_spans: Word 对象列表（若指定了 word_type_group，则仅包含该组的词汇）
            
    Raises:
        ValueError: 若 word_type_group 无效
    """
    # 验证词汇类型分组
    if word_type_group is not None and not validate_word_type_group(word_type_group):
        valid_groups = get_available_word_type_groups()
        raise ValueError(
            f"无效的词汇类型分组 '{word_type_group}'。"
            f"有效分组: {valid_groups}"
        )
    
    # 执行原始对齐
    indices, word_spans = align_cot_to_pos(cot_step_text, tokenizer, model_id, nlp)
    
    # 如果指定了词汇类型分组，则过滤
    if word_type_group is not None:
        if word_type_group == 'random':
            # 随机对照组：从所有词汇中随机选取等量的内容词数量
            content_words = [w for w in word_spans if w.is_content()]
            n_random = len(content_words)
            if n_random > 0 and len(word_spans) > n_random:
                word_spans = random.sample(word_spans, n_random)
            # 若内容词数量等于总词汇数，则保留全部
        else:
            word_spans = [w for w in word_spans if w.belongs_to_group(word_type_group)]
    
    return indices, word_spans


def generate_validation_report(dataset_cots, tokenizer, model_id, nlp, n_samples=5):
    """为数据集生成词汇类型验证报告，展示各类词汇的分布情况。
    
    Args:
        dataset_cots: CoT 数据集列表，每项需包含 'cot' 或 'segmented_cot' 字段
        tokenizer: HuggingFace tokenizer
        model_id: 模型ID
        nlp: spaCy 模型
        n_samples: 采样的样本数量
        
    Returns:
        str: 验证报告文本
    """
    report_lines = []
    report_lines.append("#" * 70)
    report_lines.append("#  词汇类型分组验证报告 (Vocabulary Type Validation Report)")
    report_lines.append("#" * 70)
    
    # 采样
    samples = dataset_cots[:n_samples] if len(dataset_cots) >= n_samples else dataset_cots
    
    all_group_stats = {}
    
    for idx, sample in enumerate(samples):
        # 获取CoT文本
        if 'segmented_cot' in sample:
            cot_text = ' '.join(sample['segmented_cot'])
        elif 'cot' in sample:
            cot_text = sample['cot']
        else:
            continue
        
        report_lines.append(f"\n## 样本 {idx + 1}")
        visualization = build_word_type_visualization(cot_text, nlp, use_color=False, group_by_type=True)
        report_lines.append(visualization)
        
        # 累积统计
        classified = classify_vocab_types(cot_text, nlp)
        for item in classified:
            group = item['group']
            all_group_stats[group] = all_group_stats.get(group, 0) + 1
    
    # 全局统计
    total = sum(all_group_stats.values())
    report_lines.append(f"\n## 全局统计 ({len(samples)} 个样本, 总计 {total} 个词汇)")
    report_lines.append("-" * 50)
    for group in ['entity', 'attribute', 'action', 'modifier', 'function', 'unknown']:
        count = all_group_stats.get(group, 0)
        pct = count / total * 100 if total > 0 else 0
        display_name = WORD_TYPE_DISPLAY_NAMES.get(group, '未知')
        report_lines.append(f"  {display_name:6s} ({group:12s}): {count:5d} ({pct:5.1f}%)")
    
    report_lines.append("\n" + "#" * 70)
    return '\n'.join(report_lines)


# 模块导出列表
__all__ = [
    'Word',
    'TARGET_TAGS',
    'WORD_TYPE_GROUPS',
    'WORD_TYPE_DISPLAY_NAMES',
    'WORD_TYPE_COLORS',
    'COLOR_RESET',
    'get_word_type_group',
    'get_available_word_type_groups',
    'validate_word_type_group',
    'classify_vocab_types',
    'build_word_type_visualization',
    'build_word_type_visualization_plain',
    'classify_and_filter_cot_step',
    'generate_validation_report',
    'sentencize',
    'pos_tag',
    'words_to_token_spans',
    'align_cot_to_pos',
    'WHITESPACE_CHARS',
]
