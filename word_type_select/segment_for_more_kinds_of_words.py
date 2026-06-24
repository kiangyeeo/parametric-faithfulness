"""Segmentation + POS alignment, extended with vocabulary-type groups.

This is the upstream ``segment.py`` plus a vocabulary-type layer used by the
word-type ablation: each Universal Dependencies POS tag maps to a coarse group
(entity / attribute / action / function / modifier), and the unlearning target
can be restricted to a single group.
"""
import nltk
import spacy
import torch
import random

import numpy as np

from dataclasses import dataclass, field

# POS tags -> coarse vocabulary-type group.
WORD_TYPE_GROUPS = {
    'entity':      {'NOUN', 'PROPN'},                          # nouns / proper nouns (factual content)
    'attribute':   {'ADJ', 'NUM'},                             # adjectives / numbers (quantities, descriptions)
    'action':      {'VERB'},                                   # verbs (relations, processes)
    'function':    {'ADP', 'AUX', 'CCONJ', 'DET',              # function words
                    'PART', 'PRON', 'SCONJ'},
    'modifier':    {'ADV'},                                    # adverbs (degree / manner)
    'all_content': {'NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM',     # all content words (the FUR default)
                    'ADV'},
    'random':      None,                                       # random control group (special-cased)
}

# Original content-word set used by Word.is_content (matches upstream segment.py).
TARGET_TAGS = set([
    'VERB',
    'NUM',
    'ADJ',
    'NOUN',
    'PROPN',
])

# Display names for validation output.
WORD_TYPE_DISPLAY_NAMES = {
    'entity':      'Entity',
    'attribute':   'Attribute',
    'action':      'Action',
    'function':    'Function',
    'modifier':    'Modifier',
    'all_content': 'Content',
    'random':      'Random',
}

# ANSI colors for terminal validation output.
WORD_TYPE_COLORS = {
    'entity':      '\033[94m',   # blue
    'attribute':   '\033[93m',   # yellow
    'action':      '\033[91m',   # red
    'function':    '\033[90m',   # grey
    'modifier':    '\033[95m',   # magenta
    'all_content': '\033[92m',   # green
    'random':      '\033[96m',   # cyan
    'unknown':     '\033[0m',    # default
}
COLOR_RESET = '\033[0m'


def get_word_type_group(pos_tag):
    """Return the vocabulary-type group for a POS tag, or 'unknown'."""
    for group_name, tag_set in WORD_TYPE_GROUPS.items():
        if tag_set is not None and pos_tag in tag_set:
            return group_name
    return 'unknown'


def get_available_word_type_groups():
    """Valid group names, excluding the special-cased 'random'."""
    return [g for g in WORD_TYPE_GROUPS if g != 'random']


def validate_word_type_group(group_name):
    """True if ``group_name`` is a known group."""
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
        return get_word_type_group(self.pos)

    def belongs_to_group(self, group_name):
        """True if this word's POS falls in ``group_name``."""
        if group_name not in WORD_TYPE_GROUPS:
            return False
        tag_set = WORD_TYPE_GROUPS[group_name]
        if tag_set is None:
            return False
        return self.pos in tag_set

    def get_display_name(self):
        return WORD_TYPE_DISPLAY_NAMES.get(self.get_group(), 'Unknown')

WHITESPACE_CHARS = {
    'meta-llama/Meta-Llama-3-8B-Instruct': 'Ġ',
    'microsoft/Phi-3-mini-4k-instruct': '▁',
    'mistralai/Mistral-7B-Instruct-v0.2': '▁',
    'meta-llama/Llama-3.2-3B-Instruct': 'Ġ',
    # local snapshot path
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


# ---- Vocabulary-type validation / visualization helpers (used by test_vocab_types) ----

def classify_vocab_types(text, nlp):
    """Tag every (non-space) token in ``text`` with its vocabulary-type group.

    Returns a list of dicts with word, pos, group, display_name.
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
            'display_name': WORD_TYPE_DISPLAY_NAMES.get(group, 'Unknown'),
        })
    return classified


def build_word_type_visualization(text, nlp, use_color=True, group_by_type=True):
    """Render a vocabulary-type annotated view of ``text`` for inspection.

    ``group_by_type=True`` lists words grouped by type; otherwise annotates the
    text in original order. ``use_color`` toggles ANSI colors.
    """
    classified = classify_vocab_types(text, nlp)

    if not classified:
        return "[empty text, nothing to classify]"

    lines = []
    lines.append("=" * 70)
    lines.append("  Vocabulary Type Group Validation")
    lines.append("=" * 70)
    lines.append(f"  Input: \"{text}\"")
    lines.append(f"  Tokens: {len(classified)}")
    lines.append("=" * 70)

    if group_by_type:
        grouped = {}
        for item in classified:
            group = item['group']
            if group not in grouped:
                grouped[group] = []
            grouped[group].append(item)

        group_order = ['entity', 'attribute', 'action', 'modifier', 'function', 'unknown']

        for group in group_order:
            if group not in grouped:
                continue
            items = grouped[group]
            display_name = WORD_TYPE_DISPLAY_NAMES.get(group, 'Unknown')
            pos_tags = sorted(set(item['pos'] for item in items))
            words = [item['word'] for item in items]

            if use_color:
                color = WORD_TYPE_COLORS.get(group, WORD_TYPE_COLORS['unknown'])
                header = f"{color}[{display_name} | {group} | POS: {', '.join(pos_tags)}]{COLOR_RESET}"
                word_str = ' '.join(words)
                lines.append(f"\n  {header}")
                lines.append(f"  words: {word_str}")
            else:
                lines.append(f"\n  [{display_name} | {group} | POS: {', '.join(pos_tags)}]")
                lines.append(f"  words: {' '.join(words)}")
            lines.append(f"  count: {len(items)}")
    else:
        lines.append("\n  Original order (each word annotated with its type):")
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

    group_counts = {}
    for item in classified:
        group = item['group']
        group_counts[group] = group_counts.get(group, 0) + 1

    lines.append("  Summary:")
    for group, count in group_counts.items():
        display_name = WORD_TYPE_DISPLAY_NAMES.get(group, 'Unknown')
        percentage = count / len(classified) * 100
        indicator = "█" * max(1, int(percentage / 5))
        if use_color:
            color = WORD_TYPE_COLORS.get(group, WORD_TYPE_COLORS['unknown'])
            lines.append(f"    {color}{display_name:8s} ({group:12s}): {count:3d} ({percentage:5.1f}%) {indicator}{COLOR_RESET}")
        else:
            lines.append(f"    {display_name:8s} ({group:12s}): {count:3d} ({percentage:5.1f}%) {indicator}")

    lines.append("=" * 70)
    return '\n'.join(lines)


def build_word_type_visualization_plain(text, nlp):
    """Plain-text (no ANSI) vocabulary-type visualization, for file output."""
    return build_word_type_visualization(text, nlp, use_color=False, group_by_type=True)


def classify_and_filter_cot_step(cot_step_text, tokenizer, model_id, nlp, word_type_group=None):
    """``align_cot_to_pos`` plus optional filtering to a single vocabulary group.

    With ``word_type_group=None`` the behavior matches ``align_cot_to_pos``.
    For 'random' a random subset of size = #content-words is kept; otherwise
    only words in the requested group survive. Raises ValueError on an unknown
    group.
    """
    if word_type_group is not None and not validate_word_type_group(word_type_group):
        valid_groups = get_available_word_type_groups()
        raise ValueError(
            f"Invalid word_type_group '{word_type_group}'. Valid groups: {valid_groups}"
        )

    indices, word_spans = align_cot_to_pos(cot_step_text, tokenizer, model_id, nlp)

    if word_type_group is not None:
        if word_type_group == 'random':
            # Random control: keep a random subset matching the content-word count.
            content_words = [w for w in word_spans if w.is_content()]
            n_random = len(content_words)
            if n_random > 0 and len(word_spans) > n_random:
                word_spans = random.sample(word_spans, n_random)
        else:
            word_spans = [w for w in word_spans if w.belongs_to_group(word_type_group)]

    return indices, word_spans


def generate_validation_report(dataset_cots, tokenizer, model_id, nlp, n_samples=5):
    """Build a vocabulary-type distribution report over the first ``n_samples`` CoTs."""
    report_lines = []
    report_lines.append("#" * 70)
    report_lines.append("#  Vocabulary Type Validation Report")
    report_lines.append("#" * 70)

    samples = dataset_cots[:n_samples] if len(dataset_cots) >= n_samples else dataset_cots

    all_group_stats = {}

    for idx, sample in enumerate(samples):
        if 'segmented_cot' in sample:
            cot_text = ' '.join(sample['segmented_cot'])
        elif 'cot' in sample:
            cot_text = sample['cot']
        else:
            continue

        report_lines.append(f"\n## Sample {idx + 1}")
        visualization = build_word_type_visualization(cot_text, nlp, use_color=False, group_by_type=True)
        report_lines.append(visualization)

        classified = classify_vocab_types(cot_text, nlp)
        for item in classified:
            group = item['group']
            all_group_stats[group] = all_group_stats.get(group, 0) + 1

    total = sum(all_group_stats.values())
    report_lines.append(f"\n## Global ({len(samples)} samples, {total} tokens)")
    report_lines.append("-" * 50)
    for group in ['entity', 'attribute', 'action', 'modifier', 'function', 'unknown']:
        count = all_group_stats.get(group, 0)
        pct = count / total * 100 if total > 0 else 0
        display_name = WORD_TYPE_DISPLAY_NAMES.get(group, 'Unknown')
        report_lines.append(f"  {display_name:8s} ({group:12s}): {count:5d} ({pct:5.1f}%)")

    report_lines.append("\n" + "#" * 70)
    return '\n'.join(report_lines)


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
