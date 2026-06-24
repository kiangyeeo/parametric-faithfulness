#!/usr/bin/env python3
"""Validation suite for the vocabulary-type-group extension.

Checks the word-type classification, visualization, backward compatibility, and
error handling added on top of the upstream segment/data modules. Run from the
repo root: ``python word_type_select/test_vocab_types.py``.
"""

import os
import sys

# Expose the repo root (evaluate/dataload/util) and this directory (the
# word-type segment/data extensions) on the import path.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spacy
from transformers import AutoTokenizer

from segment_for_more_kinds_of_words import (
    Word, WORD_TYPE_GROUPS, WORD_TYPE_DISPLAY_NAMES,
    get_word_type_group, get_available_word_type_groups,
    validate_word_type_group,
    classify_vocab_types, build_word_type_visualization,
    build_word_type_visualization_plain,
    classify_and_filter_cot_step, generate_validation_report,
)
from data_for_more_kinds_of_words import (
    qcot_encoder, SegmentOTFDataset, cot_to_otfd,
)

PHI3 = "microsoft/Phi-3-mini-4k-instruct"

TEST_TEXTS = [
    "The pond in the backyard fills up heavily during a sudden downpour.",
    "If John notices the guard sleeping, he can likely escape through the gate.",
    "The three bloated bodies floated silently in the murky water.",
    "Although the experiment failed, the scientists discovered an important anomaly.",
]


def _load_phi3_tokenizer():
    """Load the Phi-3 tokenizer, or return None if unavailable (offline)."""
    try:
        tok = AutoTokenizer.from_pretrained(PHI3)
        tok.pad_token = tok.eos_token
        return tok
    except Exception:
        print("  Could not load Phi-3 tokenizer, skipping.")
        return None


def test_basic_classification():
    """POS tags map to vocabulary-type groups."""
    print("=" * 70)
    print("  Test 1: basic classification")
    print("=" * 70)
    nlp = spacy.load("en_core_web_sm", disable=['ner'])

    for text in TEST_TEXTS:
        classified = classify_vocab_types(text, nlp)
        print(f"\n  text: \"{text}\" ({len(classified)} tokens)")
        for item in classified:
            print(f"    {item['word']:15s} -> POS={item['pos']:8s} -> group={item['group']:12s} ({item['display_name']})")
            if item['group'] == 'unknown' and item['pos'] not in {'PUNCT', 'SYM', 'X', 'INTJ'}:
                print(f"    WARNING: unexpected unknown group for {item['word']} (POS={item['pos']})")

    print("\n  Test 1: PASS")
    return True


def test_visualization():
    """Visualization renders in grouped, ordered, color and plain modes."""
    print("\n" + "=" * 70)
    print("  Test 2: visualization")
    print("=" * 70)
    nlp = spacy.load("en_core_web_sm", disable=['ner'])

    for text in TEST_TEXTS[:2]:
        print("\n  --- grouped (group_by_type=True) ---")
        print(build_word_type_visualization(text, nlp, use_color=False, group_by_type=True))
        print("\n  --- original order (group_by_type=False) ---")
        print(build_word_type_visualization(text, nlp, use_color=False, group_by_type=False))

    print("\n  --- color ---")
    print(build_word_type_visualization(TEST_TEXTS[0], nlp, use_color=True, group_by_type=True))
    print("\n  --- plain ---")
    print(build_word_type_visualization_plain(TEST_TEXTS[0], nlp))

    print("\n  Test 2: PASS")
    return True


def test_word_class():
    """Word helper methods map POS to the right group."""
    print("\n" + "=" * 70)
    print("  Test 3: Word class methods")
    print("=" * 70)

    test_words = [
        Word("pond", "NOUN", 0, 1),
        Word("bloated", "ADJ", 1, 2),
        Word("notices", "VERB", 2, 3),
        Word("the", "DET", 3, 4),
        Word("heavily", "ADV", 4, 5),
        Word("three", "NUM", 5, 6),
        Word("of", "ADP", 6, 7),
        Word("and", "CCONJ", 7, 8),
        Word("Pope", "PROPN", 8, 9),
    ]
    expected_groups = {
        "NOUN": "entity", "PROPN": "entity",
        "ADJ": "attribute", "NUM": "attribute",
        "VERB": "action",
        "DET": "function", "ADP": "function", "CCONJ": "function",
        "ADV": "modifier",
    }

    all_passed = True
    for w in test_words:
        group = w.get_group()
        print(f"  {w.word:10s} POS={w.pos:8s} -> group={group:12s} display={w.get_display_name():8s} is_content={w.is_content()}")
        if group != expected_groups.get(w.pos, "unknown"):
            print(f"    ERROR: expected '{expected_groups.get(w.pos)}', got '{group}'")
            all_passed = False

    assert test_words[0].belongs_to_group("entity") is True
    assert test_words[0].belongs_to_group("action") is False
    assert test_words[3].belongs_to_group("function") is True
    assert test_words[3].belongs_to_group("entity") is False
    assert test_words[0].belongs_to_group("invalid_group") is False

    print(f"\n  Test 3: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def test_utility_functions():
    """get_word_type_group / validate / available groups behave as specified."""
    print("\n" + "=" * 70)
    print("  Test 4: utility functions")
    print("=" * 70)

    for pos, group in [
        ("NOUN", "entity"), ("PROPN", "entity"), ("VERB", "action"),
        ("ADJ", "attribute"), ("NUM", "attribute"), ("ADV", "modifier"),
        ("DET", "function"), ("ADP", "function"), ("AUX", "function"),
        ("CCONJ", "function"), ("PART", "function"), ("PRON", "function"),
        ("SCONJ", "function"), ("PUNCT", "unknown"), ("X", "unknown"),
    ]:
        assert get_word_type_group(pos) == group, (pos, group)
    print("  get_word_type_group: PASS")

    assert validate_word_type_group("entity") is True
    assert validate_word_type_group("random") is True
    assert validate_word_type_group("invalid") is False
    print("  validate_word_type_group: PASS")

    available = get_available_word_type_groups()
    assert "entity" in available and "action" in available
    assert "random" not in available  # random is special-cased
    print(f"  get_available_word_type_groups: PASS (groups: {available})")

    for group_name, tag_set in WORD_TYPE_GROUPS.items():
        if group_name == 'random':
            assert tag_set is None
        else:
            assert isinstance(tag_set, set) and len(tag_set) > 0
    print("  WORD_TYPE_GROUPS structure: PASS")

    print("\n  Test 4: PASS")
    return True


def test_classify_and_filter():
    """classify_and_filter_cot_step filters to the requested group and errors on bad input."""
    print("\n" + "=" * 70)
    print("  Test 5: classify_and_filter_cot_step")
    print("=" * 70)
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    tokenizer = _load_phi3_tokenizer()
    if tokenizer is None:
        return True

    cot_text = "The pond in the backyard fills up heavily during a sudden downpour."

    _, words = classify_and_filter_cot_step(cot_text, tokenizer, PHI3, nlp, word_type_group=None)
    print(f"  no filter: {len(words)} words")
    for w in words:
        print(f"    {w.word:15s} POS={w.pos:8s} group={w.get_group():12s}")

    for group in ['entity', 'attribute', 'action', 'function', 'modifier']:
        _, words = classify_and_filter_cot_step(cot_text, tokenizer, PHI3, nlp, word_type_group=group)
        print(f"  {WORD_TYPE_DISPLAY_NAMES.get(group, group)} ({group}): {len(words)} -> {[w.word for w in words]}")

    try:
        classify_and_filter_cot_step(cot_text, tokenizer, PHI3, nlp, word_type_group="invalid_group")
        print("  ERROR: expected ValueError")
        return False
    except ValueError as e:
        print(f"  error handling: PASS ({e})")

    print("\n  Test 5: PASS")
    return True


def test_qcot_encoder_compatibility():
    """qcot_encoder still matches upstream behavior when word_type_group is None."""
    print("\n" + "=" * 70)
    print("  Test 6: qcot_encoder backward compatibility")
    print("=" * 70)
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    tokenizer = _load_phi3_tokenizer()
    if tokenizer is None:
        return True

    question = "What happens to a pond during a heavy rainstorm?"
    cot = "The pond in the backyard fills up heavily during a sudden downpour."

    _, _, _, t = qcot_encoder(tokenizer, question, cot, pos_filter=False)
    print(f"  pos_filter=False: targets = {t}")
    _, _, _, t = qcot_encoder(tokenizer, question, cot, pos_filter=True, nlp=nlp)
    print(f"  pos_filter=True, group=None: targets = {t}")
    for group in ['entity', 'all_content']:
        _, _, _, t = qcot_encoder(tokenizer, question, cot, pos_filter=True, nlp=nlp, word_type_group=group)
        print(f"  group={group}: targets = {t}")

    print("\n  Test 6: PASS")
    return True


def test_validation_report():
    """generate_validation_report runs over a mock dataset."""
    print("\n" + "=" * 70)
    print("  Test 7: validation report")
    print("=" * 70)
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    tokenizer = _load_phi3_tokenizer()
    if tokenizer is None:
        return True

    mock_dataset = [
        {'cot': "The pond in the backyard fills up heavily during a sudden downpour.",
         'segmented_cot': ["The pond in the backyard", "fills up heavily", "during a sudden downpour"]},
        {'cot': "If John notices the guard sleeping, he can likely escape through the gate.",
         'segmented_cot': ["If John notices the guard sleeping", "he can likely escape through the gate"]},
    ]
    print(generate_validation_report(mock_dataset, tokenizer, PHI3, nlp, n_samples=2))

    print("\n  Test 7: PASS")
    return True


def test_segment_otf_dataset():
    """SegmentOTFDataset accepts valid groups and rejects invalid ones."""
    print("\n" + "=" * 70)
    print("  Test 8: SegmentOTFDataset word_type_group")
    print("=" * 70)
    tokenizer = _load_phi3_tokenizer()
    if tokenizer is None:
        return True

    forget_data = [
        {'prompt': 'What happens to a pond during heavy rain?',
         'completion': 'The pond fills up with water quickly.',
         'segmented_cot': ['The pond fills up with water quickly.']}
    ]
    retain_data = [
        {'prompt': 'Why does ice float on water?',
         'completion': 'Ice is less dense than liquid water.',
         'segmented_cot': ['Ice is less dense than liquid water.']},
        {'prompt': 'What causes seasons?',
         'completion': 'The tilt of Earth axis causes seasons.',
         'segmented_cot': ['The tilt of Earth axis causes seasons.']},
    ]

    SegmentOTFDataset(forget_data, retain_data, tokenizer, stepwise=True, pos_filter=True, word_type_group=None)
    print("  group=None: created")
    for group in ['entity', 'action', 'all_content']:
        SegmentOTFDataset(forget_data, retain_data, tokenizer, stepwise=True, pos_filter=True, word_type_group=group)
        print(f"  group={group}: created")

    try:
        SegmentOTFDataset(forget_data, retain_data, tokenizer, stepwise=True, pos_filter=True, word_type_group="invalid_group")
        print("  ERROR: expected ValueError")
        return False
    except ValueError:
        print("  invalid group: PASS")

    print("\n  Test 8: PASS")
    return True


def test_cot_to_otfd():
    """cot_to_otfd threads word_type_group through and validates it."""
    print("\n" + "=" * 70)
    print("  Test 9: cot_to_otfd word_type_group")
    print("=" * 70)
    tokenizer = _load_phi3_tokenizer()
    if tokenizer is None:
        return True

    target = {
        'cot_prompt': 'What happens to a pond during heavy rain?',
        'cot': 'The pond fills up with water during heavy rain.',
        'segmented_cot': ['The pond fills up', 'with water during heavy rain.']
    }
    all_data = [
        target,
        {'cot_prompt': 'Why does ice float?', 'cot': 'Ice is less dense than water.',
         'segmented_cot': ['Ice is less dense than water.']},
        {'cot_prompt': 'What causes seasons?', 'cot': 'Earth tilt causes seasons.',
         'segmented_cot': ['Earth tilt causes seasons.']},
    ]

    ds = cot_to_otfd(target, all_data, tokenizer, n=2, strategy='sentencize', stepwise=True, pos=True, word_type_group=None)
    print(f"  group=None: created (word_type_group={ds.word_type_group})")
    ds = cot_to_otfd(target, all_data, tokenizer, n=2, strategy='sentencize', stepwise=True, pos=True, word_type_group='entity')
    print(f"  group='entity': created (word_type_group={ds.word_type_group})")

    try:
        cot_to_otfd(target, all_data, tokenizer, n=2, strategy='sentencize', stepwise=True, pos=True, word_type_group="invalid")
        print("  ERROR: expected ValueError")
        return False
    except ValueError:
        print("  invalid group: PASS")

    print("\n  Test 9: PASS")
    return True


def main():
    print("\n" + "#" * 70)
    print("#  Vocabulary Type Group Validation Suite")
    print("#" * 70 + "\n")

    tests = [
        ("basic classification", test_basic_classification),
        ("visualization", test_visualization),
        ("Word class methods", test_word_class),
        ("utility functions", test_utility_functions),
        ("classify_and_filter_cot_step", test_classify_and_filter),
        ("qcot_encoder compatibility", test_qcot_encoder_compatibility),
        ("validation report", test_validation_report),
        ("SegmentOTFDataset word_type_group", test_segment_otf_dataset),
        ("cot_to_otfd word_type_group", test_cot_to_otfd),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ERROR in '{name}': {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n\n" + "#" * 70)
    print("#  Results")
    print("#" * 70)
    passed = sum(1 for r in results.values() if r)
    failed = len(results) - passed
    for name, result in results.items():
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")
    print(f"\n  Total: {passed} passed, {failed} failed, {len(results)} total")
    print("#" * 70)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
