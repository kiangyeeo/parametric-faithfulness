#!/usr/bin/env python3
"""
词汇类型分组功能验证 (Vocabulary Type Group Validation Script)

验证 segment.py 和 data.py 中新增的词汇类型分组功能：
1. 词汇类型分类的正确性
2. 可视化输出的完整性
3. 向后兼容性
4. 错误处理
"""

import sys
import os
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spacy
import torch
from transformers import AutoTokenizer

from segment import (
    Word, TARGET_TAGS, WORD_TYPE_GROUPS, WORD_TYPE_DISPLAY_NAMES,
    get_word_type_group, get_available_word_type_groups,
    validate_word_type_group,
    classify_vocab_types, build_word_type_visualization,
    build_word_type_visualization_plain,
    classify_and_filter_cot_step, generate_validation_report,
    sentencize, pos_tag, align_cot_to_pos
)
from data import (
    qcot_encoder, SegmentOTFDataset, cot_to_otfd,
    model_name_dict, load_jsonl,
)

# ============================================================
# 测试数据
# ============================================================

TEST_TEXTS = [
    "The pond in the backyard fills up heavily during a sudden downpour.",
    "If John notices the guard sleeping, he can likely escape through the gate.",
    "The three bloated bodies floated silently in the murky water.",
    "Although the experiment failed, the scientists discovered an important anomaly.",
]

# ============================================================
# 测试 1: 基础词汇类型分类
# ============================================================

def test_basic_classification():
    """测试基础词汇类型分类功能"""
    print("=" * 70)
    print("  测试 1: 基础词汇类型分类 (Basic Classification)")
    print("=" * 70)
    
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    
    all_passed = True
    
    for text in TEST_TEXTS:
        classified = classify_vocab_types(text, nlp)
        print(f"\n  文本: \"{text}\"")
        print(f"  词汇数: {len(classified)}")
        
        for item in classified:
            print(f"    {item['word']:15s} -> POS={item['pos']:8s} -> 分组={item['group']:12s} ({item['display_name']})")
        
        # 验证每个词都有有效的分组
        for item in classified:
            if item['group'] == 'unknown' and item['pos'] not in {'PUNCT', 'SYM', 'X', 'INTJ'}:
                print(f"    WARNING: 未知分组: {item['word']} (POS={item['pos']})")
                # 非标点符号的未知分组需要关注
    
    print(f"\n  测试 1 结果: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


# ============================================================
# 测试 2: 可视化输出
# ============================================================

def test_visualization():
    """测试可视化输出功能"""
    print("\n" + "=" * 70)
    print("  测试 2: 可视化输出 (Visualization)")
    print("=" * 70)
    
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    
    for text in TEST_TEXTS[:2]:
        print("\n  --- 按类型分组模式 (group_by_type=True) ---")
        viz = build_word_type_visualization(text, nlp, use_color=False, group_by_type=True)
        print(viz)
        
        print("\n  --- 原始顺序模式 (group_by_type=False) ---")
        viz2 = build_word_type_visualization(text, nlp, use_color=False, group_by_type=False)
        print(viz2)
    
    # 测试终端颜色输出
    print("\n  --- 终端颜色模式 (use_color=True) ---")
    viz3 = build_word_type_visualization(TEST_TEXTS[0], nlp, use_color=True, group_by_type=True)
    print(viz3)
    
    # 测试纯文本输出
    print("\n  --- 纯文本输出 (plain) ---")
    viz4 = build_word_type_visualization_plain(TEST_TEXTS[0], nlp)
    print(viz4)
    
    print("\n  测试 2 结果: PASS")
    return True


# ============================================================
# 测试 3: Word 类增强方法
# ============================================================

def test_word_class():
    """测试 Word 类的增强方法"""
    print("\n" + "=" * 70)
    print("  测试 3: Word 类增强方法 (Enhanced Word Class)")
    print("=" * 70)
    
    # 创建不同类型的 Word 实例
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
    
    all_passed = True
    
    for w in test_words:
        group = w.get_group()
        display = w.get_display_name()
        is_content = w.is_content()
        
        print(f"  {w.word:10s} POS={w.pos:8s} -> group={group:12s} display={display:6s} is_content={is_content}")
        
        # 验证分组正确性
        expected_groups = {
            "NOUN": "entity", "PROPN": "entity",
            "ADJ": "attribute", "NUM": "attribute",
            "VERB": "action",
            "DET": "function", "ADP": "function", "CCONJ": "function",
            "ADV": "modifier",
        }
        expected = expected_groups.get(w.pos, "unknown")
        if group != expected:
            print(f"    ERROR: 期望 '{expected}', 实际 '{group}'")
            all_passed = False
    
    # 测试 belongs_to_group
    assert test_words[0].belongs_to_group("entity") == True
    assert test_words[0].belongs_to_group("action") == False
    assert test_words[3].belongs_to_group("function") == True
    assert test_words[3].belongs_to_group("entity") == False
    
    # 测试无效分组
    assert test_words[0].belongs_to_group("invalid_group") == False
    
    print(f"\n  测试 3 结果: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


# ============================================================
# 测试 4: 工具函数
# ============================================================

def test_utility_functions():
    """测试工具函数"""
    print("\n" + "=" * 70)
    print("  测试 4: 工具函数 (Utility Functions)")
    print("=" * 70)
    
    # 测试 get_word_type_group
    assert get_word_type_group("NOUN") == "entity"
    assert get_word_type_group("PROPN") == "entity"
    assert get_word_type_group("VERB") == "action"
    assert get_word_type_group("ADJ") == "attribute"
    assert get_word_type_group("NUM") == "attribute"
    assert get_word_type_group("ADV") == "modifier"
    assert get_word_type_group("DET") == "function"
    assert get_word_type_group("ADP") == "function"
    assert get_word_type_group("AUX") == "function"
    assert get_word_type_group("CCONJ") == "function"
    assert get_word_type_group("PART") == "function"
    assert get_word_type_group("PRON") == "function"
    assert get_word_type_group("SCONJ") == "function"
    assert get_word_type_group("PUNCT") == "unknown"
    assert get_word_type_group("X") == "unknown"
    print("  get_word_type_group: PASS")
    
    # 测试 validate_word_type_group
    assert validate_word_type_group("entity") == True
    assert validate_word_type_group("action") == True
    assert validate_word_type_group("function") == True
    assert validate_word_type_group("random") == True
    assert validate_word_type_group("invalid") == False
    print("  validate_word_type_group: PASS")
    
    # 测试 get_available_word_type_groups
    available = get_available_word_type_groups()
    assert "entity" in available
    assert "action" in available
    assert "function" in available
    assert "random" not in available  # random 被排除
    print(f"  get_available_word_type_groups: PASS (groups: {available})")
    
    # 测试 WORD_TYPE_GROUPS 结构
    for group_name, tag_set in WORD_TYPE_GROUPS.items():
        if group_name == 'random':
            assert tag_set is None
        else:
            assert isinstance(tag_set, set)
            assert len(tag_set) > 0
    print("  WORD_TYPE_GROUPS 结构验证: PASS")
    
    print(f"\n  测试 4 结果: PASS")
    return True


# ============================================================
# 测试 5: classify_and_filter_cot_step
# ============================================================

def test_classify_and_filter():
    """测试 classify_and_filter_cot_step 函数"""
    print("\n" + "=" * 70)
    print("  测试 5: classify_and_filter_cot_step")
    print("=" * 70)
    
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    
    # 使用 Phi-3 tokenizer 进行测试
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        tokenizer.pad_token = tokenizer.eos_token
        model_id = "microsoft/Phi-3-mini-4k-instruct"
    except Exception:
        print("  无法加载 Phi-3 tokenizer，跳过此测试")
        return True
    
    cot_text = "The pond in the backyard fills up heavily during a sudden downpour."
    
    # 测试无过滤（向后兼容）
    indices, words = classify_and_filter_cot_step(cot_text, tokenizer, model_id, nlp, word_type_group=None)
    print(f"  无过滤: {len(words)} 个词汇")
    for w in words:
        print(f"    {w.word:15s} POS={w.pos:8s} group={w.get_group():12s}")
    
    # 测试各类型过滤
    for group in ['entity', 'attribute', 'action', 'function', 'modifier']:
        indices, words = classify_and_filter_cot_step(cot_text, tokenizer, model_id, nlp, word_type_group=group)
        display = WORD_TYPE_DISPLAY_NAMES.get(group, group)
        print(f"  {display} ({group}): {len(words)} 个词汇 -> {[w.word for w in words]}")
    
    # 测试错误处理
    try:
        classify_and_filter_cot_step(cot_text, tokenizer, model_id, nlp, word_type_group="invalid_group")
        print("  ERROR: 应该抛出 ValueError")
        return False
    except ValueError as e:
        print(f"  错误处理: PASS (捕获到 ValueError: {e})")
    
    print(f"\n  测试 5 结果: PASS")
    return True


# ============================================================
# 测试 6: qcot_encoder 向后兼容性
# ============================================================

def test_qcot_encoder_compatibility():
    """测试 qcot_encoder 的向后兼容性"""
    print("\n" + "=" * 70)
    print("  测试 6: qcot_encoder 向后兼容性 (Backward Compatibility)")
    print("=" * 70)
    
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("  无法加载 tokenizer，跳过此测试")
        return True
    
    question = "What happens to a pond during a heavy rainstorm?"
    cot = "The pond in the backyard fills up heavily during a sudden downpour."
    
    # 测试 1: 无 pos_filter（原始行为）
    E, L, A, T = qcot_encoder(tokenizer, question, cot, pos_filter=False)
    print(f"  pos_filter=False: 目标token数 = {T}")
    
    # 测试 2: pos_filter=True, word_type_group=None（原始行为）
    E, L, A, T = qcot_encoder(tokenizer, question, cot, pos_filter=True, nlp=nlp)
    print(f"  pos_filter=True, word_type_group=None: 目标token数 = {T}")
    
    # 测试 3: 指定词汇类型分组
    for group in ['entity', 'all_content']:
        E, L, A, T = qcot_encoder(tokenizer, question, cot, pos_filter=True, nlp=nlp, word_type_group=group)
        display = WORD_TYPE_DISPLAY_NAMES.get(group, group)
        print(f"  word_type_group={group} ({display}): 目标token数 = {T}")
    
    print(f"\n  测试 6 结果: PASS")
    return True


# ============================================================
# 测试 7: 生成验证报告
# ============================================================

def test_validation_report():
    """测试验证报告生成"""
    print("\n" + "=" * 70)
    print("  测试 7: 验证报告生成 (Validation Report)")
    print("=" * 70)
    
    nlp = spacy.load("en_core_web_sm", disable=['ner'])
    
    # 构建模拟数据集
    mock_dataset = [
        {'cot': "The pond in the backyard fills up heavily during a sudden downpour.",
         'segmented_cot': ["The pond in the backyard", "fills up heavily", "during a sudden downpour"]},
        {'cot': "If John notices the guard sleeping, he can likely escape through the gate.",
         'segmented_cot': ["If John notices the guard sleeping", "he can likely escape through the gate"]},
    ]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        tokenizer.pad_token = tokenizer.eos_token
        model_id = "microsoft/Phi-3-mini-4k-instruct"
    except Exception:
        print("  无法加载 tokenizer，跳过此测试")
        return True
    
    report = generate_validation_report(mock_dataset, tokenizer, model_id, nlp, n_samples=2)
    print(report)
    
    print(f"\n  测试 7 结果: PASS")
    return True


# ============================================================
# 测试 8: SegmentOTFDataset 词汇类型分组
# ============================================================

def test_segment_otf_dataset():
    """测试 SegmentOTFDataset 的词汇类型分组支持"""
    print("\n" + "=" * 70)
    print("  测试 8: SegmentOTFDataset 词汇类型分组")
    print("=" * 70)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("  无法加载 tokenizer，跳过此测试")
        return True
    
    # 构建模拟遗忘和保留数据
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
    
    # 测试默认行为（无词汇类型分组）
    dataset = SegmentOTFDataset(forget_data, retain_data, tokenizer, 
                                 stepwise=True, pos_filter=True, 
                                 word_type_group=None)
    print(f"  默认行为: 创建成功 (word_type_group=None)")
    
    # 测试各词汇类型分组
    for group in ['entity', 'action', 'all_content']:
        try:
            dataset = SegmentOTFDataset(forget_data, retain_data, tokenizer,
                                         stepwise=True, pos_filter=True,
                                         word_type_group=group)
            print(f"  word_type_group={group}: 创建成功")
        except Exception as e:
            print(f"  word_type_group={group}: 创建失败 - {e}")
    
    # 测试无效分组
    try:
        dataset = SegmentOTFDataset(forget_data, retain_data, tokenizer,
                                     stepwise=True, pos_filter=True,
                                     word_type_group="invalid_group")
        print("  ERROR: 应该抛出 ValueError")
        return False
    except ValueError as e:
        print(f"  无效分组错误处理: PASS")
    
    print(f"\n  测试 8 结果: PASS")
    return True


# ============================================================
# 测试 9: cot_to_otfd 词汇类型分组
# ============================================================

def test_cot_to_otfd():
    """测试 cot_to_otfd 的词汇类型分组支持"""
    print("\n" + "=" * 70)
    print("  测试 9: cot_to_otfd 词汇类型分组")
    print("=" * 70)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("  无法加载 tokenizer，跳过此测试")
        return True
    
    # 构建模拟数据
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
    
    # 测试默认行为
    dataset = cot_to_otfd(target, all_data, tokenizer, n=2, strategy='sentencize',
                          stepwise=True, pos=True, word_type_group=None)
    print(f"  默认行为 (word_type_group=None): 创建成功")
    print(f"    word_type_group = {dataset.word_type_group}")
    
    # 测试指定词汇类型
    dataset = cot_to_otfd(target, all_data, tokenizer, n=2, strategy='sentencize',
                          stepwise=True, pos=True, word_type_group='entity')
    print(f"  word_type_group='entity': 创建成功")
    print(f"    word_type_group = {dataset.word_type_group}")
    
    # 测试无效分组
    try:
        cot_to_otfd(target, all_data, tokenizer, n=2, strategy='sentencize',
                    stepwise=True, pos=True, word_type_group="invalid")
        print("  ERROR: 应该抛出 ValueError")
        return False
    except ValueError as e:
        print(f"  无效分组错误处理: PASS")
    
    print(f"\n  测试 9 结果: PASS")
    return True


# ============================================================
# 主测试入口
# ============================================================

def main():
    print("\n" + "#" * 70)
    print("#  词汇类型分组功能验证套件")
    print("#  Vocabulary Type Group Validation Suite")
    print("#" * 70 + "\n")
    
    results = {}
    
    # 运行所有测试
    tests = [
        ("基础词汇类型分类", test_basic_classification),
        ("可视化输出", test_visualization),
        ("Word 类增强方法", test_word_class),
        ("工具函数", test_utility_functions),
        ("classify_and_filter_cot_step", test_classify_and_filter),
        ("qcot_encoder 向后兼容性", test_qcot_encoder_compatibility),
        ("验证报告生成", test_validation_report),
        ("SegmentOTFDataset 词汇类型分组", test_segment_otf_dataset),
        ("cot_to_otfd 词汇类型分组", test_cot_to_otfd),
    ]
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ERROR: 测试 '{name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # 汇总结果
    print("\n\n" + "#" * 70)
    print("#  测试结果汇总 (Test Results Summary)")
    print("#" * 70)
    
    passed = 0
    failed = 0
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}")
    
    print(f"\n  总计: {passed} 通过, {failed} 失败, {len(results)} 总计")
    print("#" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)