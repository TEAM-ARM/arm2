# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.
"""
import re
import hydra
import os
import json
from datetime import datetime
from verl.utils.fs import copy_local_path_from_hdfs

import pandas as pd
import numpy as np
from verl.utils.reward_score import gsm8k, csqa, aime, mme_world, mmk12, geo3k, mmmu
from verl.utils.reward_score.code_exec_utils import rebuild_solution_str
from math_verify import parse, verify, LatexExtractionConfig
from collections import Counter, defaultdict
from tqdm import tqdm
import transformers
import random
import warnings
import logging

# 忽略所有警告
warnings.filterwarnings("ignore")

# 配置日志级别来忽略 pylatexenc 的警告
logging.getLogger("pylatexenc.latex2text").setLevel(logging.ERROR)
logging.getLogger("pylatexenc.latexwalker").setLevel(logging.ERROR)

token_dict = {
    "direct": "<ANSWER>\n",
    "long_cot": "<LONG_COT>\n",
    "cot": "<COT>\n",
    "code": "<CODE>\n",
}

def find_instruct(_file):
    if 'code' in _file:
        return "<CODE>\n"
    elif 'long_cot' in _file:
        return "<LONG_COT>\n"
    elif 'cot' in _file:
        return "<COT>\n"
    elif 'direct' in _file:
        return "<ANSWER>\n"
    else:
        return ""

def extract_solution(solution_str):

    pattern = r"<ANSWER>(.*)</ANSWER>"
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[1]
    solution = re.search(pattern, solution_str, re.DOTALL)
    if solution is None:
        pattern = r"\\boxed{(.*?)}"
        solution = re.search(pattern, solution_str, re.DOTALL)
        if solution is None:
            pattern = r"<\|begin_of_box\|>(.*)<\|end_of_box\|>"
            solution = re.search(pattern, solution_str, re.DOTALL)
            if solution is None:
                final_answer = None
            else:
                final_answer = solution.group(1).strip()
        else:
            final_answer = solution.group(1).strip()
    else:
        final_answer = solution.group(1).strip()
    
    # 处理 \text{(C)} 格式，提取括号内的字母
    if final_answer:
        text_pattern = r"\\text\{\(([A-Z])\)"
        text_match = re.search(text_pattern, final_answer)
        if text_match:
            final_answer = text_match.group(1)
        else:
            text_pattern = r"\\text\{\(([A-Z])\)\}"
            text_match = re.search(text_pattern, final_answer)
            if text_match:
                final_answer = text_match.group(1)

    
    return final_answer


def get_type(solution_str):
    if "<COT>" in solution_str:
        if len(solution_str.lower().split(" ")) < 20:
            return "DIRECT"
        return "COT"
    elif "<CODE>" in solution_str:
        return "CODE"
    elif "<LONG_COT>" in solution_str:
        return "LONG_COT"
    else:
        return "DIRECT"


def select_reward_fn(data_source):
    if data_source in ['gsm8k']:
        return gsm8k.compute_score_test
    elif data_source in ['CSQA','GPQA-Diamond','obqa']:
        return csqa.compute_score_test
    elif data_source in ['AIME']:
        return aime.compute_score_test
    elif data_source == 'hiyouga/geometry3k':
        return geo3k.compute_score_test
    elif data_source in ['MME_RealWorld','BLINK']:
        return mme_world.compute_score_test
    elif data_source in ['FanqingM/MMK12','math500','ChartQA']:
        return mmk12.compute_score_test
    elif data_source in ['MMMU']:
        return mmmu.compute_score_test
    else:
        raise NotImplementedError


def find_most_common(answer_span_list, data_source):
    if data_source in ['gsm8k', 'AIME', 'hiyouga/geometry3k','FanqingM/MMK12']:
        parsed_list = [parse(answer_span) for answer_span in answer_span_list]
        ret_idx = 0
        max_common_num = 0
        for i, parsed_answer_span in enumerate(parsed_list):
            common_num = 0
            for j, parsed_answer_span_2 in enumerate(parsed_list):
                if i != j and verify(parsed_answer_span, parsed_answer_span_2):
                    common_num += 1
            if common_num > max_common_num:
                max_common_num = common_num
                ret_idx = i
        return answer_span_list[ret_idx]
    elif data_source in ['csqa', 'MME_RealWorld']:
        answer_span_list = [answer_span.replace(r'(', '').replace(r')', '') for answer_span in answer_span_list]
        most_common_answer = Counter(answer_span_list).most_common(1)[0][0]
        return most_common_answer
    else:
        most_common_answer = Counter(answer_span_list).most_common(1)[0][0]
        return most_common_answer


def calculate_avg_at_k(dataset, config, tokenizer, file_name, k_samples=5, dataset_name=""):
    """
    独立计算avg@k的指标：平均准确率、token数量和格式分布
    Args:
        dataset: 包含prompts, responses, data_sources, reward_model_data的DataFrame
        config: 配置对象
        tokenizer: 分词器
        file_name: 文件名，用于判断前缀类型
        k_samples: 采样数量，默认5
        dataset_name: 数据集名称，用于显示结果
    """
    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    
    total = len(dataset)
    total_passes = 0
    total_token_count = 0
    total_responses = 0
    type_format_dict = defaultdict(list)
    # 新增：按format统计准确率（基于逐条response的判定）
    format_correct_counts = defaultdict(int)
    format_total_counts = defaultdict(int)
    
    # 新增：保存详细的token数据
    token_data_list = []
    
    print(f"计算avg@{k_samples}指标...")
    
    for i in tqdm(range(total), desc="avg@k计算"):
        response_lst_full = responses.iloc[i]
        # print(dataset.iloc[i])
        correctness_lst_full = dataset.iloc[i]["answer_correctness"]
        raw_response_lst_full = []
        raw_correctness_lst_full = []
        
        # 为每个响应添加前缀，同时保留对应的correctness
        for idx, r in enumerate(response_lst_full):
            # 根据文件名判断前缀类型
            prefix = find_instruct(file_name)
            r = prefix + r
            raw_response_lst_full.append(r)
            raw_correctness_lst_full.append(correctness_lst_full[idx])
        
        # 确保有足够的响应进行采样，如果不够就使用所有可用的响应
        if len(raw_response_lst_full) < k_samples:
            # 如果响应数量不足，使用所有可用的响应
            response_lst = raw_response_lst_full
            correctness_lst = raw_correctness_lst_full
        else:
            # 随机采样k_samples条响应和对应的correctness
            sampled_indices = random.sample(range(len(raw_response_lst_full)), k_samples)
            response_lst = [raw_response_lst_full[idx] for idx in sampled_indices]
            correctness_lst = [raw_correctness_lst_full[idx] for idx in sampled_indices]
        
        data_source = data_sources.iloc[i]
        prompt = prompts.iloc[i]
        reward_data = reward_model_data.iloc[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        if data_source == 'ChartQA':
            ground_truth = ground_truth[0]
        sample_passes = 0
        sample_token_count = 0
        
        # 新增：保存当前样本的token数据
        sample_token_data = {
            'responses': []
        }
        
        for idx, r in enumerate(response_lst):
            # print(r)
            # exit(0)
            _type = get_type(r)
            # print(config.data.code_executable)
            # print(_type)
            # exit(0)
            if _type == 'CODE' and config.data.code_executable:
                # print(r)
                # exit(0)
                r, code_executed = rebuild_solution_str(r)
                if code_executed:
                    _type = 'CODE_EXECUTED'
                else:
                    _type = 'CODE_TEXT'
                # print(r)
                # exit(0)
            type_format_dict[_type].append(r)
            total_responses += 1
            
            output = extract_solution(r)
            tokenized_response = tokenizer(r)
            token_count = len(tokenized_response['input_ids'])
            sample_token_count += token_count
            
            if output is None:
                if 'aime' in str(config.data.path):
                    output = '\n'.join(r.strip().split('\n')[-5:])
                else:
                    # 即使output为None，也要保存token数据
                    response_token_data = {
                        'token_count': token_count,
                        'format': _type
                    }
                    sample_token_data['responses'].append(response_token_data)
                    continue
            
            # 计算单个响应的得分
            score = reward_fn(output, ground_truth, method='sc')

            # if score != 1:
            #     print(r)
            #     print(ground_truth)
            #     exit(0)
            
            # 如果reward_fn返回的score不为1，再用correctness判断
            if score != 1:
                correctness_value = correctness_lst[idx].lower()
                # the correctness value is like this: <CORRECTNESS>correct</CORRECTNESS> or <CORRECTNESS>incorrect</CORRECTNESS>
                pattern = r"<correctness>(.*)</correctness>"
                correctness_value = re.search(pattern, correctness_value, re.DOTALL)
                if correctness_value is not None:
                    correctness_value = correctness_value.group(1).strip()
                if correctness_value == 'correct':
                    score = 1
                    # print(111)
                elif correctness_value == 'incorrect':
                    score = 0
            
            if score == 1:
                sample_passes += 1
                format_correct_counts[_type] += 1
                format_total_counts[_type] += 1
            else:
                format_total_counts[_type] += 1
            
            # 新增：保存每个响应的token数据
            response_token_data = {
                'token_count': token_count,
                'format': _type
            }
            sample_token_data['responses'].append(response_token_data)
        
        # 计算这个样本的平均通过率
        if len(response_lst) > 0:
            avg_pass_rate = sample_passes / len(response_lst)
            total_passes += avg_pass_rate
            total_token_count += sample_token_count
            
            # 新增：添加样本token数据到列表
            token_data_list.append(sample_token_data)
    
    # 计算总体指标
    avg_accuracy = (total_passes / total) * 100 if total > 0 else 0
    avg_token_count = total_token_count / total_responses if total_responses > 0 else 0
    
    # 输出结果
    result_title = f"avg@{k_samples} 结果"
    if dataset_name:
        result_title = f"{dataset_name} - {result_title}"
    print(f"\n=== {result_title} ===")
    print(f"平均准确率: {avg_accuracy:.2f}%")
    print(f"平均token数量: {avg_token_count:.1f}")
    print(f"格式分布:")
    for key, value in type_format_dict.items():
        percentage = len(value) / total_responses * 100 if total_responses > 0 else 0
        print(f"  {key}: {len(value)} ({percentage:.1f}%)")
    # 新增：各格式准确率
    if len(format_total_counts) > 0:
        print("各格式准确率:")
        for key in format_total_counts.keys():
            acc = (format_correct_counts[key] / format_total_counts[key] * 100) if format_total_counts[key] > 0 else 0.0
            print(f"  {key}: {acc:.2f}%  ({format_correct_counts[key]}/{format_total_counts[key]})")
    print("=" * 30)
    
    return {
        'avg_accuracy': avg_accuracy,
        'avg_token_count': avg_token_count,
        'format_distribution': dict(type_format_dict),
        'format_accuracy': {k: (format_correct_counts[k] / format_total_counts[k] * 100) if format_total_counts[k] > 0 else 0.0 for k in format_total_counts.keys()},
        'token_data': token_data_list  # 新增：返回详细的token数据
    }


def merge_with_subs_data(dataset, subs_path, file_name):
    """
    将原始数据与替代数据进行合并
    Args:
        dataset: 原始数据集
        subs_path: 替代数据路径
        file_name: 文件名
    Returns:
        合并后的数据集
    """
    # 检查替代文件是否存在
    subs_file_path = os.path.join(subs_path, file_name)
    if not os.path.exists(subs_file_path):
        print(f"警告：替代文件不存在 {subs_file_path}，使用原始数据")
        return dataset
    
    # 读取替代数据
    subs_dataset = pd.read_parquet(subs_file_path)
    
    if len(dataset) != len(subs_dataset):
        print(f"[警告] 行数不一致：{file_name} 原始={len(dataset)} 替代={len(subs_dataset)}，按最小长度处理")
    
    n_rows = min(len(dataset), len(subs_dataset))
    TRIGGERS = "<LONG_COT>"
    
    replace_cnt = 0
    skip_cnt = 0
    
    # 创建副本避免修改原始数据
    merged_dataset = dataset.copy()
    
    for i in range(n_rows):
        # 取出两边的 responses
        r1 = dataset.iloc[i]["responses"]
        r2 = subs_dataset.iloc[i]["responses"]

        correctness_r1 = dataset.iloc[i]["answer_correctness"]
        correctness_r2 = subs_dataset.iloc[i]["answer_correctness"]
        
        # 统一转为 ndarray(object) 处理
        r1_is_nd = isinstance(r1, np.ndarray)
        r2_is_nd = isinstance(r2, np.ndarray)
        correctness_r1_is_nd = isinstance(correctness_r1, np.ndarray)
        correctness_r2_is_nd = isinstance(correctness_r2, np.ndarray)
        
        r1_arr = r1 if (r1_is_nd and r1.dtype == object) else np.array(r1, dtype=object)
        r2_arr = r2 if (r2_is_nd and r2.dtype == object) else np.array(r2, dtype=object)
        correctness_r1_arr = correctness_r1 if (correctness_r1_is_nd and correctness_r1.dtype == object) else np.array(correctness_r1, dtype=object)
        correctness_r2_arr = correctness_r2 if (correctness_r2_is_nd and correctness_r2.dtype == object) else np.array(correctness_r2, dtype=object)
        
        if r1_arr.size == 0 or r2_arr.size == 0:
            continue
        
        # 当前行内：r2 的候选不可重复使用
        used_k = set()
        available_idx = [k for k in range(len(r2_arr)) if k not in used_k]
        
        # 遍历 r1，遇到触发词就从 r2_arr 抽一个未用过的替换
        for j in range(len(r1_arr)):
            item = r1_arr[j]
            
            if isinstance(item, str) and TRIGGERS in item:
                if not available_idx:
                    skip_cnt += 1
                    continue
                k = random.choice(available_idx)
                r1_arr[j] = "<LONG_COT>\n<think>\n" + r2_arr[k]
                correctness_r1_arr[j] = correctness_r2_arr[k]
                used_k.add(k)
                available_idx.remove(k)
                replace_cnt += 1
        
        # 回写到合并数据集
        if r1_is_nd:
            merged_dataset.iloc[i]["responses"] = r1_arr
            if correctness_r1_is_nd:
                merged_dataset.iloc[i]["answer_correctness"] = correctness_r1_arr
            else:
                merged_dataset.iloc[i]["answer_correctness"] = correctness_r1_arr.tolist()
        else:
            merged_dataset.iloc[i]["responses"] = r1_arr.tolist()
            if correctness_r1_is_nd:
                merged_dataset.iloc[i]["answer_correctness"] = correctness_r1_arr.tolist()
            else:
                merged_dataset.iloc[i]["answer_correctness"] = correctness_r1_arr
    
    print(f"数据合并完成: {file_name}, 替换 {replace_cnt} 处，候选耗尽跳过 {skip_cnt} 处")
    return merged_dataset


def calculate_cons_at_k(dataset, config, tokenizer, file_name, dataset_name=""):
    """
    独立计算cons@k的指标：一致性准确率、token数量和格式分布
    Args:
        dataset: 包含prompts, responses, data_sources, reward_model_data的DataFrame
        config: 配置对象
        tokenizer: 分词器
        file_name: 文件名，用于判断前缀类型
        dataset_name: 数据集名称，用于显示结果
    """
    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    
    # 新增：保存详细的token数据
    token_data_list = []
    
    print(f"计算cons@{config.data.cons}指标...")
    # for cons in range(1, config.data.cons+1):
    total = len(dataset)
    passes = 0
    total_token_count = 0
    num_of_response = 0
    type_format_dict = defaultdict(list)
    # 新增：按format统计准确率（基于逐条response的判定）
    format_correct_counts = defaultdict(int)
    format_total_counts = defaultdict(int)
    for i in tqdm(range(total), desc="cons@k计算"):
        response_lst_full = responses.iloc[i]
        correctness_lst_full = dataset.iloc[i]["answer_correctness"]
        raw_response_lst_full = []
        raw_correctness_lst_full = []
        
        # 为每个响应添加前缀，同时保留对应的correctness
        for idx, r in enumerate(response_lst_full):
            prefix = find_instruct(file_name)
            r = prefix + r
            raw_response_lst_full.append(r)
            raw_correctness_lst_full.append(correctness_lst_full[idx])
        
        k = min(config.data.cons, len(raw_response_lst_full))
        # 随机采样k条响应和对应的correctness
        sampled_indices = random.sample(range(len(raw_response_lst_full)), k)
        response_lst = [raw_response_lst_full[idx] for idx in sampled_indices]
        correctness_lst = [raw_correctness_lst_full[idx] for idx in sampled_indices]

        data_source = data_sources.iloc[i]
        prompt = prompts.iloc[i]
        reward_data = reward_model_data.iloc[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        if data_source == 'ChartQA':
            ground_truth = ground_truth[0]
        score_lst = []
        answer_span_list = []
        
        # 新增：保存当前样本的token数据
        sample_token_data = {
            'responses': []
        }
        
        for idx, r in enumerate(response_lst):
            _type = get_type(r)
            if _type == 'CODE' and config.data.code_executable:
                r, code_executed = rebuild_solution_str(r)
            type_format_dict[_type].append(r)
            num_of_response += 1

            output = extract_solution(r)
            tokenized_response = tokenizer(r)
            token_count = len(tokenized_response['input_ids'])
            total_token_count += token_count

            if output is None:
                if 'aime' in file_name:
                    output = '\n'.join(r.strip().split('\n')[-5:])
                else:
                    # 即使output为None，也要保存token数据
                    response_token_data = {
                        'token_count': token_count,
                        'format': _type
                    }
                    sample_token_data['responses'].append(response_token_data)
                    continue
            answer_span_list.append(output)
            # 新增：逐条response按format统计正确性（基于reward_fn or correctness）
            indiv_score = reward_fn(output, ground_truth, method='sc')
            if indiv_score != 1:
                # 回落到对应correctness
                correctness_value = correctness_lst[idx].lower()
                pattern = r"<correctness>(.*)</correctness>"
                correctness_match = re.search(pattern, correctness_value, re.DOTALL)
                if correctness_match is not None:
                    correctness_value = correctness_match.group(1).strip()
                if correctness_value == 'correct':
                    indiv_score = 1
                elif correctness_value == 'incorrect':
                    indiv_score = 0
            if indiv_score == 1:
                format_correct_counts[_type] += 1
                format_total_counts[_type] += 1
            else:
                format_total_counts[_type] += 1
            
            # 新增：保存每个响应的token数据
            response_token_data = {
                'token_count': token_count,
                'format': _type
            }
            sample_token_data['responses'].append(response_token_data)
            
        if len(answer_span_list) == 0:
            continue

        sc_answer = find_most_common(answer_span_list, data_source)
        score = reward_fn(sc_answer, ground_truth, method='sc')
        
        # 如果reward_fn返回的score不为1，再用correctness判断
        if score != 1:
            # 对于cons@k，我们需要找到与sc_answer对应的correctness
            # 由于sc_answer是多个答案中的最频繁答案，我们需要找到对应的correctness
            # 这里我们取第一个匹配的correctness作为参考
            for idx, output in enumerate(answer_span_list):
                if output == sc_answer:
                    correctness_value = correctness_lst[idx].lower()
                    pattern = r"<correctness>(.*)</correctness>"
                    correctness_match = re.search(pattern, correctness_value, re.DOTALL)
                    if correctness_match is not None:
                        correctness_value = correctness_match.group(1).strip()
                    if correctness_value == 'correct':
                        score = 1
                    elif correctness_value == 'incorrect':
                        score = 0
                    break
        
        score_lst.append(score)
        max_score = np.max(score_lst)

        if max_score == 1:
            passes += 1
        
        # 新增：添加样本token数据到列表
        token_data_list.append(sample_token_data)
    
    # 计算总体指标
    cons_accuracy = (passes / total) * 100 if total > 0 else 0
    avg_token_count = total_token_count / num_of_response if num_of_response > 0 else 0
    
    # 输出结果
    result_title = f"cons@{config.data.cons} 结果"
    if dataset_name:
        result_title = f"{dataset_name} - {result_title}"
    print(f"\n=== {result_title} ===")
    print(f"一致性准确率: {cons_accuracy:.2f}%")
    print(f"平均token数量: {avg_token_count*config.data.cons:.1f}")
    print(f"格式分布:")
    for key, value in type_format_dict.items():
        percentage = len(value) / num_of_response * 100 if num_of_response > 0 else 0
        print(f"  {key}: {len(value)} ({percentage:.1f}%)")
    # 新增：各格式准确率
    if len(format_total_counts) > 0:
        print("各格式准确率:")
        for key in format_total_counts.keys():
            acc = (format_correct_counts[key] / format_total_counts[key] * 100) if format_total_counts[key] > 0 else 0.0
            print(f"  {key}: {acc:.2f}%  ({format_correct_counts[key]}/{format_total_counts[key]})")
    print("=" * 30)
    
    return {
        'cons_accuracy': cons_accuracy,
        'avg_token_count': avg_token_count,
        'format_distribution': dict(type_format_dict),
        'format_accuracy': {k: (format_correct_counts[k] / format_total_counts[k] * 100) if format_total_counts[k] > 0 else 0.0 for k in format_total_counts.keys()},
        'token_data': token_data_list  # 新增：返回详细的token数据
    }


def save_token_data(token_data, file_name, dataset_name, metric_type, output_dir="./token_data"):
    """
    保存token数据到JSONL文件，每行包含token count
    Args:
        token_data: token数据列表
        file_name: 原始文件名
        dataset_name: 数据集名称
        metric_type: 指标类型 (avg@k 或 cons@k)
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成文件名，处理特殊字符
    base_name = file_name.replace('.parquet', '')
    # 替换文件名中不允许的字符
    safe_dataset_name = dataset_name.replace('/', '_').replace('\\', '_')
    safe_base_name = base_name.replace('/', '_').replace('\\', '_')
    jsonl_filename = f"{safe_base_name}_{safe_dataset_name}_{metric_type}_tokens_{timestamp}.jsonl"
    
    # 保存为JSONL格式，每行一个token count
    jsonl_path = os.path.join(output_dir, jsonl_filename)
    token_count = 0
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sample in token_data:
            for response in sample['responses']:
                token_record = {
                    "token_count": response['token_count'],
                    "format": response['format']
                }
                f.write(json.dumps(token_record, ensure_ascii=False) + '\n')
                token_count += 1
    
    print(f"Token数据已保存: {jsonl_path} (共{token_count}条记录)")
    
    return jsonl_path


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    tokenizer_path = config.data.tokenizer_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )

    print(f"config.data.cons: {config.data.cons}")
    print(f"config.data.k: {config.data.k}")
    # if config.data.cons > 1:
    #     assert config.data.k == 1, "cons > 1, k must be 1"
    # if config.data.k > 1:
    #     assert config.data.cons == 1, "k > 1, cons must be 1"
    print(f"config.data.path: {config.data.path}")
    
    # 检查是否有subs_path参数
    subs_path = getattr(config.data, 'subs_path', None)
    if subs_path and subs_path != 'null' and subs_path != 'None':
        print(f"config.data.subs_path: {subs_path}")
        print("将进行数据合并操作...")
    else:
        print("未提供subs_path，将使用原始数据...")
    print(f'-------------------')
    for _file in os.listdir(config.data.path):
        if 'answer_extraction.' not in _file or not (_file.endswith('.parquet')) or (not str(config.data.temp) in _file) or (not '_'+str(config.data.samples) in _file) or any(keyword in _file for keyword in []):
            continue

        print(f'Processing {_file}')

        local_path = copy_local_path_from_hdfs(os.path.join(config.data.path, _file))
        dataset = pd.read_parquet(local_path)
        
        # 如果有subs_path，进行数据合并
        if subs_path and subs_path != 'null' and subs_path != 'None':
            dataset = merge_with_subs_data(dataset, subs_path, _file)
        
        print(f"数据源: {dataset[config.data.data_source_key][0]}")
        print('-------------------')
        
        # 检查是否为AIME数据集
        data_source = dataset[config.data.data_source_key][0]
        speficed_data_source = config.data.specified_data_source
        if speficed_data_source and data_source != speficed_data_source:
            print(f"指定数据源: {speficed_data_source}，跳过计算...")
            continue
        if data_source == 'AIME':
            print("检测到AIME数据集，将分为AIME24和AIME25两部分计算...")
            
            # 分割数据集：前30行为AIME24，后30行为AIME25
            aime24_dataset = dataset.iloc[:30].copy()
            aime25_dataset = dataset.iloc[30:60].copy()  # 取后30行
            
            print(f"AIME24数据集大小: {len(aime24_dataset)}")
            print(f"AIME25数据集大小: {len(aime25_dataset)}")
            
            # 计算AIME24的指标
            avg_k_results_24 = calculate_avg_at_k(aime24_dataset, config, tokenizer, _file, k_samples=config.data.k, dataset_name="AIME24")
            cons_k_results_24 = calculate_cons_at_k(aime24_dataset, config, tokenizer, _file, dataset_name="AIME24")
            
            # 保存AIME24的token数据
            # save_token_data(avg_k_results_24['token_data'], _file, "AIME24", f"avg@{config.data.k}")
            
            # 计算AIME25的指标
            avg_k_results_25 = calculate_avg_at_k(aime25_dataset, config, tokenizer, _file, k_samples=config.data.k, dataset_name="AIME25")
            cons_k_results_25 = calculate_cons_at_k(aime25_dataset, config, tokenizer, _file, dataset_name="AIME25")
            
            # 保存AIME25的token数据
            # save_token_data(avg_k_results_25['token_data'], _file, "AIME25", f"avg@{config.data.k}")
            
        else:
            # 计算avg@k指标
            avg_k_results = calculate_avg_at_k(dataset, config, tokenizer, _file, k_samples=config.data.k, dataset_name=data_source)

            # 计算cons@k指标
            cons_k_results = calculate_cons_at_k(dataset, config, tokenizer, _file, dataset_name=data_source)
            
            # 保存token数据
            # save_token_data(avg_k_results['token_data'], _file, data_source, f"avg@{config.data.k}")
        
        # print(avg_k_results)
        # print(cons_k_results)
        
        print('-------------------\n\n')




if __name__ == '__main__':
    main()