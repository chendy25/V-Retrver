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

import os
import torch
import json
import time
import numpy as np
import regex as re
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path
from verl import DataProto
from verl.workers.reward_manager import register
#from .utils import replace_consecutive_tokens
from verl_tool.workers.reward_manager.utils import replace_consecutive_tokens

# ============= 🔧 核心奖励函数 =============

def parse_ranking_from_response(response_str: str) -> Tuple[Optional[List[int]], bool]:
    """
    从响应中解析排序列表
    支持格式：[2,3,4,1,5] 或 [2, 3, 4, 1, 5]
    """
    is_list_format = False
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match = pattern.search(response_str)
    if not match:
        return None, is_list_format
    
    answer_content = match.group(1).strip()
    
    
    # 尝试解析为列表
    try:
        # 检查是否为列表格式
        if answer_content.startswith('[') and answer_content.endswith(']'):
            ranking = eval(answer_content)
            if isinstance(ranking, list) and all(isinstance(x, int) for x in ranking):
                is_list_format = True
                return ranking, is_list_format
        
        # 宽松模式：提取所有数字
        numbers = re.findall(r'\d+', answer_content)
        if numbers:
            ranking = [int(x) for x in numbers]
            is_list_format = False
            return ranking, is_list_format
    except:
        pass
    
    return None, is_list_format


def compute_format_reward(response_str: str, num_candidates: int) -> Dict[str, float]:
    """
    计算格式奖励
    
    步骤：
    1. 检查 <think>...</think><answer>...</answer> 格式（fullmatch）
    2. 如果 fullmatch，format_reward = 1.0
    3. 在 1.0 基础上添加惩罚：
       - 索引越界惩罚
       - 长度不一致惩罚
       - 不是 [x,x,x] 格式的惩罚
    
    Returns:
        包含各项得分的字典
    """
    rewards = {
        'format_reward': 0.0,
        'is_valid_format': 0.0,
        'is_list_format': 0.0,
        'index_penalty': 0.0,
        'length_penalty': 0.0,
    }
    
    # 检查 <think>...</think><answer>...</answer> 格式（使用 fullmatch）
    pattern = re.compile(r"<think>(.*?)</think>.*<answer>(.*?)</answer>.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response_str)
    
    if not format_match:
        # 不符合基础格式，format_reward = 0
        return rewards
    
    # 符合基础格式，format_reward 从 1.0 开始
    rewards['is_valid_format'] = 1.0
    rewards['format_reward'] = 1.0
    
    # 尝试解析排序列表
    ranking, is_list_format = parse_ranking_from_response(response_str)
    
    if ranking is None or not is_list_format:
        # 不是列表格式，扣除惩罚
        rewards['format_reward'] -= 0.8  # 不是列表格式惩罚
        return rewards
    
    # 是列表格式
    rewards['is_list_format'] = 1.0
    
    # 检查索引越界
    invalid_indices = [idx for idx in ranking if idx < 1 or idx > num_candidates]
    if invalid_indices:
        # 索引越界惩罚：每个越界索引扣 0.2/num_candidates
        index_penalty = -0.8
        rewards['index_penalty'] = index_penalty
        rewards['format_reward'] += index_penalty
    
    # 检查长度不一致
    if len(ranking) != num_candidates:
        # 长度不一致惩罚：差异越大惩罚越重
        length_diff = abs(len(ranking) - num_candidates)
        length_penalty = -0.8
        rewards['length_penalty'] = length_penalty
        rewards['format_reward'] += length_penalty
    
    # 确保 format_reward 在 [0, 1] 范围内
    rewards['format_reward'] = max(0.0, min(1.0, rewards['format_reward']))
    
    return rewards


def compute_ranking_reward(
    predicted_ranking: List[int], 
    ground_truth_position: int,
    sigma: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """
    使用高斯核计算排序奖励
    
    只使用高斯核，sigma = 0.5
    
    Args:
        predicted_ranking: 预测排序 [2,3,4,1,5]
        ground_truth_position: 真实答案位置
        sigma: 高斯核宽度，默认 0.5
    
    Returns:
        (gaussian_reward, detailed_scores)
    """
    if not predicted_ranking:
        return 0.0, {'rank_position': -1, 'gaussian_reward': 0.0}
    
    # 找到真实答案在预测排序中的位置
    try:
        rank_of_gt = predicted_ranking.index(ground_truth_position) + 1  # 1-indexed
    except ValueError:
        # 真实答案不在排序中
        return 0.0, {'rank_position': -1, 'gaussian_reward': 0.0}
    
    # 高斯核奖励: exp(-((rank - 1)^2) / (2 * sigma^2))
    # rank=1 → reward=1.0
    # rank=2 → reward=0.32 (sigma=0.5)
    # rank=3 → reward=0.02 (sigma=0.5)
    gaussian_reward = np.exp(-((rank_of_gt - 1) ** 2) / (2 * sigma ** 2))
    
    detailed_scores = {
        'rank_position': rank_of_gt,
        'gaussian_reward': float(gaussian_reward),
    }
    
    return gaussian_reward, detailed_scores


def compute_reward(
    response_str: str,
    ground_truth_position: int,
    num_candidates: int,
    sigma: float = 0.5,
    format_weight: float = 0.3,
    ranking_weight: float = 0.7
) -> Dict[str, float]:
    """
    综合奖励计算
    
    最终奖励 = format_weight * format_reward + ranking_weight * ranking_reward
    总权重和为1（format_weight + ranking_weight = 1）
    
    Args:
        response_str: 模型响应
        ground_truth_position: 真实答案位置
        num_candidates: 候选数量
        sigma: 高斯核宽度
        format_weight: 格式奖励权重
        ranking_weight: 排序奖励权重
    
    Returns:
        包含所有得分的字典
    """
    # 1. 计算格式奖励
    format_scores = compute_format_reward(response_str, num_candidates)
    format_reward = format_scores['format_reward']
    
    # 2. 解析排序
    ranking, is_list_format = parse_ranking_from_response(response_str)
    
    if ranking is None:
        # 无法解析排序，只返回格式分
        return {
            **format_scores,
            'ranking_reward': 0.0,
            'rank_position': -1,
            'gaussian_reward': 0.0,
            'final_reward': format_weight * format_reward,
        }
    
    # 3. 计算排序奖励
    ranking_reward, ranking_details = compute_ranking_reward(
        ranking, ground_truth_position, sigma=sigma
    )
    
    # 4. 组合最终奖励
    final_reward = format_weight * format_reward + ranking_weight * ranking_reward
    
    return {
        **format_scores,
        **ranking_details,
        'ranking_reward': ranking_reward,
        'final_reward': final_reward,
    }


# ============= 🔧 奖励管理器 =============

@register("text_cot")
class TextCoTRewardManager:
    """
    纯文本CoT排序任务奖励管理器
    """
    name = "text_cot"
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.step = None
        
        # 排序奖励参数
        self.gaussian_sigma = kwargs.get('gaussian_sigma', 0.5)  # 高斯核宽度
        self.format_weight = kwargs.get('format_weight', 0.3)    # 格式奖励权重
        self.ranking_weight = kwargs.get('ranking_weight', 0.7)  # 排序奖励权重
        
        if "record_dir" in kwargs:
            self.record_dir = Path(kwargs['record_dir'])
            self.record_dir.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, data: DataProto, return_dict=False):
        """计算奖励"""
        save_record = data.meta_info.get('save_record', True)

        # 初始化记录目录
        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
            else:
                import time
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"text_cot-{time.strftime('%Y%m%d-%H%M%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查step索引
        if self.step is None:
            last_step_idx = 0
            import os
            # 确保目录存在
            if not self.record_dir.exists():
                self.record_dir.mkdir(parents=True, exist_ok=True)
            for file in os.listdir(self.record_dir):
                if self.num_examine == 1:
                    if re.search(r"step-val-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
                else:
                    if re.search(r"step-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
            self.step = last_step_idx + 1
        
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']

        # 如果已有rm_scores，直接返回
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        from collections import defaultdict
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        to_save_records = []

        for i in range(len(data)):
            data_item = data[i]

            # 解码prompt和response
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            if "loss_mask" in data_item.batch:
                loss_mask = data_item.batch['loss_mask']
                valid_response_ids_with_loss_mask = torch.where(loss_mask[prompt_length:prompt_length + valid_response_length] == 1, valid_response_ids, self.tokenizer.pad_token_id)
            else:
                valid_response_ids_with_loss_mask = valid_response_ids

            # 解码
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # 获取额外信息
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            ground_truth_position = extra_info.get('ground_truth_position', 1)
            num_candidates = extra_info.get('num_candidates', 5)
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, 'unknown')
            
            # ============= 计算奖励 =============
            
            # 基础奖励（格式 + 排序）
            base_scores = compute_reward(
                response_str,
                ground_truth_position,
                num_candidates,
                sigma=self.gaussian_sigma,
                format_weight=self.format_weight,
                ranking_weight=self.ranking_weight
            )
            
            final_reward = base_scores['final_reward']
            
            # 限制在合理范围
            final_reward = max(min(final_reward, 1.5), -0.5)
            
            # 合并所有得分
            all_scores = {
                **base_scores,
                'final_reward': final_reward,
            }

            all_scores["accuracy"] = 1 if final_reward > 0.6 else 0
            if all_scores['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

            if isinstance(all_scores, dict):
                reward = all_scores["final_reward"]
                # 存储信息
                for key, value in all_scores.items():
                    reward_extra_info[key].append(value)
                if self.num_examine == 1:
                    reward = all_scores["accuracy"] # 验证时使用
            else:
                if self.num_examine == 1:
                    reward = all_scores if all_scores > 0.6 else 0.0
                else:
                    reward = all_scores

            # 记录到reward tensor
            reward_tensor[i, valid_response_length - 1] = reward 

            # 打印示例
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("\n" + "="*80)
                print(f"[示例 {already_print_data_sources[data_source]}]")
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[response] {response_str}")
                print(f"[ground_truth] 位置 {ground_truth_position}")
                print(f"[num_candidates] {num_candidates}")
                print("[得分]")
                for key, value in all_scores.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                print("="*80 + "\n")
                    
            # 保存记录
            to_save_prompt = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            to_save_response = self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False)
            if 'responses_with_loss_mask' in data_item.batch:
                to_save_response_with_loss_mask = self.tokenizer.decode(valid_response_ids_with_loss_mask, skip_special_tokens=False)
            
            to_save_records.append({
                'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch['extra_info'] else None,
                'data_source': data_source,
                "prompt": to_save_prompt,
                "response": to_save_response,
                'response_with_loss_mask': to_save_response_with_loss_mask if 'responses_with_loss_mask' in data_item.batch else None,
                'ground_truth_position': ground_truth_position,
                'score': all_scores,
                'reward': reward,
                'extra_info': data_item.non_tensor_batch.get('extra_info', None),
            })
            
        if save_record:
            # 保存记录到文件
            if self.num_examine == 1:
                temp_file = self.record_dir / f"{self.name}-step-val-{self.step}.json"
            else:
                temp_file = self.record_dir / f"{self.name}-step-{self.step}.json"
            self.step += 1
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            if temp_file.exists() and temp_file.stat().st_size > 0: 
                try:
                    with open(temp_file, "r") as f:
                        existing_records = json.load(f)
                    to_save_records = existing_records + to_save_records
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⚠️ 警告: 无法从 {temp_file} 加载现有记录: {e}")
            with open(temp_file, "w") as f:
                json.dump(to_save_records, f, indent=4)
            print(f"记录已保存到 {temp_file}")
        
        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info['correct_response_length'] else 0.0
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info['wrong_response_length'] else 0.0
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

if __name__ == "__main__":
    print("="*80)
    print("🧪 纯文本CoT奖励管理器 - 单元测试")
    print("="*80)
    
    # ============= 测试 1: 格式奖励测试 =============
    print("\n### 测试 1: 格式奖励 ###")
    
    test_cases_format = [
        {
            "name": "完美格式",
            "response": "<think>让我分析...</think><answer>[2,3,4,1,5]</answer>",
            "num_candidates": 5,
            "expected": {"is_valid_format": 1.0, "is_list_format": 1.0, "format_reward": 1.0}
        },
        {
            "name": "缺少think标签",
            "response": "<answer>[2,3,4,1,5]</answer>",
            "num_candidates": 5,
            "expected": {"is_valid_format": 0.0, "format_reward": 0.0}
        },
        {
            "name": "索引越界",
            "response": "<think>思考中...</think><answer>[2,3,4,1,10]</answer>",
            "num_candidates": 5,
            "expected": {"is_valid_format": 1.0, "is_list_format": 1.0}
        },
        {
            "name": "长度不一致",
            "response": "<think>思考中...</think><answer>[2,3,4]</answer>",
            "num_candidates": 5,
            "expected": {"is_valid_format": 1.0, "is_list_format": 1.0}
        },
        {
            "name": "非列表格式",
            "response": "<think>思考中...</think><answer>3</answer>",
            "num_candidates": 5,
            "expected": {"is_valid_format": 1.0, "is_list_format": 0.0, "format_reward": 0.0}
        },
    ]
    
    for i, test in enumerate(test_cases_format, 1):
        result = compute_format_reward(test["response"], test["num_candidates"])
        print(f"\n测试用例 {i}: {test['name']}")
        print(f"  Response: {test['response'][:60]}...")
        print(f"  Results:")
        for key in ['is_valid_format', 'is_list_format', 'format_reward', 'index_penalty', 'length_penalty']:
            value = result.get(key, 0.0)
            print(f"    {key}: {value:.3f}")
        
        # 验证关键指标
        for key, expected_val in test["expected"].items():
            actual_val = result[key]
            status = "✅" if abs(actual_val - expected_val) < 0.01 else "❌"
            print(f"  {status} {key}: expected={expected_val}, actual={actual_val:.3f}")
    
    # ============= 测试 2: 排序奖励测试 =============
    print("\n\n### 测试 2: 排序奖励（高斯核） ###")
    
    test_cases_ranking = [
        {
            "name": "第1名（完美）",
            "ranking": [2, 3, 4, 1, 5],
            "gt": 2,
            "expected_rank": 1,
            "expected_reward": 1.0,
        },
        {
            "name": "第2名",
            "ranking": [3, 2, 4, 1, 5],
            "gt": 2,
            "expected_rank": 2,
            "expected_reward": 0.32,  # exp(-1/(2*0.5^2)) ≈ 0.135
        },
        {
            "name": "第3名",
            "ranking": [3, 4, 2, 1, 5],
            "gt": 2,
            "expected_rank": 3,
            "expected_reward": 0.02,  # exp(-4/(2*0.5^2)) ≈ 0.018
        },
        {
            "name": "不在排序中",
            "ranking": [1, 3, 4, 5],
            "gt": 2,
            "expected_rank": -1,
            "expected_reward": 0.0,
        },
    ]
    
    for i, test in enumerate(test_cases_ranking, 1):
        reward, details = compute_ranking_reward(test["ranking"], test["gt"], sigma=0.5)
        print(f"\n测试用例 {i}: {test['name']}")
        print(f"  Ranking: {test['ranking']}")
        print(f"  Ground Truth: {test['gt']}")
        print(f"  Rank Position: {details['rank_position']}")
        print(f"  Gaussian Reward: {reward:.4f}")
        
        status_rank = "✅" if details['rank_position'] == test["expected_rank"] else "❌"
        status_reward = "✅" if abs(reward - test["expected_reward"]) < 0.1 else "❌"
        print(f"  {status_rank} Expected Rank: {test['expected_rank']}")
        print(f"  {status_reward} Expected Reward: {test['expected_reward']:.3f}")
    
    # ============= 测试 3: 综合奖励测试 =============
    print("\n\n### 测试 3: 综合奖励（格式 + 排序） ###")
    
    test_cases_combined = [
        {
            "name": "完美答案",
            "response": "<think>基于文本相似性，我将它们排序为...</think><answer>[2,3,4,1,5]</answer>",
            "gt": 2,
            "num_candidates": 5,
        },
        {
            "name": "格式错误但答案对",
            "response": "<think>分析...</think><answer>2</answer>",
            "gt": 2,
            "num_candidates": 5,
        },
        {
            "name": "格式对但排序差",
            "response": "<think>让我想想...</think><answer>[5,4,3,1,2]</answer>",
            "gt": 2,
            "num_candidates": 5,
        },
        {
            "name": "格式和排序都差",
            "response": "<think>我认为...</think><answer>5</answer>",
            "gt": 2,
            "num_candidates": 5,
        },
    ]
    
    for i, test in enumerate(test_cases_combined, 1):
        result = compute_reward(
            test["response"],
            test["gt"],
            test["num_candidates"],
            sigma=0.5,
            format_weight=0.3,
            ranking_weight=0.7
        )
        print(f"\n测试用例 {i}: {test['name']}")
        print(f"  Response: {test['response'][:60]}...")
        print(f"  Ground Truth: {test['gt']}")
        print(f"  Results:")
        print(f"    Format Reward: {result['format_reward']:.3f}")
        print(f"    Ranking Reward: {result['ranking_reward']:.3f}")
        print(f"    Final Reward: {result['final_reward']:.3f} (0.3*{result['format_reward']:.2f} + 0.7*{result['ranking_reward']:.2f})")
    
    print("\n" + "="*80)
    print("🎉 测试完成！")
    print("="*80)