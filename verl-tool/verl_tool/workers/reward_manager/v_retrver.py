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
from .utils import replace_consecutive_tokens


# ============= Core reward functions =============



def parse_ranking_from_response(response_str: str) -> Tuple[Optional[List[int]], bool]:
    """
    Parse ranking list from response string.
    Supported formats: [2,3,4,1,5] or [2, 3, 4, 1, 5]
    """
    is_list_format = False
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match = pattern.search(response_str)
    if not match:
        return None,is_list_format
    
    answer_content = match.group(1).strip()
    
    
    # Try to parse as a list
    try:
        # Check explicit list format
        if answer_content.startswith('[') and answer_content.endswith(']'):
            ranking = eval(answer_content)
            if isinstance(ranking, list) and all(isinstance(x, int) for x in ranking):
                is_list_format = True
                return ranking,is_list_format
        
        # Loose mode: extract all numbers
        numbers = re.findall(r'\d+', answer_content)
        if numbers:
            ranking = [int(x) for x in numbers]
            is_list_format = False
            return ranking,is_list_format
    except:
        pass
    
    return None,is_list_format


def compute_format_reward(response_str: str, num_candidates: int) -> Dict[str, float]:
    """
    Return scores: 0.5 if response matches <think>...</think><answer>...</answer> (fullmatch),
    and +0.5 if the answer part is in explicit list format [x,x,x].
    """
    rewards = {
        'format_reward': 0.0,
        'is_valid_format': 0.0,
        'is_list_format': 0.0,
    }

    pattern = re.compile(r"<think>(.*?)</think>.*<answer>(.*?)</answer>.*", re.DOTALL)
    if not re.fullmatch(pattern, response_str):
        return rewards

    rewards['is_valid_format'] = 1.0
    rewards['format_reward'] = 0.5

    ranking, is_list_format = parse_ranking_from_response(response_str)
    if is_list_format and isinstance(ranking, list):
        rewards['is_list_format'] = 1.0
        rewards['format_reward'] += 0.5

    rewards['format_reward'] = max(0.0, min(1.0, rewards['format_reward']))
    return rewards


def compute_ranking_reward(
    predicted_ranking: List[int], 
    ground_truth_position: int,
    sigma: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """
    Compute ranking reward using a Gaussian kernel.
    Using Gaussian kernel only; sigma controls the width.
    Args:
        predicted_ranking: predicted ranking list (e.g. [2,3,4,1,5])
        ground_truth_position: ground truth index (1-indexed)
        sigma: Gaussian kernel width (default 0.5)
    Returns:
        (gaussian_reward, detailed_scores)
    """
    if not predicted_ranking:
        return 0.0, {'rank_position': -1, 'gaussian_reward': 0.0}
    
    # Find position of ground truth in predicted ranking
    try:
        rank_of_gt = predicted_ranking.index(ground_truth_position) + 1  # 1-indexed
    except ValueError:
        # Ground truth not in predicted ranking
        return 0.0, {'rank_position': -1, 'gaussian_reward': 0.0}
    
    # Gaussian kernel reward: exp(-((rank - 1)^2) / (2 * sigma^2))
    # rank=1 -> reward=1.0
    # rank=2 -> reward~0.32 when sigma=0.5
    # rank=3 -> reward~0.02 when sigma=0.5
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
    Compute combined reward.
    Final reward = format_weight * format_reward + ranking_weight * ranking_reward.
    Args:
        response_str: model response
        ground_truth_position: ground truth index (1-indexed)
        num_candidates: number of candidates
        sigma: gaussian kernel width
        format_weight: weight for format reward
        ranking_weight: weight for ranking reward
    Returns:
        dict with detailed scores
    """
    # 1. compute format reward
    format_scores = compute_format_reward(response_str, num_candidates)
    format_reward = format_scores['format_reward']
    
    # 2. parse ranking
    ranking,is_list_format = parse_ranking_from_response(response_str)
    
    if ranking is None:
        # Unable to parse ranking, return format-only scores
        return {
            **format_scores,
            'ranking_reward': 0.0,
            'rank_position': -1,
            'gaussian_reward': 0.0,
            'final_reward': format_weight * format_reward,
        }
    
    # 3. compute ranking reward
    ranking_reward, ranking_details = compute_ranking_reward(
        ranking, ground_truth_position, sigma=sigma
    )
    # 4. combine final reward
    final_reward = format_weight * format_reward + ranking_weight * ranking_reward
    
    return {
        **format_scores,
        **ranking_details,
        'ranking_reward': ranking_reward,
        'final_reward': final_reward,
    }


# ============= Reward manager =============

@register("pixel_reasoner")
class PixelReasonerRewardManager:
    """
    Pixel Reasoner reward manager (supports ranking tasks).
    """
    name = "pixel_reasoner"
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.step = None
        
        # ranking reward parameters
        self.gaussian_sigma = kwargs.get('gaussian_sigma', 1)
        self.format_weight = kwargs.get('format_weight', 0.3)
        self.ranking_weight = kwargs.get('ranking_weight', 0.7)
        
        # process reward parameters (curiosity & redundancy)
        self.enable_curiosity_reward = True
        self.enable_redundancy_penalty = True
        

        self.group_tool_call_rate_lower_bound = kwargs.get('group_tool_call_rate_lower_bound', 0.6)
        self.curiosity_weight = kwargs.get('curiosity_weight', 0.6)
        

        self.action_redundancy_limit = 1
        self.redundancy_weight = 0.05
        
        if "record_dir" in kwargs:
            self.record_dir = Path(kwargs['record_dir'])
            self.record_dir.mkdir(parents=True, exist_ok=True)
        
    # Curriculum learning removed — fixed parameters are used instead.
    
    def get_group_info(self, data: DataProto):
        group_info = {}
        for i in range(len(data)):
            data_item = data[i]
            num_turn = data_item.non_tensor_batch.get("turns_stats", 0)
            num_valid_action = data_item.non_tensor_batch.get("valid_action_stats", 0)
            uid = data_item.non_tensor_batch.get('uid', i)

            if uid not in group_info:   
                group_info[uid] = {
                    'num_turns': [],
                    'num_valid_actions': []
                }
            
            group_info[uid]['num_turns'].append(num_turn)
            group_info[uid]['num_valid_actions'].append(num_valid_action)

        for uid, info in group_info.items():
            info['num_turns'] = np.array(info['num_turns'])
            info['num_valid_actions'] = np.array(info['num_valid_actions'])
            info['group_tool_call_rate'] = np.mean([1 if nva > 0 else 0 for nva in info['num_valid_actions']])
            info['tool_call_total'] = info['num_valid_actions'].sum()
        
        return group_info    
    
    def compute_process_rewards(self, response: str, data_i, group_info: dict) -> Tuple[float, Dict[str, float]]:
        num_turn = data_i.non_tensor_batch.get("turns_stats", 0)
        num_valid_action = data_i.non_tensor_batch.get("valid_action_stats", 0)
        
        process_reward_total = 0.0
        detailed_scores = {}
        
        detailed_scores['curiosity_reward']=0.0
        detailed_scores['action_redundancy_penalty']=0.0

        
   
        if self.enable_curiosity_reward:
            group_tool_call_rate = group_info.get('group_tool_call_rate', 0)
            
            if num_valid_action > 0:
                if group_tool_call_rate < self.group_tool_call_rate_lower_bound:
                    curiosity_reward = self.curiosity_weight * (
                        self.group_tool_call_rate_lower_bound - group_tool_call_rate
                    )
                    process_reward_total += curiosity_reward
                    detailed_scores['curiosity_reward'] = curiosity_reward
                else:
                    detailed_scores['curiosity_reward'] = 0.0

        if self.enable_redundancy_penalty:
            if num_valid_action > self.action_redundancy_limit:
                excess_actions = num_valid_action - self.action_redundancy_limit
                penalty = -excess_actions * self.redundancy_weight
                process_reward_total += penalty
                detailed_scores['action_redundancy_penalty'] = penalty
            else:
                detailed_scores['action_redundancy_penalty'] = 0.0
        
        return process_reward_total, detailed_scores
    
    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards"""
        save_record = data.meta_info.get('save_record', True)


        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
            else:
                import time
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"pixel_reasoner-ranking-{time.strftime('%Y%m%d-%H%M%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)
        

        if self.step is None:
            last_step_idx = 0
            import os

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

        group_info = self.get_group_info(data)
        
        for i in range(len(data)):
            data_item = data[i]

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


            #decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)


            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            ground_truth_position = extra_info.get('ground_truth_position', 1)
            num_candidates = extra_info.get('num_candidates', 5)
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, 'unknown')

            base_scores = compute_reward(
                response_str,
                ground_truth_position,
                num_candidates,
                sigma=self.gaussian_sigma,
                format_weight=self.format_weight,
                ranking_weight=self.ranking_weight
            )
            
            base_reward = base_scores['final_reward']

            uid = data_item.non_tensor_batch.get('uid', i)
            process_reward, process_scores = self.compute_process_rewards(
                response_str,
                data_item,
                group_info.get(uid, {})
            )
            
            final_reward = base_reward + process_reward
            
            final_reward = max(min(final_reward, 1.5), -0.5)

            all_scores = {
                **base_scores,
                **process_scores,
                'process_reward': process_reward,
                'base_reward': base_reward,
                'final_reward': final_reward,
            }


            all_scores["accuracy"] = 1 if final_reward > 0.6 else 0
            if all_scores['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

            if isinstance(all_scores, dict):
                reward = all_scores["final_reward"]
                # Store the information including original reward
                for key, value in all_scores.items():
                    reward_extra_info[key].append(value)
                if self.num_examine == 1:
                    reward = all_scores["accuracy"] # for validation
            else:
                if self.num_examine == 1:
                    reward = all_scores if all_scores > 0.6 else 0.0
                else:
                    reward = all_scores


            reward_tensor[i, valid_response_length - 1] = reward 

            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("\n" + "="*80)
                print(f"[Example {already_print_data_sources[data_source]}]")
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[response] {response_str}")
                print(f"[ground_truth] Position {ground_truth_position}")
                print(f"[num_candidates] {num_candidates}")
                print("[scores]")
                for key, value in all_scores.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                print("="*80 + "\n")

            tool_interact_info_i = data_item.non_tensor_batch.get('tool_interact_info', None)
            if tool_interact_info_i is not None:
                # crop the image
                for tool_interact in tool_interact_info_i:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x for x in tool_interact['image']]  
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'] # for debug

            
            to_save_prompt = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            to_save_response = self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False)
            to_save_prompt = replace_consecutive_tokens(to_save_prompt, token="<|image_pad|>")
            to_save_response = replace_consecutive_tokens(to_save_response, token="<|image_pad|>")
            if 'responses_with_loss_mask' in data_item.batch:
                to_save_response_with_loss_mask = self.tokenizer.decode(valid_response_ids_with_loss_mask, skip_special_tokens=False)
                to_save_response_with_loss_mask = replace_consecutive_tokens(to_save_response_with_loss_mask, token=self.tokenizer.pad_token)
            
            to_save_records.append({
                'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch['extra_info'] else None,
                'data_source': data_source,
                "prompt": to_save_prompt,
                "response": to_save_response,
                'response_with_loss_mask': to_save_response_with_loss_mask if 'responses_with_loss_mask' in data_item.batch else None,
                'ground_truth_position': ground_truth_position,
                'score': all_scores,
                'reward': reward,
                'tool_interact_info': tool_interact_info_i,
                'extra_info': data_item.non_tensor_batch.get('extra_info', None),
            })
            if "turns_stats" in data_item.non_tensor_batch:
                to_save_records[i]['num_turn'] = data[i].non_tensor_batch["turns_stats"]
                to_save_records[i]['num_valid_action'] = data[i].non_tensor_batch["valid_action_stats"]
                to_save_records[i]['is_done'] = not data[i].non_tensor_batch["active_mask"]
        if save_record:
            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"{self.name}-step-val-{self.step}.json"
            else:
                temp_file = self.record_dir / f"{self.name}-step-{self.step}.json"
            self.step += 1
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            # if temp_file.exists():
            #     with open(temp_file, "r") as f:
            #         existing_records = json.load(f)
            #     to_save_records = existing_records + to_save_records
            if temp_file.exists() and temp_file.stat().st_size > 0: 
                try:
                    with open(temp_file, "r") as f:
                        existing_records = json.load(f)
                    to_save_records = existing_records + to_save_records
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⚠️ Warning: Failed to load existing records from {temp_file}: {e}")
          
            with open(temp_file, "w") as f:
                json.dump(to_save_records, f, indent=4)
            print(f"Saved records to {temp_file}")
        
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
   pass