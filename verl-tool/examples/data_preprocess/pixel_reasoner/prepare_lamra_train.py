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
Preprocess the GSM8k dataset to parquet format
"""
import fire
import os
import datasets
import zipfile
import cv2
import os
import regex as re
import json
from glob import glob
from pathlib import Path
from huggingface_hub import hf_hub_download
from collections import defaultdict
from copy import deepcopy

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

system_prompt = """You are a helpful assistant for multimodal retrieval tasks.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}
{"type": "function", "function": {"name": "select_images", "description": "Select exactly one candidate image for detailed analysis from the provided candidate list.", "parameters": {"type": "object", "properties": {"target_images": {"type": "array", "description": "List containing exactly one candidate image index to select for detailed analysis (e.g., [3] to select candidate 3).", "items": {"type": "integer", "description": "Candidate image index from 1 to the number of candidates."}, "minItems": 1, "maxItems": 1}}, "required": ["target_images"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

system_prompt_original_01 ="""You are a helpful assistant for multimodal retrieval tasks. Your goal is to evaluate a query (which can be text, image, or both) against multiple candidate items (which can be text, image, or both) and select the best matching candidate.

## Step-by-Step Analysis Process
1. **Analyze All Text and Image Content**: 
First, thoroughly examine the textual and visual features of the query and each candidate. For text, read and understand the content, context, and key points. For images, observe objects, colors, layout, and any relevant details.
Compare the query to each candidate individually, noting similarities and differences.

2. **Assess Analysis Difficulty**:
Identify if any candidate image has areas that are hard to analyze due to small size, blurriness, or complex details. This may require using `crop_image` to zoom in for a closer look.
If multiple candidates are similarly matching or difficult to distinguish, consider using `select_images` to focus on a subset for detailed comparison.

3. **Decide on Tool Usage**:
Use `crop_image` only if zooming in on a specific image region would provide critical information not available in the initial view. This tool is best for single images with high analysis difficulty.
Use `select_images` only if you need to narrow down candidates for in-depth analysis, such as when visual features are subtle or multiple images require side-by-side inspection.
Tools should be employed sparingly; only when they significantly enhance your ability to evaluate the match.

4. **Perform Final Evaluation**:
After any tool usage, integrate the new insights into your analysis.
Select the candidate that best matches the query based on all available information.

## Tools 
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}
{"type": "function", "function": {"name": "select_images", "description": "Select specific candidate images for detailed analysis from the provided candidate list.", "parameters": {"type": "object", "properties": {"target_images": {"type": "array", "description": "List of candidate image indices to select for detailed analysis (e.g., [1, 3, 5] to select candidates 1, 3, and 5).", "items": {"type": "integer", "description": "Candidate image index from 1 to the number of candidates."}}}, "required": ["target_images"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>""

"""

system_prompt_original_02="""You are a helpful assistant for multimodal retrieval tasks. Your goal is to evaluate a query (which can be text, image, or both) against multiple candidate items (which can be text, image, or both) and select the best matching candidate.

## Step-by-Step Analysis Process
1. **Analyze All Text and Image Content**: 
First, thoroughly examine the textual and visual features of the query and each candidate. For text, read and understand the content, context, and key points. For images, observe objects, colors, layout, and any relevant details.
Compare the query to each candidate individually, noting similarities and differences.

2. **Identify Areas Needing Detailed Inspection**:
Actively use the available tools to get more detailed information when needed. Don't hesitate to call tools - they are there to help you see details that might be missed at first glance.
If any candidate image has areas that are hard to analyze due to small size, blurriness, or complex details, use `crop_image` to zoom in for a closer look.
If multiple candidates are similarly matching or difficult to distinguish, use `select_images` to focus on a subset for detailed comparison.

3. **Reflect on Tool Usage and Adjust Approach**:
After each tool call, carefully examine the results and reflect: "Wait a moment, did this tool call provide the information I expected? Was the bounding box correctly placed? Should I adjust my approach based on what I see now?"
If a tool call doesn't yield useful information, consider why and adjust your strategy: "Maybe I need to look at a different area, or perhaps the detail I'm seeking isn't visible even with zooming."

4. **Integrate Insights and Finalize Selection**:
After gathering detailed information through tools, integrate all insights into your analysis.
Select the candidate that best matches the query based on comprehensive examination of both original views and detailed inspections.

## Tools 
You may call one or more functions to assist with the user query. Use tools proactively to get the detailed information needed for accurate matching.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}
{"type": "function", "function": {"name": "select_images", "description": "Select specific candidate images for detailed analysis from the provided candidate list.", "parameters": {"type": "object", "properties": {"target_images": {"type": "array", "description": "List of candidate image indices to select for detailed analysis (e.g., [1, 3, 5] to select candidates 1, 3, and 5).", "items": {"type": "integer", "description": "Candidate image index from 1 to the number of candidates."}}}, "required": ["target_images"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""



guideline = """Guidelines: You are given a query (which can be text, image, or both) and multiple candidate items (which can be text, image, or both). Your task is to evaluate each candidate against the query and select the best match. Determine if it is beneficial to employ the given visual operations (tools). You can use the following tools:
1. `crop_image`: Zoom in on specific areas of the query image or selected candidate images for detailed analysis
2. `select_images`: Select specific candidate images from the provided list for detailed comparison

Analyze the visual and textual features step by step and provide your final answer in the format:
<think>Your reasoning process here</think><answer>Directly answer with the most matching number</answer>"""

guideline_original_01 = """Guidelines:Always begin by analyzing the query and all candidates in detail, covering both text and image components.
Use tools only when analysis is challenging and additional visual inspection would lead to a more accurate match.
After analysis, provide your reasoning in <think> tags and the final answer in <answer> tags. The answer should be the only number index of the most matching candidate (e.g., <think>Your reasoning process here</think><answer>1</answer>).
"""
guideline_original_02="""Guidelines:Always begin by analyzing the query and all candidates in detail, covering both text and image components.Proactively use tools to examine difficult areas - don't hesitate to call them when you need more detailed visual information.After each tool call, reflect on whether it provided useful information and adjust your approach if needed.Use self-reflection phrases like "Wait a moment" to check your reasoning and tool usage throughout the process.
After analysis, provide your reasoning in <think> tags and the final answer in <answer> tags. The answer should be the index of the most matching candidate (e.g., <think>Your reasoning process here</think><answer>1</answer>).
"""

def process_lamra_data(lamra_file_path: str, output_dir: str, filter_len=None):
    """
    处理LamRA数据，转换为verl-tool训练格式
    支持多模态检索：query和candidate都可能是text、image或text+image
    保留原始数据结构标记（Query:, Candidates:等）
    """
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载LamRA数据
    print(f"Loading data from {lamra_file_path}")
    with open(lamra_file_path, 'r', encoding='utf-8') as f:
        lamra_data = json.load(f)
    print(f"Loaded {len(lamra_data)} samples")

    processed_data = []
    
    print(f"Processing data...")
    for item in lamra_data:
        qid = item['qid']
        message = item['message']
        ground_truth_position = item['ground_truth_position']
        num_candidates = item['num_candidates']
        
        # 提取用户消息内容
        user_content = message[0]['content']
        
        # 直接重建完整的多模态内容，保留所有标记
        mm_content = ""
        query_image = None
        query_text = ""
        candidate_images = []
        candidate_texts = []
        
        in_query_section = False
        in_candidates_section = False
        current_candidate_idx = -1
        
        for content_item in user_content:
            if content_item['type'] == 'text' and content_item['text']:
                text = content_item['text']
                
                # 检测Query部分
                if 'Query:' in text:
                    in_query_section = True
                    in_candidates_section = False
                    mm_content += text  # 保留"Query:"
                    continue
                
                # 检测Candidates部分
                if 'Candidates:' in text:
                    in_query_section = False
                    in_candidates_section = True
                    mm_content += text
                    continue
                
                # 候选编号，如"(1) "
                if text.startswith('(') and ')' in text[:5]:
                    mm_content += text
                    current_candidate_idx += 1
                    continue
                
                # Query部分的文本
                if in_query_section:
                    query_text += text
                    mm_content += text
                    
                
                # Candidate的文本描述
                elif in_candidates_section and text and text.strip():
                    if current_candidate_idx >= 0:
                        if current_candidate_idx >= len(candidate_texts):
                            candidate_texts.append(text)
                        mm_content += text
                    
            elif content_item['type'] == 'image':
                # 添加图像占位符
                mm_content += "<image>"

                # 修改：根据当前section区分图像归属
                if in_query_section and query_image is None:
                    # query section的第一个图像赋给query_image
                    query_image = content_item['image']
                elif in_candidates_section:
                    # candidates section的图像追加到candidate_images
                    candidate_images.append(content_item['image'])
                else:
                    # 其他情况（非query或candidates），可记录警告或忽略
                    print(f"Warning: Image found outside query or candidates section for qid {qid}")
                    candidate_images.append(content_item['image'])  # 保守处理，追加到candidates

        mm_content += f"\n\n{guideline}"
        # 构建训练数据
        all_images = []
        if query_image:
            all_images.append({"image": query_image})
        all_images.extend([{"image": img} for img in candidate_images])
        
        data = {
            "data_source": "lamra_rerank",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user", 
                    "content": mm_content,
                }
            ],
            "images": all_images,
            "ability": "multimodal_reranking",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth_position,
            },
            "extra_info": {
                'qid': qid,
                'ground_truth_position': ground_truth_position,
                'num_candidates': num_candidates,
                'query_image': query_image,
                'query_text': query_text.strip(),
                'candidate_images': candidate_images,
                'candidate_texts': candidate_texts,
                'images': [query_image] + candidate_images if query_image else candidate_images,
            }
        }
        
        # if filter_len and filter_len > 0:
        #     mm_content_len = get_mm_content_len(processor, data)
        #     data['extra_info']['mm_content_len'] = mm_content_len
        
        processed_data.append(data)
        print(f"Processed {len(processed_data)} samples")
    
    with open(output_dir / "lamra_train_300K.json", "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    # 转换为datasets格式
    dataset = datasets.Dataset.from_list(processed_data)
    
    
    # # 过滤长度
    # if filter_len and filter_len > 0:
    #     dataset = dataset.filter(lambda x: x['extra_info']['mm_content_len'] <= filter_len)
    #     print(f"Filtered dataset to {len(dataset)} examples with content length <= {filter_len}")
    
    # 分割训练和验证集
    if len(dataset) > 1:
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        val_dataset = dataset_split['test']
    else:
        train_dataset = dataset
        val_dataset = dataset
    
    print(f"Processed {len(dataset)} total samples")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 保存数据
    train_dataset.to_parquet(output_dir / "lamra_train_300K.parquet")
    val_dataset.to_parquet(output_dir / "lamra_val_300K.parquet")
    
    print(f"Saved training data to {output_dir / 'lamra_train_300K.parquet'}")
    print(f"Saved validation data to {output_dir / 'lamra_val_300K.parquet'}")
    
    return train_dataset, val_dataset




def images_to_video(image_folder, output_path, fps=24):
    images = sorted(glob(os.path.join(image_folder, "*.jpg")))
    if not images:
        raise ValueError("No .jpg images found in folder.")

    # Read the first image to get size
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

def get_mm_content_len(processor, example):
    messages = deepcopy(example['prompt'])
    for message in messages:
        content = message["content"]
        content_list = []
        segments = re.split("(<image>|<video>)", content)
        segments = [item for item in segments if item != ""]
        segment_idx = defaultdict(int)
        for segment in segments:
            if segment == "<image>":
                content_list.append({"type": "image", "image": example['images'][segment_idx[segment]]["image"]})
                segment_idx[segment] += 1
            elif segment == "<video>":
                content_list.append({"type": "video", "video": example['videos'][segment_idx[segment]]["video"]})
                segment_idx[segment] += 1
            else:
                content_list.append({"type": "text", "text": segment})

        message["content"] = content_list
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[raw_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.input_ids.shape[1]

def main(
    lamra_file_path: str = '/remote-home1/hxzhuang/cdy/LamRA/data/rerank_messages_300K_quick.json',
    output_dir: str = 'data/pixel_reasoner/LamRA_RL_Data',
    filter_len: int = None,
    dataset_path: str = 'DongyangChen/pixelreasoner_rl_test',
    local_dir: str = 'data/pixel_reasoner',
    version: str = None,
    seed: int = 42,
    image_sep = "<image>",
    video_sep = "<video>",
    include_videos=False,
):
    """
    主函数:处理LamRA数据
    """
    train_dataset, val_dataset = process_lamra_data(lamra_file_path, output_dir, filter_len)
    print("Data processing completed successfully!")
    print(f"Example training sample:")
    print(train_dataset[0])
if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/data_preprocess/pixel_reasoner/prepare_train.py --dataset_path=DongyangChen/pixelreasoner_rl_test --local_dir=data/pixel_reasoner --version test --include_videos=False
"""