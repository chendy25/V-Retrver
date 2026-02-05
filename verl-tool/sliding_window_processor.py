import json
import os
import copy
from pathlib import Path
from datasets import load_dataset, Dataset
import pandas as pd
from typing import List, Dict, Any
import argparse
import re

def regenerate_window_prompt_from_hits(original_prompt: str, window_candidate_texts: List[str], window_candidates: List[str], window_size: int = 20) -> str:
    """
    Regenerate prompt using candidate_texts and candidate_images within the window.
    Args:
        original_prompt: original prompt string
        window_candidate_texts: list of candidate texts in the window
        window_candidates: list of candidate image paths in the window
        window_size: window size
    Returns:
        regenerated prompt
    """
    import re

    candidates_pattern = r'(Candidates:.*?)Guidelines:'
    match = re.search(candidates_pattern, original_prompt, re.DOTALL)

    if not match:
        print("Warning: Could not find Candidates section in prompt")
        return original_prompt

    candidates_section = match.group(1)

    new_candidates_parts = []
    for i in range(window_size):
        candidate_num = i + 1

        if window_candidate_texts:
            candidate_text = window_candidate_texts[i] if i < len(window_candidate_texts) else ""
        else:
            candidate_text = ""

        if window_candidates:
            new_candidates_parts.append(f"({candidate_num}) <image>{candidate_text}")
        else:
            new_candidates_parts.append(f"({candidate_num}) {candidate_text}")

    new_candidates_section = "Candidates:" + "".join(new_candidates_parts) + "\n\n"
    new_prompt = original_prompt.replace(candidates_section, new_candidates_section)
    return new_prompt



def create_window_data_from_hit(sample: Dict, window_start: int, window_size: int = 20) -> Dict:
    """
    Create a windowed sample from hit entries and regenerate the prompt.
    """
    window_sample = copy.deepcopy(sample)

    hit_items = sample.get('hit', [])
    if not hit_items:
        return window_sample

    window_end = min(window_start + window_size, len(hit_items))
    if window_start >= len(hit_items):
        return window_sample

    window_hits = hit_items[window_start:window_end]

    window_candidates = []
    window_candidate_texts = []

    for hit in window_hits:
        candidate_image = hit.get('candidate_images', '')
        candidate_text = hit.get('candidate_texts', '')
        if candidate_image:
            window_candidates.append(candidate_image)
        if candidate_text:
            window_candidate_texts.append(candidate_text)

    extra_info = sample.get('extra_info', {})
    query_image = extra_info.get('query_image') or sample.get('query_image')
    query_text = extra_info.get('query_text') or sample.get('query_text')

    window_images = []
    window_extra_images = []

    if query_image and query_image.strip():
        window_images.append({"image": query_image.strip()})
        window_extra_images.append(query_image.strip())
    for candidate_image in window_candidates:
        window_images.append({"image": candidate_image})
        window_extra_images.append(candidate_image.strip())

    original_prompt = sample.get('prompt', '')
    if isinstance(original_prompt, list) and original_prompt:
        window_prompt = []
        for msg in original_prompt:
            if msg.get('role') == 'user' and 'content' in msg:
                new_msg = msg.copy()
                new_msg['content'] = regenerate_window_prompt_from_hits(msg['content'], window_candidate_texts, window_candidates, window_size)
                window_prompt.append(new_msg)
            else:
                window_prompt.append(msg.copy())
    else:
        window_prompt = regenerate_window_prompt_from_hits(str(original_prompt), window_candidate_texts, window_candidates, window_size)

    window_extra_info = copy.deepcopy(extra_info)
    window_extra_info['candidate_images'] = window_candidates
    window_extra_info['candidate_texts'] = window_candidate_texts
    window_extra_info['num_candidates'] = max(len(window_candidates), len(window_candidate_texts))
    window_extra_info['window_start'] = window_start
    window_extra_info['window_size'] = window_size
    window_extra_info['qid'] = extra_info.get('qid', sample.get('qid', 'unknown'))
    window_extra_info['images'] = window_extra_images
    if query_image:
        window_extra_info['query_image'] = query_image
    if query_text:
        window_extra_info['query_text'] = query_text

    window_sample['images'] = window_images
    window_sample['extra_info'] = window_extra_info
    window_sample['prompt'] = window_prompt
    return window_sample

def create_window_parquet_from_samples(samples: List[Dict], window_start: int, 
                                      output_dir: str, window_size: int = 20,
                                      dataset_name: str = "") -> str:
    """
    Create a parquet file containing windowed samples.
    """
    window_data = []
    for sample in samples:
        window_sample = create_window_data_from_hit(sample, window_start, window_size)
        window_data.append(window_sample)

    window_dataset = Dataset.from_list(window_data)

    os.makedirs(output_dir, exist_ok=True)
    safe_dataset_name = dataset_name.replace('_', '-').replace('/', '-') if dataset_name else "unknown"
    output_file = os.path.join(output_dir, f"{safe_dataset_name}_window_{window_start}_{window_start+window_size-1}.parquet")
    window_dataset.to_parquet(output_file)

    print(f"Created window parquet: {output_file} with {len(window_data)} samples")
    return output_file

def parse_verltool_result(json_file: str) -> List[Dict]:
    """
    Parse VerlTool evaluation JSON file and return a list of result dicts.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            print(f"Warning: {json_file} is empty")
            return []

        results = json.loads(content)
        if isinstance(results, list):
            return results
        else:
            print(f"Warning: Expected JSON array, got {type(results)}")
            return [results] if results else []

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_file}: {e}")
        return []
    except FileNotFoundError:
        print(f"File not found: {json_file}")
        return []
    except Exception as e:
        print(f"Unexpected error parsing {json_file}: {e}")
        return []


def update_hit_ranking_from_window_results(sample: Dict, window_results: List[Dict], 
                                         window_start: int, window_size: int = 20) -> Dict:
    """
    Update sample's hit ranking based on windowed results with robust validation and error handling.
    """
    updated_sample = copy.deepcopy(sample)

    if window_size <= 0:
        print(f"Error: Invalid window_size {window_size} for qid {sample.get('extra_info', {}).get('qid', sample.get('qid', 'unknown'))}")
        return updated_sample

    if not isinstance(window_size, int):
        print(f"Error: window_size must be integer, got {type(window_size)}")
        return updated_sample

    original_qid = sample.get('extra_info', {}).get('qid') or sample.get('qid', '')
    window_result = None
    for result in window_results:
        result_qid = result.get('extra_info', {}).get('qid', '')
        if result_qid == original_qid:
            window_result = result
            break

    if window_result is None:
        print(f"Warning: No result found for qid {original_qid}")
        return updated_sample

    response = window_result.get('response', '')
    ranking_match = re.search(r'<answer>\s*\[([^\]]+)\]\s*</answer>', response)
    if not ranking_match:
        print(f"Warning: No ranking found in response for qid {original_qid}")
        return updated_sample

    ranking_str = ranking_match.group(1)
    ranking = []
    for num_str in ranking_str.split(','):
        num_str = num_str.strip()
        try:
            ranking.append(int(num_str) - 1)
        except ValueError:
            continue

    if not ranking:
        print(f"Warning: Empty ranking after parsing for qid {original_qid}")
        ranking = list(range(window_size))

    try:
        ranking = list(dict.fromkeys(ranking))
    except TypeError as e:
        print(f"Warning: Unable to deduplicate ranking due to non-hashable elements for qid {original_qid}: {e}")
        seen = set()
        ranking = [x for x in ranking if not (x in seen or seen.add(x))]

    valid_ranking = [idx for idx in ranking if 0 <= idx < window_size]
    if not valid_ranking:
        print(f"Warning: All ranking elements filtered out for qid {original_qid}")
        valid_ranking = list(range(window_size))

    if not all(isinstance(idx, int) for idx in valid_ranking):
        print(f"Warning: Non-integer indices found in ranking for qid {original_qid}")
        valid_ranking = [idx for idx in valid_ranking if isinstance(idx, int)]
        if not valid_ranking:
            valid_ranking = list(range(window_size))

    if len(valid_ranking) < window_size:
        try:
            used_indices = set(valid_ranking)
        except TypeError as e:
            print(f"Warning: Unable to create set from valid_ranking for qid {original_qid}: {e}")
            used_indices = []
            for idx in valid_ranking:
                if idx not in used_indices:
                    used_indices.append(idx)
            used_indices = set(used_indices)

        all_possible = set(range(window_size))
        missing = all_possible - used_indices
        valid_ranking.extend(sorted(missing))

    if len(valid_ranking) > window_size:
        print(f"Warning: Ranking length {len(valid_ranking)} exceeds window_size {window_size} for qid {original_qid}")
        truncated = valid_ranking[window_size:]
        print(f"Truncated elements: {truncated}")

    ranking = valid_ranking[:window_size]

    if len(ranking) != window_size:
        print(f"Error: Final ranking length {len(ranking)} != window_size {window_size} for qid {original_qid}")
        print(f"Original ranking_str: {ranking_str}")
        print(f"Parsed ranking: {ranking}")
        return updated_sample

    expected_indices = set(range(window_size))
    actual_indices = set(ranking)
    if expected_indices != actual_indices:
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices
        print(f"Warning: Ranking permutation incomplete for qid {original_qid}")
        if missing:
            print(f"Missing indices: {sorted(missing)}")
        if extra:
            print(f"Extra indices: {sorted(extra)}")

    hit_items = updated_sample.get('hit', [])
    if window_start + window_size > len(hit_items):
        print(f"Warning: Window range exceeds hit items for qid {original_qid}")
        return updated_sample

    window_hits = hit_items[window_start:window_start + window_size]
    reordered_hits = []
    for rank_idx in ranking:
        if 0 <= rank_idx < len(window_hits):
            reordered_hits.append(window_hits[rank_idx])

    if len(reordered_hits) != window_size:
        print(f"Warning: Reordered hits length {len(reordered_hits)} != window size {window_size}")
        return updated_sample

    updated_sample['hit'][window_start:window_start + window_size] = reordered_hits
    return updated_sample

def run_verltool_eval(window_parquet: str, run_name: str) -> bool:
    """
    Run VerlTool evaluation script with improved process handling and log error detection.
    """
    import subprocess
    import tempfile
    import time
    
    # VerlTool project root
    verltool_root = "./verl-tool"
    eval_script = os.path.join(verltool_root, "examples/train/v_retrver/eval.sh")
    
    # Create log directory and log file
    log_dir = "./verl-tool/data/V_Retrver_eval_data/log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"eval_{run_name}_{int(time.time())}.log")
    
    # Create a temporary script file in the log directory
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, 
                                   prefix='verl_eval_', dir=log_dir) as temp_file:
        temp_script = temp_file.name
    
    try:
        # Read original script
        with open(eval_script, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # Replace dataset paths
        script_content = script_content.replace(
            'train_data=[$(pwd)/data/${dataset_name}/mbeir_cirr_task7_test.parquet]',
            f'train_data=[{window_parquet}]'
        )
        script_content = script_content.replace(
            'val_data=[$(pwd)/data/${dataset_name}/mbeir_cirr_task7_test.parquet]', 
            f'val_data=[{window_parquet}]'
        )
        
        # Set experiment/run name
        script_content = script_content.replace(
            'run_name_postfix="cirr"',
            f'run_name_postfix="{run_name}"'
        )
        
        # Ensure script has correct shebang
        if not script_content.startswith('#!/'):
            script_content = '#!/bin/bash\n' + script_content
        
        # Write temporary script
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Set execute permission
        os.chmod(temp_script, 0o755)
        
        # Validate script accessibility
        if not os.path.exists(temp_script) or not os.access(temp_script, os.R_OK | os.X_OK):
            print(f"Error: Script file is not accessible: {temp_script}")
            return False
        
        print(f"Running VerlTool eval for {window_parquet}")
        print(f"Run name: {run_name}")
        print(f"Log file: {log_file}")
        
        # Use tee to pipe output to log file
        cmd = f'bash {temp_script} 2>&1 | tee {log_file}'
        
        print(f"Command: {cmd}")
        print(f"Working directory: {verltool_root}")
        print("-" * 60)
        
        # Run the script and capture output
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            cwd=verltool_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect output without printing to terminal
        full_output = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Append to log buffer
                full_output += output
        
        # Get return code
        return_code = process.poll()
        
        # Save full output to log file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(full_output)
        
        print("-" * 60)
        
        # Check return code
        script_failed = return_code != 0
        
        # Additionally scan the log file for error patterns (e.g., Ray background errors)
        log_has_errors = False
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
                # Check specific error patterns to avoid false positives
                error_patterns = [
                    r'IndexError:', r'KeyError:', r'ValueError:', r'TypeError:', r'AttributeError:',
                    r'Exception:', r'ERROR:', r'Error:', r'ray\.exceptions', r'Traceback \(most recent call last\):',
                    r'CRITICAL:', r'Failed to', r'failed to'
                ]
                for pattern in error_patterns:
                    if re.search(pattern, log_content, re.IGNORECASE):
                        log_has_errors = True
                        break
        
        if script_failed or log_has_errors:
            print(f"Error: Script failed with return code {return_code}")
            if log_has_errors:
                print("Error: Detected errors in log file")
            print(f"Check log file: {log_file}")
            return False
        else:
            print("VerlTool evaluation completed successfully")
            print(f"Full log saved to: {log_file}")
            return True
            
    except Exception as e:
        print(f"Error in run_verltool_eval: {e}")
        return False
        
    finally:
        # Cleanup temporary script file
        try:
            if os.path.exists(temp_script):
                os.unlink(temp_script)
        except Exception as e:
            print(f"Warning: Could not clean up {temp_script}: {e}")

def cleanup_ray_processes():
    """
    Cleanup leftover Ray processes and related servers.
    """
    import subprocess
    try:
        # Stop Ray gracefully
        result = subprocess.run(['ray', 'stop'], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("Successfully stopped Ray processes")
        else:
            print(f"Warning: ray stop returned {result.returncode}")

        # Force-kill any remaining processes
        subprocess.run(['pkill', '-f', 'ray'], capture_output=True)
        subprocess.run(['pkill', '-f', 'verl_tool.servers.serve'], capture_output=True)

    except Exception as e:
        print(f"Warning: Could not cleanup ray processes: {e}")

def find_results_file(run_name: str) -> str:
    """
    Search for a results JSON file under VerlTool's standard output directory.
    VerlTool outputs to verl_step_records/<run_name>.
    """
    verltool_root = "./verl-tool"
    results_base_dir = os.path.join(verltool_root, "verl_step_records", run_name)

    print(f"Looking for results in: {results_base_dir}")

    # Search for JSON files
    if os.path.exists(results_base_dir):
        for root, dirs, files in os.walk(results_base_dir):
            for file in files:
                if file.endswith('.json'):
                    result_file = os.path.join(root, file)
                    print(f"Found results file: {result_file}")
                    return result_file

    print(f"No JSON files found in {results_base_dir}")
    return None

def process_sliding_window_rerank(input_parquet: str, output_parquet: str, window_size: int = 20):
    """
    Execute the full sliding-window reranking pipeline.
    """

    # Define window order (process later windows first)
    windows = [
        (30, window_size),  # 30-49
        (20, window_size),  # 20-39  
        (10, window_size),  # 10-29
        (0, window_size),   # 0-19
    ]
    
    # Load input dataset
    original_dataset = load_dataset("parquet", data_files=input_parquet)["train"]
    updated_samples = list(original_dataset)
    
    # Get dataset name for file/run naming
    dataset_name = Path(input_parquet).stem
    
    print(f"Starting sliding window reranking for {len(updated_samples)} samples")
    print(f"Dataset: {dataset_name}")
    print(f"Output will be saved to: {output_parquet}")
    
    try:
        for window_idx, (window_start, window_size) in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"Processing window {window_idx + 1}/{len(windows)}: {window_start}-{window_start+window_size-1}")
            print(f"{'='*60}")
            
            # 1. Create window parquet from current updated samples
            temp_dir = "./verl-tool/data/V_Retrver_eval_data/windows_data"
            window_parquet = create_window_parquet_from_samples(
                updated_samples, window_start, temp_dir, window_size, dataset_name
            )
            
            if window_parquet is None:
                print(f"Warning: Failed to create window parquet for {window_start}")
                continue
            
            # 2. Run VerlTool evaluation for this window
            window_run_name = f"window_{dataset_name}_{window_start}_{window_start+window_size-1}"
            
            print(f"Running VerlTool evaluation...")
            success = run_verltool_eval(window_parquet, window_run_name)
            
            if not success:
                print(f"ERROR: VerlTool evaluation failed for window {window_start}")
                print(f"Stopping the entire process due to window failure")
                raise RuntimeError(f"VerlTool evaluation failed for window {window_start}. Check logs for details.")
            
            # 3. Parse evaluation results
            results_file = find_results_file(window_run_name)
            if results_file:
                print(f"Parsing results from {results_file}")
                window_results = parse_verltool_result(results_file)
                print(f"Found {len(window_results)} result entries")
                
                # 4. Update samples' hit ranking based on window results
                updated_count = 0
                for i, sample in enumerate(updated_samples):
                    original_sample = updated_samples[i]
                    updated_sample = update_hit_ranking_from_window_results(
                        original_sample, window_results, window_start, window_size
                    )
                    if updated_sample != original_sample:
                        updated_samples[i] = updated_sample
                        updated_count += 1
                
                print(f"Updated ranking for {updated_count} samples in window {window_start}")
            else:
                print(f"Warning: No results file found for window {window_start}")
            
            # 5. Cleanup Ray processes after each window
            print(f"Cleaning up Ray processes after window {window_start}...")
            cleanup_ray_processes()
            print(f"Ray cleanup completed for window {window_start}")
        
        # Save final results
        final_dataset = Dataset.from_list(updated_samples)
        final_dataset.to_parquet(output_parquet)
        print(f"\n{'='*60}")
        print(f"SUCCESS: Saved final reranked results to: {output_parquet}")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Exception during processing: {e}")
        return False
        
    finally:
        # Final cleanup (just in case)
        print("\nFinal cleanup of Ray processes...")
        cleanup_ray_processes()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sliding Window Reranking with VerlTool')
    parser.add_argument('--input_parquet', required=True, help='Input parquet file')
    parser.add_argument('--output_parquet', required=True, help='Output parquet file')
    parser.add_argument('--window_size', type=int, default=20, help='Window size')
    
    args = parser.parse_args()
    
    try:
        success = process_sliding_window_rerank(
            args.input_parquet,
            args.output_parquet, 
            args.window_size
        )
        if not success:
            print("ERROR: Sliding window reranking failed")
            exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
