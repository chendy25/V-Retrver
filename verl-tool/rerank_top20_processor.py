#!/usr/bin/env python3
"""
Sliding-window reranking processor - supports direct processing of top 20 candidates.
Based on existing functions, directly evaluate top 20 candidates.
"""

import json
import os
import copy
from pathlib import Path
from datasets import load_dataset, Dataset
import pandas as pd
from typing import List, Dict, Any
import argparse
import re
import subprocess
import tempfile
import time

# import existing functions
from sliding_window_processor import (
    create_window_data_from_hit,
    create_window_parquet_from_samples,
    run_verltool_eval,
    find_results_file,
    parse_verltool_result,
    update_hit_ranking_from_window_results,
    cleanup_ray_processes
)

def process_top20_direct_rerank(input_parquet: str, output_parquet: str) -> bool:
    """
    Directly process top 20 candidates using functions from sliding_window_processor.
    Skip sliding-window logic and evaluate top 20 candidates for each sample.
    """
    
    print("="*60)
    print("DIRECT TOP20 CANDIDATE RERANKING")
    print("="*60)
    
    # Load original dataset
    original_dataset = load_dataset("parquet", data_files=input_parquet)["train"]
    updated_samples = list(original_dataset)
    
    # Get dataset name for filenames and run names
    dataset_name = Path(input_parquet).stem
    
    print(f"Starting direct top20 reranking for {len(updated_samples)} samples")
    print(f"Dataset: {dataset_name}")
    print(f"Output will be saved to: {output_parquet}")
    
    try:
        # 1. Create window data for the top 20 candidates
        # Use window_start=0 and window_size=20 to get the top 20 candidates
        temp_dir = "/vepfs_c/uiagent/sdz/GenAI_project/cdy/verl-tool/data/RetrvTool_Eval_Data/windows_data"
        
        print("\nStep 1: Creating top20 window data...")
        window_parquet = create_window_parquet_from_samples(
            updated_samples, window_start=0, output_dir=temp_dir, 
            window_size=20, dataset_name=dataset_name
        )
        
        if window_parquet is None:
            print("ERROR: Failed to create top20 window parquet")
            return False
        
        # 2. Run VerlTool evaluation
        run_name = f"top20_{dataset_name}"
        
        print("\nStep 2: Running VerlTool evaluation...")
        success = run_verltool_eval(window_parquet, run_name)
        
        if not success:
            print("ERROR: VerlTool evaluation failed for top20 candidates")
            return False
        
        # 3. Parse evaluation results
        print("\nStep 3: Parsing evaluation results...")
        results_file = find_results_file(run_name)
        if results_file:
            print(f"Parsing results from {results_file}")
            window_results = parse_verltool_result(results_file)
            print(f"Found {len(window_results)} result entries")
            
            # 4. Update samples' hit rankings
            print("\nStep 4: Updating sample rankings...")
            updated_count = 0
            for i, sample in enumerate(updated_samples):
                original_sample = updated_samples[i]
                updated_sample = update_hit_ranking_from_window_results(
                    original_sample, window_results, window_start=0, window_size=20
                )
                if updated_sample != original_sample:
                    updated_samples[i] = updated_sample
                    updated_count += 1
            
            print(f"Updated ranking for {updated_count} samples")
        else:
            print("Warning: No results file found for top20 evaluation")
        
        # 5. Save final results
        print("\nStep 5: Saving final results...")
        final_dataset = Dataset.from_list(updated_samples)
        final_dataset.to_parquet(output_parquet)
        print(f"\n{'='*60}")
        print(f"SUCCESS: Saved final reranked results to: {output_parquet}")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Exception during direct top20 processing: {e}")
        return False
        
    finally:
        # Final cleanup
        print("\nFinal cleanup of Ray processes...")
        cleanup_ray_processes()

# Modify main to add new processing options
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sliding Window Reranking with VerlTool')
    parser.add_argument('--input_parquet', required=True, help='Input parquet file')
    parser.add_argument('--output_parquet', required=True, help='Output parquet file')
    parser.add_argument('--window_size', type=int, default=20, help='Window size')
    parser.add_argument('--mode', choices=['sliding', 'top20'], default='sliding', 
                       help='Processing mode: sliding (default) or top20')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'top20':
            # Directly process top 20 candidates
            success = process_top20_direct_rerank(
                args.input_parquet,
                args.output_parquet
            )
        else:
            # Original sliding-window processing
            from sliding_window_processor import process_sliding_window_rerank
            success = process_sliding_window_rerank(
                args.input_parquet,
                args.output_parquet, 
                args.window_size
            )
            
        if not success:
            print("ERROR: Processing failed")
            exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)