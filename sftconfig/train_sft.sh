#!/bin/bash
cd /vepfs_c/uiagent/sdz/GenAI_project/cdy/LLaMA-Factory
torchrun --nproc-per-node=8 --master-port=29502 \
  -m llamafactory.cli train examples/train_full/qwen2_5vl_retrv_full_sft.yaml