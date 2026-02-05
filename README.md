# V-Retrver: Evidence-Driven Agentic Reasoning for Universal Multimodal Retrieval

[[📖 Paper]()] [[🤗 V-Retrver-7B-model](https://huggingface.co/V-Retrver/V-Retrver-7B)] [🤗[V-Retrver-RFT-model](https://huggingface.co/V-Retrver/V-Retrver-RFT-7B)] [🤗[V-Retrver-SFT-model](https://huggingface.co/V-Retrver/V-Retrver-SFT-7B)] [[🤗 V-Retrver-train-data](https://huggingface.co/datasets/V-Retrver/V-Retrver-train-data)] [🤗 [V-Retrver-eval-data](https://huggingface.co/datasets/V-Retrver/V-Retrver-eval-data)]



## 👀 About V-Retrver

<div align="center">
  <img src="./assets/intro.png" alt="Descriptive alt text" width="95%">
</div>

We introduce **V-Retrver**, an evidence-driven retrieval framework that reformulates multimodal retrieval as an agentic reasoning process grounded in visual inspection. V-Retrver enables an MLLM to selectively acquire visual evidence during reasoning via external visual tools, performing a **multimodal interleaved reasoning** process that alternates between hypothesis generation and targeted visual verification. 

To train such an evidence-gathering retrieval agent, we adopt a curriculum-based learning strategy combining **supervised reasoning activation, rejection-based refinement, and reinforcement learning** with an evidence-aligned objective. 

Experiments across multiple multimodal retrieval benchmarks demonstrate consistent improvements in retrieval accuracy **(with 23.0\% improvements on average)**, perception-driven reasoning reliability, and generalization.

All code, models, and data are fully released.



## 🔥 News
- [2026/2/06] We release the code, model, data of V-Retrver

## 📍 Features

+ Support Qwen3-VL/Qwen2.5-VL Training
+ Provide full pipeline (dataset, SFT training, RFT training, RL training, evaluation, etc) 


## 🏆 Performance

V-Retrver-7B demonstrates strong performance across multiple multimodal retrieval benchmarks.



<div align="center">
  <img src="./assets/tab4.png" alt="Descriptive alt text" width="90%">
</div>



## 🎥 Reasoning Examples

 Some reasoning examples are as follows.

<div align="center">
  <img src="assets/exp1.png" width="80%">
</div>
<div align="center">
  <img src="assets/exp2.png" width="80%">
</div>
<div align="center">
  <img src="assets/exp3.png" width="80%">
</div>


## 📐 Set up
```
cd verltool
git submodule update --init --recursive
conda create --name v-retrver python=3.10
conda activate v-retrver
pip install -e verl
pip install -e ".[vllm,acecoder,torl,search_tool]"
pip install "flash-attn==2.8.3" --no-build-isolation
```



## 🚀 Training
### Stage 1: Cold-start Supervised Fine-tuning (SFT)

We recommend to use the popular [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to perform SFT on our cold-start data.
1. Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
2. Follow the instructions in LLaMA-Factory to configure the cold-start data in `data/dataset_info.json`, as shown below, then copy the config file `sftconfig/qwen2_5vl_retrv_full_sft.yaml` into your LLaMA-Factory codebase.
```
"V-Retrver_SFT": {
  "file_name": "[YOUR_DATASET_FOLDER]/V-Retrver_SFT.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "images": "images"
  },
  "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "system_tag": "system"
  }
}
```
4. Train Cold-start data with the training configs.
```
llamafactory-cli train sft_configs/qwen2_5vl_retrv_full_sft.yaml
```
### Stage 2: Rejection Sampling Fine-Tuning (RSFT)
In this stage, we improve reasoning reliability through Rejection Sampling.The training process and configurations for this stage are identical to Stage 1 (SFT). You simply need to prepare the RSFT dataset and follow the same training steps described in Stage 1.
### Stage 3: Reinforcement Learning (RL)
#### Training
The reinforcement learning is based on the RSFT model. You could either use the model produced in stage 1, or directly download it from [V-Retrver/V-Retrver-RFT-7B](https://huggingface.co/V-Retrver/V-Retrver-RFT-7B). 
```
cd verltool
bash examples/train/v-retrver/train_qwen25vl.sh
```
It should be able to run under 8 A800 GPUs with 80GB memory. From more details，please refer to [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool).

Tips:
- if output shared memory, try lower the `data.dataloader_num_workers`
- if out of cuda memory during vllm rollout, try set `actor_rollout_ref.rollout.enforce_eager=True`, might be slower.
- if out of cuda memory during training, try lower the `use_dynamic_bs=False`.



## 🔮 Inference & Evaluation
We recommend using our provided json files and scripts for easier evaluation. 

The json files can be downloaded at: [🤗 [V-Retrver-eval-data](https://huggingface.co/datasets/V-Retrver/V-Retrver-eval-data)].

You can conduct inference on all benchmarks using the following scripts
```
cd verltool
bash examples/train/AdaTooler-V/eval.sh
```
### Sliding Window Evaluation
Use `sliding_window_processor.py` for reranking over a large candidate pool. This script splits the input Parquet file into specific windows (default strategy: 30-49, 20-39, etc.), calls the core eval.sh for each window, and merges the results to update the final rankings.
```
cd verltool
python sliding_window_processor.py \
  --input_parquet /path/to/your/input_data.parquet \
  --output_parquet /path/to/save/reranked_data.parquet \
  --window_size 20
```
### Top-20 Direct Evaluation
Use `rerank_top20_processor.py` (or the corresponding mode) to directly evaluate and rerank the top-20 candidates, skipping the sliding window logic. This script also wraps the core eval.sh to perform the evaluation.
```
cd verltool
python rerank_top20_processor.py \
  --input_parquet /path/to/your/input_data.parquet \
  --output_parquet /path/to/save/top20_reranked.parquet \
  --mode top20
```


## Acknowledgements

We sincerely appreciate the contributions of the open-source community. The related projects are as follows: [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool), [verl](https://github.com/volcengine/verl),  [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [LamRA](https://github.com/Code-kunkun/LamRA)
## Citations

If you find our work helpful for your research, please consider citing our work.   

```

```
