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
2. Use the script `scripts/preprocess_coldstart.py` to download [AdaTooler-V-train-data](https://huggingface.co/datasets/AdaTooler-V/AdaTooler-V-train-data) and produce the required data format by LLaMA-Factory. This script automatically extracts images and generates a JSON file from the original parquet-format dataset.
```
python3 scripts/preprocess_coldstart.py --dataset_path AdaTooler-V/AdaTooler-V-train-data --output_dir [YOUR_DATASET_FOLDER]
```
3. After processing, please follow the instructions in LLaMA-Factory to configure the cold-start data in `data/dataset_info.json`, as shown below, then copy the config file `sft_configs/qwen2.5-vl.yaml` into your LLaMA-Factory codebase.
```
"AdaTooler-V-CoT-100k": {
  "file_name": "[YOUR_DATASET_FOLDER]/AdaTooler-V-CoT-100k.json",
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
llamafactory-cli train sft_configs/qwen2.5-vl.yaml
```

### Stage 2: Reinforcement Learning (RL)
#### Data Preprocessing
We extract the video data into a multi-frame(64 frames), which can be directly obtained from [AdaTooler-V-train-data](https://huggingface.co/datasets/AdaTooler-V/AdaTooler-V-300k). 

We also provide the raw video data. If you would like to customize the number of video frames used for training, you can refer to the code in `scripts/extact_frames.py` to implement this yourself.
```
cd verltool
python examples/data_preprocess/pixel_reasoner/prepare_train.py --dataset_path=AdaTooler-V/AdaTooler-V-300k --local_dir=data/AdaTooler-V --version max_8192 --include_videos=True --filter_len=8192
```
note that the data preprocessing step will **filter out samples whose length exceeds 8192 tokens**, and this process may take some time to complete (approximately **0.5–1 hour**). If you prefer not to apply this filtering, you can remove the `--filter_len` argument. However, be aware that some samples are longer than 8192 tokens, which **may cause issues during training**. Therefore, if filtering is disabled, please ensure that the `max_prompt_length` is **properly configured during training** to avoid potential problems.


#### Training
The reinforcement learning is based on the cold-start model. You could either use the model produced in stage 1, or directly download it from [AdaTooler-V-SFT-model](https://huggingface.co/AdaTooler-V/AdaTooler-V-SFT-model). 
```
cd verltool
bash examples/train/AdaTooler-V/train_qwen25vl.sh
```
It should be able to run under 8 H100/A100 GPUs with 80GB memory. From more details，please refer to [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool).

Tips:
- if output shared memory, try lower the `data.dataloader_num_workers`
- if out of cuda memory during vllm rollout, try set `actor_rollout_ref.rollout.enforce_eager=True`, might be slower.
- if out of cuda memory during training, try lower the `use_dynamic_bs=False`.



## 🔮 Inference & Evaluation




## Acknowledgements

We sincerely appreciate the contributions of the open-source community. The related projects are as follows: [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool), [verl](https://github.com/volcengine/verl),  [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [LamRA](https://github.com/Code-kunkun/LamRA)
## Citations

If you find our work helpful for your research, please consider citing our work.   

```

```
