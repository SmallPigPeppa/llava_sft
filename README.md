# minimal_llava_trainer

这是从原始 LFactory 项目里压缩出来的最小训练项目，保留三件事：

1. 基于 Hugging Face `Trainer` 的训练主干；
2. ShareGPT/LLaVA 格式数据加载与图文样本组 batch 的核心逻辑；
3. 所有关键参数都放在 YAML 里控制。

默认配置对应你原来的 `train_llava15_lora_demo_2k.sh`：

```bash
WANDB_PROJECT=CL-debug python src/train.py examples/train_lora/llava15_lora_next_data.yaml \
  dataset_dir=data/llava_779k_demo \
  dataset=demo_2000 \
  output_dir=saves/llava-1.5-7b/lora/llava-779k-demo-2k \
  run_name=llava15-lora-779k-demo-2k \
  report_to=wandb \
  per_device_train_batch_size=1 \
  gradient_accumulation_steps=8 \
  num_train_epochs=1.0 \
  learning_rate=2e-4 \
  warmup_ratio=0.03 \
  val_size=0.001 \
  bf16=true \
  fp16=false
```

在这个精简版里，这些参数都进入了 `configs/demo2k.yaml`。

## 文件结构

```text
minimal_llava_trainer/
├── train.py              # 模型加载、LoRA、Trainer、W&B、是否保存模型
├── data.py               # dataset_info.json 读取、Parquet/JSON/HF 数据加载、ShareGPT->SFT 编码
├── configs/demo2k.yaml   # demo2k 的完整 YAML 配置
├── requirements.txt
├── run_demo2k.sh
└── README.md
```

## 安装

```bash
cd minimal_llava_trainer
pip install -r requirements.txt
```

## 运行 demo2k

确认 `configs/demo2k.yaml` 里的数据路径可访问：

```yaml
data:
  dataset_dir: data/llava_779k_demo
  dataset: demo_2000
```

其中 `data/llava_779k_demo/dataset_info.json` 需要包含类似：

```json
{
  "demo_2000": {
    "file_name": "/ppio_net0/datasets/parquet/llava_779k_demo_2000",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "image"
    }
  }
}
```

启动：

```bash
export WANDB_API_KEY=你的_key
./run_demo2k.sh
```

也可以临时覆盖 YAML 参数：

```bash
python train.py --config configs/demo2k.yaml \
  train.learning_rate=1e-4 \
  data.max_samples=64 \
  train.num_train_epochs=1
```

## W&B 与 checkpoint

已按你的要求处理：

```yaml
wandb_project: CL-debug
train:
  report_to: [wandb]
  save_checkpoint: false
  save_strategy: "no"
  save_model_at_end: false
```

这意味着：训练日志会上报到 W&B；Trainer 不保存中间 checkpoint；训练结束也不额外保存 LoRA adapter。如果之后需要保存最终 LoRA 权重，把 `save_model_at_end` 改成 `true` 即可。

## 这个精简版删掉了什么

为了保持最小化，删掉了原 LLaMA-Factory/LFactory 中的大量通用能力，例如 WebUI、多训练阶段、DPO/RM/PPO、复杂模板库、多模态视频/音频、packing、4D attention mask、量化导出、DeepSpeed/FSDP 专用保存逻辑等。

当前脚本聚焦：LLaVA-1.5 + ShareGPT 图文 SFT + LoRA + HF Trainer + W&B。

## 数据格式要求

默认支持 ShareGPT/LLaVA 风格：

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is in the image?"},
    {"from": "gpt", "value": "..."}
  ],
  "image": {"bytes": "...", "path": "xxx.jpg"}
}
```

`image` 也可以是 PIL image、bytes、路径字符串、`{"bytes": ..., "path": ...}`，或者它们的列表。路径类图片可通过 `data.media_dir` 指定根目录。

## 注意

`image_seq_len` 默认会从 processor/model 里推断。对 `llava-hf/llava-1.5-7b-hf` 通常是 576。如果你更换模型后遇到 image token 数与 image feature 数不匹配，可以在 YAML 中手动设置：

```yaml
data:
  image_seq_len: 576
```
