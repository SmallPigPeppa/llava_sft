# 极简版 LLaVA 1.5 7B SFT

只保留 3 个核心动作：

1. 读 `mllm_demo.json`
2. 按 LLaVA chat template 拼 prompt
3. 用 **Lightning Trainer** 训练

## 文件

- `train.py`：唯一训练脚本
- `config.yaml`：全部参数都在这里

## 数据格式

这个脚本就是按 LLaMA-Factory 的 `mllm_demo.json` 写的：

```json
[
  {
    "messages": [
      {"role": "user", "content": "<image>Who are they?"},
      {"role": "assistant", "content": "They're ..."}
    ],
    "images": ["mllm_demo_data/1.jpg"]
  }
]
```

## 运行

单机多卡：

```bash
torchrun --nproc_per_node=8 train.py --config config.yaml
```

单卡：

```bash
python train.py --config config.yaml
```

## 说明

- `config.yaml` 的字段名和含义保持不变，只是把 HuggingFace `Trainer` 换成了 Lightning `Trainer`。
- 为了兼容 `mllm_demo.json` 这种**一个样本里可能有多张图**的格式，极简版默认建议：
  - `per_device_train_batch_size: 1`
  - 用 `gradient_accumulation_steps` 把总 batch 顶上去
- 默认是 **LoRA**，更省显存；如果你要全参微调，把 `use_lora: false`。
- 这个脚本会把一条多轮对话自动展开成多个训练样本：
  - 每个 assistant 回复都会变成一条 SFT 样本
  - loss 只打在当前 assistant 回复上
- Lightning 训练中间 checkpoint 会保存在：
  - `train.output_dir/checkpoints/*.ckpt`
- 训练结束后仍然会把 HF/PEFT 权重和 processor 保存到：
  - `train.output_dir`
- `train.resume_from_checkpoint` 现在优先接收 Lightning 的 `.ckpt` 路径；如果直接给输出目录，也会自动尝试找 `last.ckpt`。
- 如果你的环境支持 deepspeed，可以直接把 `train.deepspeed` 改成 ds json 路径。
