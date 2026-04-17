# Minimal LLaVA-1.5-7B LoRA SFT (Lightning)

这是一个**最小可跑**版本，特点：

- 模型：`llava-hf/llava-1.5-7b-hf`
- 训练：PyTorch Lightning
- 精度：YAML 控制（默认 `16-mixed` + `torch_dtype=float16`）
- LoRA：**只作用在语言模型 `model.language_model`**
- 冻结：`vision_tower` 和 `multi_modal_projector` 全冻结
- 数据：直接读取你这种 `messages + images` 的 JSON
- 多轮：把每个 assistant 回复展开成一个 SFT 样本

## 1. 安装

```bash
pip install -r requirements.txt
```

## 2. 训练

```bash
python train.py --config config.yaml
```

## 3. 输出

训练结束后会在：

```bash
${train.output_dir}/adapter
```

保存：

- LoRA adapter
- processor / tokenizer
- 训练时使用的 YAML

## 4. 数据格式

当前脚本假设：

- JSON 根节点是 list
- 每条样本包含：
  - `messages`: 多轮对话
  - `images`: 对话中 `<image>` 占位符对应的图片路径列表
- `images` 的顺序必须和所有 `<image>` 出现的顺序一致

## 5. 说明

- 默认 `batch_size=1`，通过 `accumulate_grad_batches` 增大有效 batch。
- 如果出现 `Image tokens are truncated or mismatched`，把 `data.max_length` 调大。
- 如果你机器支持并装了 flash-attn，可以把 `model.attn_implementation` 改成 `flash_attention_2`。
