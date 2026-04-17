from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import yaml
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def split_mm_text(text: Any, image_token: str) -> tuple[list[dict[str, str]], int]:
    if isinstance(text, list):
        image_count = 0
        for x in text:
            if isinstance(x, dict) and x.get("type") == "image":
                image_count += 1
        return text, image_count

    text = "" if text is None else str(text)
    parts = text.split(image_token)
    content: list[dict[str, str]] = []
    image_count = 0

    for i, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if i < len(parts) - 1:
            content.append({"type": "image"})
            image_count += 1

    if not content:
        content = [{"type": "text", "text": ""}]
    return content, image_count


def norm_image_path(image_root: str, image_path: str) -> str:
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(image_root, image_path)


class MllmSFTDataset(Dataset):
    def __init__(self, data_cfg: dict[str, Any], processor: AutoProcessor):
        raw = load_dataset(
            data_cfg["path"],
            data_files=data_cfg["data_files"],
            split=data_cfg.get("split", "train"),
        )
        max_samples = data_cfg.get("max_samples")
        if max_samples:
            raw = raw.select(range(min(int(max_samples), len(raw))))

        self.image_root = data_cfg["image_root"]
        self.image_token = data_cfg.get("image_token", "<image>")
        self.examples: list[dict[str, Any]] = []

        for item in raw:
            self.examples.extend(self.flatten_item(item, processor))

    def flatten_item(self, item: dict[str, Any], processor: AutoProcessor) -> list[dict[str, Any]]:
        image_list = item.get("images")
        if image_list is None and item.get("image") is not None:
            image_list = [item["image"]]
        image_list = image_list or []
        image_list = [norm_image_path(self.image_root, p) for p in image_list]

        history: list[dict[str, Any]] = []
        used_image_num = 0
        out: list[dict[str, Any]] = []

        for msg in item["messages"]:
            content, n_img = split_mm_text(msg.get("content"), self.image_token)
            role = msg["role"]
            history.append({"role": role, "content": content})
            used_image_num += n_img

            if used_image_num > len(image_list):
                raise ValueError(f"图片数量不够：需要 {used_image_num} 张，但只给了 {len(image_list)} 张。")

            if role != "assistant":
                continue
            if not history[:-1]:
                continue

            prompt_text = processor.apply_chat_template(
                history[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = processor.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=False,
            )
            out.append(
                {
                    "prompt_text": prompt_text,
                    "full_text": full_text,
                    "image_paths": image_list[:used_image_num],
                }
            )
        return out

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class LlavaSFTCollator:
    def __init__(self, processor: AutoProcessor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def load_images(self, image_paths: list[str]) -> list[Image.Image]:
        images = []
        for path in image_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        return images

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError(
                "这个极简版默认按多图样本来写，建议 per_device_train_batch_size=1。"
            )

        feature = features[0]
        images = self.load_images(feature["image_paths"])
        images = images if images else None

        prompt_inputs = self.processor(
            text=feature["prompt_text"],
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        full_inputs = self.processor(
            text=feature["full_text"],
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        labels = full_inputs["input_ids"].clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        prompt_len = min(prompt_len, labels.shape[1])
        labels[:, :prompt_len] = -100
        full_inputs["labels"] = labels
        return full_inputs


def load_model_and_processor(model_cfg: dict[str, Any]) -> tuple[LlavaForConditionalGeneration, AutoProcessor]:
    dtype = get_dtype(model_cfg.get("torch_dtype", "bfloat16"))

    model = LlavaForConditionalGeneration.from_pretrained(
        model_cfg["name_or_path"],
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_cfg["name_or_path"])

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.truncation_side = "left"

    if hasattr(processor, "image_processor"):
        processor.image_processor.do_pad = bool(model_cfg.get("do_pad", True))

    if getattr(processor, "patch_size", None) is None and hasattr(model.config.vision_config, "patch_size"):
        processor.patch_size = model.config.vision_config.patch_size
    if getattr(processor, "vision_feature_select_strategy", None) is None and hasattr(model.config, "vision_feature_select_strategy"):
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    if getattr(processor, "num_additional_image_tokens", None) is None:
        processor.num_additional_image_tokens = int(model_cfg.get("num_additional_image_tokens", 1))

    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if model_cfg.get("use_lora", True):
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("use_lora=true 但环境里没有安装 peft。")
        lora_cfg = LoraConfig(
            r=int(model_cfg.get("lora_r", 64)),
            lora_alpha=int(model_cfg.get("lora_alpha", 128)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=model_cfg.get(
                "lora_target",
                ".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            ),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, processor


def build_training_args(cfg: dict[str, Any]) -> TrainingArguments:
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    kwargs = dict(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 4)),
        bf16=bool(train_cfg.get("bf16", False)),
        fp16=bool(train_cfg.get("fp16", False)),
        tf32=bool(train_cfg.get("tf32", True)),
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        logging_first_step=True,
        ddp_find_unused_parameters=bool(train_cfg.get("ddp_find_unused_parameters", False)),
        save_strategy="steps",
        eval_strategy="no",
        optim=train_cfg.get("optim", "adamw_torch"),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        deepspeed=train_cfg.get("deepspeed"),
    )

    if train_cfg.get("max_steps") is not None:
        kwargs["max_steps"] = int(train_cfg["max_steps"])

    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model, processor = load_model_and_processor(cfg["model"])
    dataset = MllmSFTDataset(cfg["data"], processor)
    print(f"train examples: {len(dataset)}")
    collator = LlavaSFTCollator(processor, int(cfg["data"].get("max_length", 2048)))
    training_args = build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=cfg["train"].get("resume_from_checkpoint"))
    trainer.save_model()
    processor.save_pretrained(cfg["train"]["output_dir"])


if __name__ == "__main__":
    main()
