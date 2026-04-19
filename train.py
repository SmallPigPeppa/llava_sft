#!/usr/bin/env python3
"""Minimal Hugging Face Trainer script for LLaVA LoRA SFT.

Example:
    WANDB_PROJECT=CL-debug python train.py --config configs/demo2k.yaml
    python train.py --config configs/demo2k.yaml train.num_train_epochs=1 data.max_samples=64
"""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed

from data import IGNORE_INDEX, ShareGPTLlavaDataset, load_raw_dataset, read_dataset_spec, split_train_eval


EXCLUDE_LORA_KEYWORDS = (
    "vision_tower",
    "vision_model",
    "visual",
    "multi_modal_projector",
    "mm_projector",
    "projector",
    "lm_head",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dot-list overrides, e.g. train.learning_rate=1e-4 data.max_samples=128",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        cursor = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = parse_value(raw_value)
    return cfg


def resolve_torch_dtype(name: Any):
    if name is None or name == "auto":
        return "auto"
    if isinstance(name, torch.dtype):
        return name
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[str(name).lower()]


def load_vision_language_model(model_cfg: dict[str, Any]):
    """Load processor + LLaVA model while keeping imports compatible across Transformers versions."""
    model_name = model_cfg["model_name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("AutoProcessor has no tokenizer; this script expects a LLaVA-style processor.")
    tokenizer.padding_side = str(model_cfg.get("padding_side", "right"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": resolve_torch_dtype(model_cfg.get("torch_dtype", "bfloat16")),
    }
    if model_cfg.get("device_map") is not None:
        model_kwargs["device_map"] = model_cfg["device_map"]
    if model_cfg.get("attn_implementation") is not None:
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    if model_cfg.get("load_in_4bit"):
        model_kwargs["load_in_4bit"] = True
    if model_cfg.get("load_in_8bit"):
        model_kwargs["load_in_8bit"] = True

    try:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
    except Exception:
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(model_name, **model_kwargs)

    if bool(model_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return model, processor, tokenizer


def freeze_by_keywords(model: torch.nn.Module, keywords: tuple[str, ...]) -> None:
    for name, param in model.named_parameters():
        if any(key in name for key in keywords):
            param.requires_grad = False


def find_linear_lora_targets(model: torch.nn.Module, exclude_keywords: tuple[str, ...] = EXCLUDE_LORA_KEYWORDS) -> list[str]:
    """Return full Linear module names outside vision tower/projector/lm_head.

    Full names are used instead of suffixes so PEFT does not accidentally inject LoRA into
    vision modules with the same names, such as q_proj/v_proj.
    """
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(key in name for key in exclude_keywords):
            continue
        targets.append(name)
    if not targets:
        raise ValueError("No Linear modules found for LoRA. Set lora.target_modules explicitly in YAML.")
    return targets


def apply_lora(model: torch.nn.Module, lora_cfg: dict[str, Any]) -> torch.nn.Module:
    if not bool(lora_cfg.get("enable", True)):
        return model

    target_modules = lora_cfg.get("target_modules", "all")
    if target_modules in {"all", "all-linear", "all_linear"}:
        target_modules = find_linear_lora_targets(model)
    elif isinstance(target_modules, str):
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]

    if lora_cfg.get("freeze_vision_tower", True):
        freeze_by_keywords(model, ("vision_tower", "vision_model", "visual"))
    if lora_cfg.get("freeze_multi_modal_projector", True):
        freeze_by_keywords(model, ("multi_modal_projector", "mm_projector", "projector"))

    if lora_cfg.get("prepare_kbit", False):
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def infer_image_seq_len(processor, model, data_cfg: dict[str, Any]) -> int:
    if data_cfg.get("image_seq_len") is not None:
        return int(data_cfg["image_seq_len"])

    image_processor = getattr(processor, "image_processor", None)
    crop_size = getattr(image_processor, "crop_size", None) or {}
    size = getattr(image_processor, "size", None) or {}

    height = crop_size.get("height") or size.get("height") or size.get("shortest_edge") or 336
    width = crop_size.get("width") or size.get("width") or size.get("shortest_edge") or 336

    patch_size = getattr(processor, "patch_size", None)
    if patch_size is None:
        vision_cfg = getattr(getattr(model, "config", None), "vision_config", None)
        patch_size = getattr(vision_cfg, "patch_size", 14)

    additional = getattr(processor, "num_additional_image_tokens", None)
    if additional is None:
        additional = 1

    strategy = getattr(processor, "vision_feature_select_strategy", None)
    if strategy is None:
        strategy = getattr(getattr(model, "config", None), "vision_feature_select_strategy", "default")

    seq_len = (int(height) // int(patch_size)) * (int(width) // int(patch_size)) + int(additional)
    if strategy == "default":
        seq_len -= 1
    return int(seq_len)


class LlavaDataCollator:
    """Pad text labels and build image tensors through the HF image processor."""

    def __init__(self, tokenizer, processor, pad_to_multiple_of: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad_labels(self, labels: list[list[int]], max_len: int) -> torch.Tensor:
        padded = []
        for row in labels:
            pad_len = max_len - len(row)
            if self.tokenizer.padding_side == "right":
                padded.append(row + [IGNORE_INDEX] * pad_len)
            else:
                padded.append([IGNORE_INDEX] * pad_len + row)
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = []
        text_features = []
        label_rows = []
        for feature in features:
            images.extend(feature.get("images") or [])
            text_features.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
            )
            label_rows.append(feature["labels"])

        batch = self.tokenizer.pad(
            text_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = self._pad_labels(label_rows, max_len=batch["input_ids"].shape[1])

        if images:
            image_inputs = self.processor.image_processor(images, return_tensors="pt")
            batch.update(image_inputs)
        return batch


def build_training_args(train_cfg: dict[str, Any]) -> TrainingArguments:
    cfg = dict(train_cfg)

    # User requested: log to wandb, no checkpoints.
    if cfg.get("save_checkpoint", False) is False:
        cfg["save_strategy"] = "no"
        cfg.pop("save_steps", None)

    cfg.setdefault("remove_unused_columns", False)
    cfg.setdefault("report_to", ["wandb"])

    # Compatibility between older/newer Transformers names.
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in cfg and "eval_strategy" not in sig and "evaluation_strategy" in sig:
        cfg["evaluation_strategy"] = cfg.pop("eval_strategy")
    if "evaluation_strategy" in cfg and "evaluation_strategy" not in sig and "eval_strategy" in sig:
        cfg["eval_strategy"] = cfg.pop("evaluation_strategy")

    save_model_at_end = cfg.pop("save_model_at_end", False)
    cfg["save_model_at_end"] = save_model_at_end  # reattached below as dynamic attribute

    allowed = {k: v for k, v in cfg.items() if k in sig}
    args = TrainingArguments(**allowed)
    setattr(args, "save_model_at_end", bool(save_model_at_end))
    return args


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.overrides)

    seed = int(cfg.get("seed", cfg.get("data", {}).get("seed", 42)))
    set_seed(seed)

    wandb_project = cfg.get("wandb_project") or cfg.get("train", {}).get("wandb_project")
    if wandb_project:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_project))

    model, processor, tokenizer = load_vision_language_model(cfg["model"])
    model = apply_lora(model, cfg.get("lora", {}))

    data_cfg = cfg["data"]
    spec = read_dataset_spec(data_cfg)
    raw_ds = load_raw_dataset(data_cfg)
    train_raw, eval_raw = split_train_eval(raw_ds, data_cfg)

    image_seq_len = infer_image_seq_len(processor, model, data_cfg)
    print(f"Using image_seq_len={image_seq_len}")
    print(f"Loaded dataset={spec.dataset_name}, train={len(train_raw)}, eval={len(eval_raw) if eval_raw is not None else 0}")

    train_ds = ShareGPTLlavaDataset(
        train_raw,
        tokenizer=tokenizer,
        spec=spec,
        cutoff_len=int(data_cfg.get("cutoff_len", 2048)),
        image_seq_len=image_seq_len,
        image_token=str(data_cfg.get("image_token", "<image>")),
        add_default_system=bool(data_cfg.get("add_default_system", True)),
        train_on_prompt=bool(data_cfg.get("train_on_prompt", False)),
    )
    eval_ds = None
    if eval_raw is not None:
        eval_ds = ShareGPTLlavaDataset(
            eval_raw,
            tokenizer=tokenizer,
            spec=spec,
            cutoff_len=int(data_cfg.get("cutoff_len", 2048)),
            image_seq_len=image_seq_len,
            image_token=str(data_cfg.get("image_token", "<image>")),
            add_default_system=bool(data_cfg.get("add_default_system", True)),
            train_on_prompt=bool(data_cfg.get("train_on_prompt", False)),
        )

    train_args = build_training_args(cfg["train"])
    collator = LlavaDataCollator(
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=cfg.get("train", {}).get("pad_to_multiple_of"),
    )

    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=cfg.get("train", {}).get("resume_from_checkpoint"))

    if getattr(train_args, "save_model_at_end", False):
        trainer.save_model(train_args.output_dir)
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(train_args.output_dir)
    else:
        print("Training finished. Checkpoint/model saving disabled; metrics/logs are sent to W&B.")


if __name__ == "__main__":
    main()
