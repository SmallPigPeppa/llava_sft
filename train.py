import argparse
import json
import os
import re
import shutil
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration


IMAGE_TOKEN = "<image>"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def save_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)



def get_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]



def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")



def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False



def freeze_all_parameters(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False



def build_lm_target_regex(model: LlavaForConditionalGeneration) -> Tuple[str, List[str]]:
    """
    只从 language_model 下面收集 Linear，并生成一个只匹配语言模型路径的正则，
    防止 q_proj/k_proj 这类名字把 vision_tower 也匹配进去。
    """
    leaf_names = set()
    for name, sub_module in model.model.language_model.named_modules():
        if isinstance(sub_module, nn.Linear):
            leaf_names.add(name.split(".")[-1])

    # 这些一般不想打 LoRA
    leaf_names.discard("lm_head")
    leaf_names.discard("embed_tokens")

    if not leaf_names:
        raise ValueError("No Linear modules found under model.model.language_model")

    escaped = [re.escape(x) for x in sorted(leaf_names)]
    pattern = rf"^model\.language_model\..*\.({'|'.join(escaped)})$"
    return pattern, sorted(leaf_names)



def assert_only_lora_trainable(model: nn.Module) -> None:
    wrong = []
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_" not in name:
            wrong.append(name)
    if wrong:
        raise RuntimeError(
            "Found non-LoRA trainable parameters, which violates the requirement that only the language model uses LoRA: "
            + ", ".join(wrong[:20])
        )



def split_text_and_collect_images(
    text: str,
    image_paths: Sequence[str],
    image_ptr: int,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    把类似 '<image>Who are they?<image>' 解析成 HF chat template 需要的 content list。
    images 按 JSON 中 images 数组的顺序逐个消费。
    """
    if not isinstance(text, str):
        raise TypeError(f"message content must be str, got {type(text)}")

    parts = text.split(IMAGE_TOKEN)
    content: List[Dict[str, str]] = []
    used_paths: List[str] = []

    for i, part in enumerate(parts):
        if i > 0:
            if image_ptr + len(used_paths) >= len(image_paths):
                raise ValueError(
                    f"Not enough images for message: {text!r}. image_ptr={image_ptr}, total_images={len(image_paths)}"
                )
            content.append({"type": "image"})
            used_paths.append(image_paths[image_ptr + len(used_paths)])

        if part:
            content.append({"type": "text", "text": part})

    if not content:
        # 纯空串时兜底；纯 <image> 则上面已经有 image block 了
        content = [{"type": "text", "text": ""}]

    return content, used_paths



def normalize_image_paths(raw_paths: Sequence[str], image_root: str) -> List[str]:
    result = []
    for p in raw_paths:
        if os.path.isabs(p):
            result.append(p)
        else:
            result.append(os.path.join(image_root, p))
    return result



def build_sft_samples(records: List[Dict[str, Any]], image_root: str) -> List[Dict[str, Any]]:
    """
    最简单的做法：把每个 assistant 回复都展开成一个训练样本。
    即：history(到当前 user 为止) -> 当前 assistant answer。
    这样最容易做 label mask。
    """
    samples: List[Dict[str, Any]] = []

    for record_idx, record in enumerate(records):
        messages = record.get("messages", [])
        raw_images = record.get("images", [])
        image_paths = normalize_image_paths(raw_images, image_root)

        history_messages: List[Dict[str, Any]] = []
        history_image_paths: List[str] = []
        image_ptr = 0

        for turn_idx, msg in enumerate(messages):
            role = msg["role"]
            text = msg["content"]

            if role == "user":
                content, used_images = split_text_and_collect_images(text, image_paths, image_ptr)
                history_messages.append({"role": "user", "content": content})
                history_image_paths.extend(used_images)
                image_ptr += len(used_images)

            elif role == "assistant":
                samples.append(
                    {
                        "conversation": deepcopy(history_messages),
                        "answer": text,
                        "image_paths": list(history_image_paths),
                        "record_idx": record_idx,
                        "turn_idx": turn_idx,
                    }
                )
                history_messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    }
                )
            else:
                raise ValueError(f"Unsupported role: {role}")

        if image_ptr != len(image_paths):
            raise ValueError(
                f"Unused images found in record {record_idx}: consumed={image_ptr}, total={len(image_paths)}"
            )

    return samples


class JsonLlavaSFTDataset(Dataset):
    def __init__(self, json_path: str, image_root: str):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError("JSON root must be a list")

        self.samples = build_sft_samples(records, image_root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


class LlavaSFTCollator:
    def __init__(self, processor: AutoProcessor, max_length: int):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.image_token_id = getattr(processor, "image_token_id", None)
        self.num_image_tokens_per_image = self._get_num_image_tokens_per_image()

    def _get_num_image_tokens_per_image(self) -> Optional[int]:
        if not hasattr(self.processor, "patch_size"):
            return None

        crop_size = self.processor.image_processor.crop_size
        if isinstance(crop_size, dict):
            height = crop_size["height"]
            width = crop_size["width"]
        elif isinstance(crop_size, (tuple, list)):
            height, width = crop_size
        else:
            height = width = int(crop_size)

        patch_size = int(self.processor.patch_size)
        num = (height // patch_size) * (width // patch_size) + int(self.processor.num_additional_image_tokens)
        if getattr(self.processor, "vision_feature_select_strategy", None) == "default":
            num -= 1
        return num

    def _build_prompt_text(self, conversation: List[Dict[str, Any]]) -> str:
        prompt_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not prompt_text.endswith((" ", "\n")):
            prompt_text += " "
        return prompt_text

    def _encode_one(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        images = [load_image(p) for p in sample["image_paths"]]
        prompt_text = self._build_prompt_text(sample["conversation"])
        full_text = prompt_text + sample["answer"] + self.tokenizer.eos_token

        processor_kwargs = {
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": self.max_length,
        }

        full_inputs = self.processor(
            text=full_text,
            images=images if len(images) > 0 else None,
            **processor_kwargs,
        )
        prompt_inputs = self.processor(
            text=prompt_text,
            images=images if len(images) > 0 else None,
            **processor_kwargs,
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        # 避免 max_length 截断掉 image placeholder 导致 embeddings merge 报错
        if len(images) > 0 and self.image_token_id is not None and self.num_image_tokens_per_image is not None:
            num_image_tokens = int((input_ids == self.image_token_id).sum().item())
            expected = len(images) * self.num_image_tokens_per_image
            if num_image_tokens != expected:
                raise ValueError(
                    f"Image tokens are truncated or mismatched: got={num_image_tokens}, expected={expected}. "
                    f"Please increase data.max_length. record_idx={sample['record_idx']}, turn_idx={sample['turn_idx']}"
                )

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "pixel_values" in full_inputs:
            out["pixel_values"] = full_inputs["pixel_values"]

        if "image_sizes" in full_inputs:
            out["image_sizes"] = full_inputs["image_sizes"]

        return out

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded = [self._encode_one(x) for x in batch]
        max_len = max(x["input_ids"].shape[0] for x in encoded)
        pad_id = self.tokenizer.pad_token_id

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        image_sizes_list = []

        for item in encoded:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids_list.append(F.pad(item["input_ids"], (0, pad_len), value=pad_id))
            attention_mask_list.append(F.pad(item["attention_mask"], (0, pad_len), value=0))
            labels_list.append(F.pad(item["labels"], (0, pad_len), value=-100))

            if "pixel_values" in item:
                pixel_values_list.append(item["pixel_values"])
            if "image_sizes" in item:
                image_sizes_list.append(item["image_sizes"])

        output = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
        }

        # LLaVA HF 在多图/多样本时可以直接把所有图按 batch 顺序拼起来，
        # 文本中的 <image> placeholder 会按同样顺序消费这些图像特征。
        if pixel_values_list:
            output["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_sizes_list:
            output["image_sizes"] = torch.cat(image_sizes_list, dim=0)

        return output


class LlavaLitModule(L.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        model_cfg = cfg["model"]
        train_cfg = cfg["train"]
        lora_cfg = cfg["lora"]
        data_cfg = cfg["data"]

        dtype = get_torch_dtype(model_cfg["torch_dtype"])

        model_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if model_cfg.get("attn_implementation"):
            model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

        self.processor = AutoProcessor.from_pretrained(model_cfg["name_or_path"], use_fast=False)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_cfg["name_or_path"],
            **model_kwargs,
        )

        # 按 HF 官方建议，把这几个属性补到 processor 上，避免 image token 展开告警/错误。
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        self.processor.num_additional_image_tokens = 1  # CLIP vision backbone 含 CLS token

        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"
        self.processor.tokenizer.model_max_length = int(data_cfg["max_length"])

        self.model.config.use_cache = False

        if model_cfg.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        # 先全冻结，再只在 language_model 上打 LoRA
        freeze_all_parameters(self.model)
        freeze_module(self.model.model.vision_tower)
        freeze_module(self.model.model.multi_modal_projector)

        auto_target_regex, auto_leaf_names = build_lm_target_regex(self.model)
        target_modules = lora_cfg.get("target_modules_regex") or auto_target_regex

        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias=lora_cfg.get("bias", "none"),
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, self.peft_config)
        assert_only_lora_trainable(self.model)

        print(f"[LoRA] auto leaf names under language model: {auto_leaf_names}")
        print(f"[LoRA] target_modules regex: {target_modules}")
        self.model.print_trainable_parameters()

        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if num_trainable == 0:
            raise RuntimeError("No trainable parameters found after applying LoRA")

        self.learning_rate = float(train_cfg["lr"])
        self.weight_decay = float(train_cfg.get("weight_decay", 0.0))
        self.adam_beta1 = float(train_cfg.get("adam_beta1", 0.9))
        self.adam_beta2 = float(train_cfg.get("adam_beta2", 0.999))
        self.adam_eps = float(train_cfg.get("adam_eps", 1e-8))

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["input_ids"].size(0),
            sync_dist=self.trainer.world_size > 1,
        )
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def save_adapter(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)



def build_dataloader(cfg: Dict[str, Any], processor: AutoProcessor) -> Tuple[Dataset, DataLoader]:
    data_cfg = cfg["data"]
    loader_cfg = cfg["loader"]

    if data_cfg.get("path") != "json":
        raise ValueError("This minimal example only supports data.path == 'json'")

    dataset = JsonLlavaSFTDataset(
        json_path=data_cfg["data_files"],
        image_root=data_cfg["image_root"],
    )
    collator = LlavaSFTCollator(
        processor=processor,
        max_length=int(data_cfg["max_length"]),
    )

    num_workers = int(loader_cfg.get("num_workers", 0))
    dataloader = DataLoader(
        dataset,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=bool(loader_cfg.get("shuffle", True)),
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)) if num_workers > 0 else False,
        collate_fn=collator,
        drop_last=bool(loader_cfg.get("drop_last", False)),
    )
    return dataset, dataloader



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    output_dir = cfg["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    save_yaml(cfg, os.path.join(output_dir, "used_config.yaml"))

    seed = int(cfg.get("seed", 42))
    L.seed_everything(seed, workers=True)

    lit_model = LlavaLitModule(cfg)
    dataset, train_loader = build_dataloader(cfg, lit_model.processor)

    if len(dataset) == 0:
        raise RuntimeError("No training samples found")

    logger = CSVLogger(save_dir=output_dir, name="csv_logs")

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if cfg["trainer"].get("save_lightning_ckpt", False):
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(output_dir, "lightning_ckpt"),
                filename="epoch{epoch:02d}-step{step}",
                save_last=True,
                save_top_k=-1,
                every_n_epochs=1,
            )
        )

    trainer = L.Trainer(
        accelerator=cfg["trainer"].get("accelerator", "gpu"),
        devices=cfg["trainer"].get("devices", 1),
        num_nodes=cfg["trainer"].get("num_nodes", 1),
        strategy=cfg["trainer"].get("strategy", "auto"),
        precision=cfg["trainer"].get("precision", "16-mixed"),
        max_epochs=int(cfg["trainer"].get("max_epochs", 1)),
        accumulate_grad_batches=int(cfg["trainer"].get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(cfg["trainer"].get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(cfg["trainer"].get("log_every_n_steps", 1)),
        default_root_dir=output_dir,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=bool(cfg["trainer"].get("save_lightning_ckpt", False)),
        num_sanity_val_steps=0,
    )

    print(f"[Data] num training samples: {len(dataset)}")
    trainer.fit(lit_model, train_dataloaders=train_loader)

    if trainer.is_global_zero:
        adapter_dir = os.path.join(output_dir, "adapter")
        lit_model.save_adapter(adapter_dir)
        shutil.copy2(args.config, os.path.join(output_dir, "train_config.yaml"))
        print(f"[Save] LoRA adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
