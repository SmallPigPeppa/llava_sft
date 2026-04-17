from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import yaml
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration, get_scheduler

try:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
except Exception:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

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

        if model_cfg.get("gradient_checkpointing", True):
            if hasattr(model, "gradient_checkpointing_enable"):
                try:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": True}
                    )
                except TypeError:
                    model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
        model.print_trainable_parameters()

    return model, processor


def apply_runtime_flags(train_cfg: dict[str, Any]) -> None:
    tf32 = bool(train_cfg.get("tf32", True))
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = tf32


def build_optimizer_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("没有可训练参数。")

    if weight_decay <= 0:
        return [{"params": [p for _, p in trainable], "weight_decay": 0.0}]

    no_decay_keywords = ("bias", "norm", "layernorm", "layer_norm", "ln_")
    decay_params = []
    no_decay_params = []

    for name, param in trainable:
        lname = name.lower()
        if any(key in lname for key in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def resolve_precision(train_cfg: dict[str, Any], accelerator: str) -> str:
    if accelerator != "gpu":
        return "32-true"

    use_bf16 = bool(train_cfg.get("bf16", False))
    use_fp16 = bool(train_cfg.get("fp16", False))
    if use_bf16 and use_fp16:
        raise ValueError("bf16 和 fp16 不能同时为 true。")
    if use_bf16:
        return "bf16-mixed"
    if use_fp16:
        return "16-mixed"
    return "32-true"


def build_logger(train_cfg: dict[str, Any]):
    report_to = train_cfg.get("report_to", "none")
    if report_to is None:
        return False

    if isinstance(report_to, (list, tuple)):
        targets = {str(x).lower() for x in report_to}
    else:
        targets = {str(report_to).lower()}

    if not targets or targets == {"none"}:
        return False

    output_dir = train_cfg["output_dir"]
    logging_steps = int(train_cfg.get("logging_steps", 10))

    if "tensorboard" in targets or "tb" in targets:
        return TensorBoardLogger(save_dir=output_dir, name="lightning_logs")

    return CSVLogger(
        save_dir=output_dir,
        name="lightning_logs",
        flush_logs_every_n_steps=logging_steps,
    )


def build_checkpoint_callback(train_cfg: dict[str, Any]) -> ModelCheckpoint:
    save_total_limit = int(train_cfg.get("save_total_limit", 2))
    save_steps = int(train_cfg.get("save_steps", 500))

    return ModelCheckpoint(
        dirpath=os.path.join(train_cfg["output_dir"], "checkpoints"),
        filename="step={step}",
        monitor="step",
        mode="max",
        save_top_k=save_total_limit,
        save_last=True,
        every_n_train_steps=save_steps,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )


def resolve_strategy(train_cfg: dict[str, Any]):
    deepspeed_cfg = train_cfg.get("deepspeed")
    if deepspeed_cfg:
        return DeepSpeedStrategy(config=deepspeed_cfg)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        return DDPStrategy(
            find_unused_parameters=bool(train_cfg.get("ddp_find_unused_parameters", False))
        )
    return "auto"


def infer_devices_and_nodes(accelerator: str) -> tuple[int, int]:
    if accelerator != "gpu":
        return 1, 1

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 1, 1

    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size is None:
        local_world_size = str(max(torch.cuda.device_count(), 1))

    devices = int(local_world_size)
    num_nodes = max(world_size // max(devices, 1), 1)
    return devices, num_nodes


class LlavaLightningModule(L.LightningModule):
    def __init__(self, cfg: dict[str, Any], model: LlavaForConditionalGeneration, processor: AutoProcessor):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.train_cfg = cfg["train"]
        self.save_hyperparameters({"config": cfg}, logger=False)

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "train_loss_epoch",
            loss.detach(),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        return loss

    def configure_optimizers(self):
        optim_name = str(self.train_cfg.get("optim", "adamw_torch")).lower()
        if optim_name not in {"adamw_torch", "adamw"}:
            raise ValueError(f"极简版目前只保留 adamw_torch / adamw，收到: {optim_name}")

        optimizer = torch.optim.AdamW(
            build_optimizer_param_groups(
                self.model,
                float(self.train_cfg.get("weight_decay", 0.0)),
            ),
            lr=float(self.train_cfg.get("learning_rate", 2e-5)),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler_name = str(self.train_cfg.get("lr_scheduler_type", "cosine"))
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * float(self.train_cfg.get("warmup_ratio", 0.03)))
        scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        scheduler_cfg: dict[str, Any] = {
            "scheduler": scheduler,
            "frequency": 1,
        }

        if scheduler_name == "reduce_lr_on_plateau":
            scheduler_cfg["interval"] = "epoch"
            scheduler_cfg["monitor"] = "train_loss_epoch"
        else:
            scheduler_cfg["interval"] = "step"

        return {"optimizer": optimizer, "lr_scheduler": scheduler_cfg}

    def save_hf_artifacts(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)


def build_train_dataloader(cfg: dict[str, Any], processor: AutoProcessor) -> tuple[MllmSFTDataset, DataLoader]:
    dataset = MllmSFTDataset(cfg["data"], processor)
    collator = LlavaSFTCollator(processor, int(cfg["data"].get("max_length", 2048)))
    num_workers = int(cfg["train"].get("dataloader_num_workers", 4))

    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("per_device_train_batch_size", 1)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collator,
    )
    return dataset, train_loader


def build_trainer(cfg: dict[str, Any]) -> L.Trainer:
    train_cfg = cfg["train"]
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices, num_nodes = infer_devices_and_nodes(accelerator)
    precision = resolve_precision(train_cfg, accelerator)
    strategy = resolve_strategy(train_cfg)
    logger = build_logger(train_cfg)
    checkpoint_callback = build_checkpoint_callback(train_cfg)

    max_steps = train_cfg.get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
        max_epochs = -1
    else:
        max_steps = -1
        max_epochs = int(float(train_cfg.get("num_train_epochs", 1)))

    return L.Trainer(
        default_root_dir=train_cfg["output_dir"],
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        accumulate_grad_batches=int(train_cfg.get("gradient_accumulation_steps", 1)),
        max_epochs=max_epochs,
        max_steps=max_steps,
        log_every_n_steps=int(train_cfg.get("logging_steps", 10)),
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        use_distributed_sampler=True,
    )


def resolve_resume_path(path: str | None) -> str | None:
    if not path:
        return None
    if path in {"last", "best", "hpc"} or str(path).endswith(".ckpt"):
        return path

    candidates = [
        os.path.join(path, "last.ckpt"),
        os.path.join(path, "checkpoints", "last.ckpt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    apply_runtime_flags(cfg["train"])

    model, processor = load_model_and_processor(cfg["model"])
    dataset, train_loader = build_train_dataloader(cfg, processor)
    print(f"train examples: {len(dataset)}")

    lit_model = LlavaLightningModule(cfg, model, processor)
    trainer = build_trainer(cfg)
    ckpt_path = resolve_resume_path(cfg["train"].get("resume_from_checkpoint"))
    trainer.fit(lit_model, train_dataloaders=train_loader, ckpt_path=ckpt_path)

    trainer.strategy.barrier()
    if trainer.is_global_zero:
        lit_model.save_hf_artifacts(cfg["train"]["output_dir"])
    trainer.strategy.barrier()


if __name__ == "__main__":
    main()
