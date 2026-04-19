"""Minimal ShareGPT/LLaVA dataset loading utilities.

The goal is to keep only the pieces needed by the demo2k run:
- read a LLaMA-Factory-like dataset_info.json
- load a local Parquet/JSON dataset or a Hugging Face dataset
- convert ShareGPT conversations + images into input_ids/labels for SFT
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"
DEFAULT_LLAVA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


@dataclass
class DatasetSpec:
    dataset_name: str
    dataset_dir: str
    formatting: str
    messages_col: str
    images_col: Optional[str]
    split: str = "train"
    media_dir: Optional[str] = None


def read_dataset_spec(data_cfg: dict[str, Any]) -> DatasetSpec:
    """Read dataset metadata from dataset_info.json, with YAML fields as overrides."""
    dataset_dir = str(data_cfg.get("dataset_dir", "data"))
    dataset_name = str(data_cfg["dataset"])
    info_path = Path(dataset_dir) / "dataset_info.json"

    if not info_path.exists():
        raise FileNotFoundError(
            f"dataset_info.json not found: {info_path}. "
            "Set data.dataset_dir to the directory containing dataset_info.json."
        )

    with info_path.open("r", encoding="utf-8") as f:
        all_info = json.load(f)

    if dataset_name not in all_info:
        raise KeyError(f"Dataset {dataset_name!r} not found in {info_path}.")

    info = dict(all_info[dataset_name])
    columns = dict(info.get("columns") or {})

    return DatasetSpec(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        formatting=str(info.get("formatting", data_cfg.get("formatting", "sharegpt"))),
        messages_col=str(data_cfg.get("messages_col") or columns.get("messages") or "conversations"),
        images_col=data_cfg.get("images_col") or columns.get("images"),
        split=str(info.get("split", data_cfg.get("split", "train"))),
        media_dir=data_cfg.get("media_dir"),
    )


def _resolve_data_path(dataset_dir: str, file_name: str) -> str:
    file_name = os.path.expanduser(file_name)
    if os.path.isabs(file_name):
        return file_name
    return str(Path(dataset_dir) / file_name)


def _load_local_dataset(path: str, split: str):
    from datasets import load_dataset, load_from_disk

    path = os.path.expanduser(path)
    if os.path.isdir(path):
        # Hugging Face Dataset.save_to_disk layout.
        if os.path.exists(os.path.join(path, "dataset_info.json")) and os.path.exists(os.path.join(path, "state.json")):
            return load_from_disk(path)

        parquet_files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        jsonl_files = sorted(glob.glob(os.path.join(path, "*.jsonl"))) + sorted(glob.glob(os.path.join(path, "*.json")))
        if parquet_files:
            return load_dataset("parquet", data_files=parquet_files, split=split)
        if jsonl_files:
            return load_dataset("json", data_files=jsonl_files, split=split)
        raise FileNotFoundError(f"No .parquet/.json/.jsonl files found in local dataset dir: {path}")

    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=path, split=split)
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=path, split=split)
    raise ValueError(f"Unsupported dataset file type: {path}")


def load_raw_dataset(data_cfg: dict[str, Any]):
    """Load the raw Hugging Face Dataset described by dataset_info.json."""
    from datasets import load_dataset

    dataset_dir = str(data_cfg.get("dataset_dir", "data"))
    dataset_name = str(data_cfg["dataset"])
    info_path = Path(dataset_dir) / "dataset_info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)[dataset_name]

    split = str(info.get("split", data_cfg.get("split", "train")))
    if "hf_hub_url" in info:
        ds = load_dataset(
            info["hf_hub_url"],
            split=split,
            trust_remote_code=bool(data_cfg.get("trust_remote_code", False)),
        )
    elif "file_name" in info:
        ds = _load_local_dataset(_resolve_data_path(dataset_dir, info["file_name"]), split=split)
    else:
        raise ValueError(f"Dataset {dataset_name!r} must define hf_hub_url or file_name in {info_path}.")

    max_samples = data_cfg.get("max_samples")
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    return ds


def split_train_eval(raw_ds, data_cfg: dict[str, Any]):
    """Split train/eval using data.val_size. val_size can be float ratio or int count."""
    val_size = data_cfg.get("val_size", 0)
    if not val_size:
        return raw_ds, None

    seed = int(data_cfg.get("seed", 42))
    if isinstance(val_size, float) and 0 < val_size < 1:
        test_size = val_size
    else:
        test_size = int(val_size)

    split = raw_ds.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    """Normalize ShareGPT messages to [{'role': 'user'|'assistant'|'system', 'content': ...}]."""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError as exc:
            raise ValueError("messages column is a string but not valid JSON") from exc

    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages)!r}")

    out: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise TypeError(f"message must be dict, got {type(msg)!r}")

        raw_role = str(msg.get("from", msg.get("role", ""))).lower()
        content = str(msg.get("value", msg.get("content", "")))

        if raw_role in {"human", "user"}:
            role = "user"
        elif raw_role in {"gpt", "assistant", "model"}:
            role = "assistant"
        elif raw_role == "system":
            role = "system"
        else:
            raise ValueError(f"Unknown ShareGPT role: {raw_role!r}")
        out.append({"role": role, "content": content})
    return out


def _open_image_from_path(path: str, media_dir: Optional[str]) -> Image.Image:
    path = os.path.expanduser(path)
    candidates = [path]
    if media_dir and not os.path.isabs(path):
        candidates.insert(0, os.path.join(os.path.expanduser(media_dir), path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return Image.open(candidate).convert("RGB")
    raise FileNotFoundError(f"Image file not found: {path}; tried: {candidates}")


def normalize_images(value: Any, media_dir: Optional[str] = None) -> list[Image.Image]:
    """Accept PIL, path, bytes, HF Image dict, or a list of them; return RGB PIL images."""
    if value is None:
        return []
    if isinstance(value, list):
        images: list[Image.Image] = []
        for item in value:
            images.extend(normalize_images(item, media_dir=media_dir))
        return images
    if isinstance(value, Image.Image):
        return [value.convert("RGB")]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return [Image.open(BytesIO(bytes(value))).convert("RGB")]
    if isinstance(value, (str, os.PathLike)):
        return [_open_image_from_path(os.fspath(value), media_dir=media_dir)]
    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        raw_path = value.get("path") or value.get("file_name") or value.get("filename")
        if raw_bytes is not None:
            return [Image.open(BytesIO(raw_bytes)).convert("RGB")]
        if raw_path is not None:
            return [_open_image_from_path(str(raw_path), media_dir=media_dir)]

    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def expand_image_tokens(text: str, image_seq_len: int, image_token: str = IMAGE_TOKEN) -> str:
    """Replace one logical image placeholder with N actual image tokens."""
    if image_seq_len <= 1:
        return text
    return text.replace(image_token, image_token * image_seq_len)


class ShareGPTLlavaDataset(Dataset):
    """Lazy ShareGPT/LLaVA SFT dataset.

    It mirrors the essential LLaMA-Factory behavior for `template: llava`:
    USER: {content} ASSISTANT:{assistant}</s>
    Prompt tokens are masked with -100, assistant tokens are trained.
    """

    def __init__(
        self,
        raw_dataset,
        tokenizer,
        spec: DatasetSpec,
        cutoff_len: int = 2048,
        image_seq_len: int = 576,
        image_token: str = IMAGE_TOKEN,
        add_default_system: bool = True,
        train_on_prompt: bool = False,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.spec = spec
        self.cutoff_len = int(cutoff_len)
        self.image_seq_len = int(image_seq_len)
        self.image_token = image_token
        self.add_default_system = bool(add_default_system)
        self.train_on_prompt = bool(train_on_prompt)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _format_user(self, content: str, is_first_user: bool, num_images: int) -> str:
        # LLaVA data often has <image> in the first user message. If the image column is present
        # but the placeholder is missing, prefix missing placeholders to keep tokens/features aligned.
        missing = max(0, num_images - content.count(self.image_token)) if is_first_user else 0
        if missing:
            content = (self.image_token + "\n") * missing + content
        content = expand_image_tokens(content, self.image_seq_len, self.image_token)

        prefix = DEFAULT_LLAVA_SYSTEM if (is_first_user and self.add_default_system) else ""
        return f"{prefix}USER: {content} ASSISTANT:"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.raw_dataset[int(idx)]
        messages = normalize_messages(row[self.spec.messages_col])
        images = normalize_images(row.get(self.spec.images_col), media_dir=self.spec.media_dir) if self.spec.images_col else []

        # Optional system message overrides the default system for the first turn.
        if messages and messages[0]["role"] == "system":
            system_text = messages[0]["content"]
            messages = messages[1:]
        else:
            system_text = DEFAULT_LLAVA_SYSTEM if self.add_default_system else ""

        input_ids: list[int] = []
        labels: list[int] = []
        first_user = True
        pending_user: Optional[str] = None

        for msg in messages:
            if msg["role"] == "user":
                pending_user = msg["content"]
                continue
            if msg["role"] != "assistant" or pending_user is None:
                continue

            source = self._format_user(pending_user, first_user, len(images))
            if first_user and system_text != DEFAULT_LLAVA_SYSTEM:
                # Preserve a dataset-provided system prompt without pulling in a full template engine.
                source = source.replace(DEFAULT_LLAVA_SYSTEM, system_text, 1)

            target = msg["content"] + (self.tokenizer.eos_token or "")
            source_ids = self._encode(source)
            target_ids = self._encode(target)

            input_ids.extend(source_ids)
            input_ids.extend(target_ids)
            labels.extend(source_ids if self.train_on_prompt else [IGNORE_INDEX] * len(source_ids))
            labels.extend(target_ids)

            first_user = False
            pending_user = None

        if not input_ids:
            raise ValueError(f"No valid user/assistant pair found at index {idx}")

        if len(input_ids) > self.cutoff_len:
            input_ids = input_ids[: self.cutoff_len]
            labels = labels[: self.cutoff_len]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "images": images,
        }
