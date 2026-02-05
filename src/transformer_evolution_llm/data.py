"""Data loading helpers for lightweight laptop runs."""

from __future__ import annotations

import os
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dsl import DataConfig, DatasetShard

# Lazy-loaded callables (kept as module globals so tests can monkeypatch them).
load_dataset: Any | None = None
AutoTokenizer: Any | None = None


def _resolve_dataset_loader() -> Any:
    global load_dataset
    if load_dataset is None:
        from datasets import load_dataset as hf_load_dataset

        load_dataset = hf_load_dataset
    return load_dataset


def _resolve_auto_tokenizer() -> Any:
    global AutoTokenizer
    if AutoTokenizer is None:
        import transformers

        AutoTokenizer = transformers.AutoTokenizer
    return AutoTokenizer


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    uids: list[str]


class DataModule:
    """Cycles through configured shards and yields token batches."""

    def __init__(self, cfg: DataConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        self._seed = int(seed)
        self._rng = random.Random(self._seed)  # noqa: S311  # nosec B311 - deterministic batches
        self._dataset_cache: dict[tuple[str, str, str, bool, str | None], object] = {}
        self._packed = bool(getattr(cfg, "packed", False))
        self._packed_cache: dict[str, np.memmap] = {}
        auto_tokenizer = _resolve_auto_tokenizer()
        self.tokenizer = auto_tokenizer.from_pretrained(
            cfg.tokenizer,
            revision=cfg.hf_revision,
        )  # nosec B615 - revision pinned via config
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset deterministic shard sampling."""

        if seed is not None:
            self._seed = int(seed)
        self._rng = random.Random(self._seed)  # noqa: S311  # nosec B311 - deterministic batches

    def _load_dataset(self, shard: DatasetShard):
        """Load a dataset (streaming or cached) with backward-compatible shard.split parsing.

        Supported shard.split formats:
        - "train" (regular split)
        - "<config>" (dataset config/subset; defaults split to "train")
        - "<config>:<split>" (explicit config + split)
        """
        revision = shard.revision or self.cfg.hf_revision
        streaming = bool(getattr(self.cfg, "streaming", True))
        split_raw = str(shard.split or "train")
        cache_dir = shard.cache_path or None
        cache_key = (shard.name, split_raw, revision, streaming, cache_dir)
        cached = self._dataset_cache.get(cache_key)
        if cached is not None:
            return cached
        dataset_loader = _resolve_dataset_loader()
        if ":" in split_raw:
            cfg_name, split_name = split_raw.split(":", 1)
            dataset = dataset_loader(  # nosec B615 - revision pinned via config
                shard.name,
                cfg_name,
                split=split_name,
                streaming=streaming,
                revision=revision,
                cache_dir=cache_dir,
            )
            self._dataset_cache[cache_key] = dataset
            return dataset
        try:
            dataset = dataset_loader(  # nosec B615 - revision pinned via config
                shard.name,
                split=split_raw,
                streaming=streaming,
                revision=revision,
                cache_dir=cache_dir,
            )
            self._dataset_cache[cache_key] = dataset
            return dataset
        except ValueError as exc:
            # Common case: configs historically used DatasetShard.split to store the dataset config
            # (e.g., wikitext-2-raw-v1). Detect and retry with a default split.
            msg = str(exc)
            if "Config name is missing" in msg or "available configs" in msg:
                dataset = dataset_loader(  # nosec B615 - revision pinned via config
                    shard.name,
                    split_raw,
                    split="train",
                    streaming=streaming,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                self._dataset_cache[cache_key] = dataset
                return dataset
            raise

    @dataclass
    class _ShardStream:
        module: DataModule
        shard: DatasetShard
        iterator: Iterator[TokenBatch]

        def next_batch(self) -> TokenBatch:
            try:
                return next(self.iterator)
            except StopIteration:
                # Restart the shard stream when it runs out (streaming datasets are often finite).
                self.iterator = iter(self.module._shard_iter(self.shard))
                return next(self.iterator)

    class _BatchIterable:
        def __init__(self, module: DataModule, max_tokens: int | None) -> None:
            self.module = module
            self.max_tokens = max_tokens
            self._iter: Iterator[TokenBatch] | None = None

        def __iter__(self) -> Iterator[TokenBatch]:
            self._iter = self.module._batch_generator(self.max_tokens)
            return self

        def __next__(self) -> TokenBatch:
            if self._iter is None:
                self._iter = self.module._batch_generator(self.max_tokens)
            return next(self._iter)

    def batches(self, max_tokens: int | None = None) -> Iterable[TokenBatch]:
        """Return a re-iterable object so training/eval can get fresh iterators."""
        return DataModule._BatchIterable(self, max_tokens)

    def _batch_generator(self, max_tokens: int | None) -> Iterator[TokenBatch]:
        if self._packed:
            yield from self._packed_batch_generator(max_tokens)
            return
        budget = max_tokens
        healing_remaining = self.cfg.healing_tokens if self.cfg.healing_shards else None

        main_streams = [
            DataModule._ShardStream(self, shard, iter(self._shard_iter(shard)))
            for shard in self.cfg.shards
        ]
        main_weights = [float(shard.weight) for shard in self.cfg.shards]
        healing_streams = (
            [
                DataModule._ShardStream(self, shard, iter(self._shard_iter(shard)))
                for shard in self.cfg.healing_shards
            ]
            if healing_remaining is not None and self.cfg.healing_shards
            else []
        )
        healing_weights = [float(shard.weight) for shard in self.cfg.healing_shards]

        def pick_stream(
            streams: list[DataModule._ShardStream], weights: list[float]
        ) -> DataModule._ShardStream:
            if len(streams) == 1:
                return streams[0]
            idx = self._rng.choices(range(len(streams)), weights=weights, k=1)[0]
            return streams[idx]

        while True:
            if healing_remaining is not None and healing_streams:
                batch = pick_stream(healing_streams, healing_weights).next_batch()
            else:
                batch = pick_stream(main_streams, main_weights).next_batch()
            yield batch
            tokens = batch.input_ids.numel()
            if healing_remaining is not None:
                healing_remaining -= tokens
                if healing_remaining <= 0:
                    healing_remaining = None
                    continue
            if budget is not None:
                budget -= tokens
                if budget <= 0:
                    return

    def _packed_batch_generator(self, max_tokens: int | None) -> Iterator[TokenBatch]:
        split = getattr(self.cfg, "packed_split", None) or "train"
        tokens = self._packed_tokens(split)
        seq_len = int(self.cfg.seq_len)
        batch_size = max(1, int(self.cfg.batch_size))
        max_start = int(tokens.shape[0]) - seq_len - 1
        if max_start <= 0:
            raise ValueError("Packed token stream is too short for seq_len.")
        budget = max_tokens
        while True:
            starts = [self._rng.randrange(0, max_start) for _ in range(batch_size)]
            rows = [np.array(tokens[s : s + seq_len], dtype=np.int64) for s in starts]
            batch = np.stack(rows, axis=0)
            input_ids = torch.from_numpy(batch)
            attention_mask = torch.ones_like(input_ids)
            uids = [f"packed-{split}-{s}" for s in starts]
            yield TokenBatch(input_ids=input_ids, attention_mask=attention_mask, uids=uids)
            if budget is not None:
                budget -= seq_len * batch_size
                if budget <= 0:
                    return

    def _packed_tokens(self, split: str) -> np.memmap:
        split_key = "val" if split == "val" else "train"
        cached = self._packed_cache.get(split_key)
        if cached is not None:
            return cached
        train_path = getattr(self.cfg, "packed_train_path", None)
        val_path = getattr(self.cfg, "packed_val_path", None)
        path = train_path if split_key == "train" else (val_path or train_path)
        if not path:
            raise ValueError("Packed token path is missing in DataConfig.")
        dtype_name = getattr(self.cfg, "packed_dtype", "uint16")
        dtype = np.uint16 if dtype_name == "uint16" else np.int32
        token_path = Path(path).expanduser()
        if not token_path.is_absolute():
            packed_root = os.environ.get("TEVO_PACKED_ROOT")
            if packed_root:
                parts = token_path.parts
                if parts and parts[0] == "runs":
                    token_path = Path(packed_root, *parts[1:])
                else:
                    token_path = Path(packed_root) / token_path
            else:
                token_path = Path.cwd() / token_path
        if not token_path.exists():
            raise FileNotFoundError(f"Packed token file not found: {token_path}")
        tokens = np.memmap(token_path, dtype=dtype, mode="r")
        self._packed_cache[split_key] = tokens
        return tokens

    def _shard_iter(self, shard: DatasetShard) -> Iterable[TokenBatch]:
        dataset = self._load_dataset(shard)
        tokenizer = self.tokenizer
        batch_size = max(1, int(self.cfg.batch_size))
        uids: list[str] = []
        input_ids: list[torch.Tensor] = []
        attention_mask: list[torch.Tensor] = []
        for idx, sample in enumerate(dataset):
            text = (
                sample.get("text")
                or sample.get("content")
                or sample.get("question")
                or "placeholder"
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=self.cfg.seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            # Skip samples that yield <2 non-padding tokens; they contribute no
            # next-token prediction targets after shifting.
            mask = encoded.get("attention_mask")
            if isinstance(mask, torch.Tensor) and int(mask.sum().item()) < 2:
                continue
            uids.append(f"{shard.name}-{shard.split}-{idx}")
            input_ids.append(encoded["input_ids"].to(dtype=torch.long))
            attention_mask.append(encoded["attention_mask"].to(dtype=torch.long))
            if len(uids) >= batch_size:
                yield TokenBatch(
                    input_ids=torch.cat(input_ids, dim=0),
                    attention_mask=torch.cat(attention_mask, dim=0),
                    uids=uids,
                )
                uids = []
                input_ids = []
                attention_mask = []
        if uids:
            yield TokenBatch(
                input_ids=torch.cat(input_ids, dim=0),
                attention_mask=torch.cat(attention_mask, dim=0),
                uids=uids,
            )
