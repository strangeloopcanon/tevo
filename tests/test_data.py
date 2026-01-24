from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from transformer_evolution_llm import dsl
from transformer_evolution_llm.data import DataModule


class DummyDataset:
    def __iter__(self) -> Iterator[dict[str, str]]:
        for i in range(3):
            yield {"text": f"sample-{i}"}


class DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"  # noqa: S105 - tokenizer sentinel

    def __call__(self, *_args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        length = kwargs["max_length"]
        ids = torch.zeros((1, length), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_data_module_batches(monkeypatch):
    from transformer_evolution_llm import data as data_module

    class Loader:
        @staticmethod
        def from_pretrained(_name: str, **_kwargs: Any) -> DummyTokenizer:
            return DummyTokenizer()

    monkeypatch.setattr(data_module, "AutoTokenizer", Loader)
    monkeypatch.setattr(data_module, "load_dataset", lambda *args, **kwargs: DummyDataset())

    cfg = dsl.DataConfig(
        tokenizer="hf-internal-testing/tiny-random-gpt2",
        seq_len=8,
        batch_size=1,
        workers=0,
        shards=[dsl.DatasetShard(name="dummy", split="train", weight=1.0)],
    )
    module = DataModule(cfg)
    batch = next(module.batches(max_tokens=16))
    assert batch.input_ids.shape[-1] == cfg.seq_len


def test_data_module_accepts_config_in_split(monkeypatch):
    from transformer_evolution_llm import data as data_module

    class Loader:
        @staticmethod
        def from_pretrained(_name: str, **_kwargs: Any) -> DummyTokenizer:
            return DummyTokenizer()

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_load_dataset(*args: Any, **kwargs: Any) -> DummyDataset:
        calls.append((args, kwargs))
        if kwargs.get("split") == "wikitext-2-raw-v1":
            raise ValueError(
                "Config name is missing.\n"
                "Please pick one among the available configs: ['wikitext-2-raw-v1']"
            )
        return DummyDataset()

    monkeypatch.setattr(data_module, "AutoTokenizer", Loader)
    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    cfg = dsl.DataConfig(
        tokenizer="hf-internal-testing/tiny-random-gpt2",
        seq_len=8,
        batch_size=1,
        workers=0,
        shards=[dsl.DatasetShard(name="wikitext", split="wikitext-2-raw-v1", weight=1.0)],
    )
    module = DataModule(cfg)
    _ = next(module.batches(max_tokens=16))

    assert len(calls) >= 2
    retry_args, retry_kwargs = calls[1]
    assert retry_args == ("wikitext", "wikitext-2-raw-v1")
    assert retry_kwargs["split"] == "train"


def test_data_module_packed_batches(monkeypatch, tmp_path):
    from transformer_evolution_llm import data as data_module

    class Loader:
        @staticmethod
        def from_pretrained(_name: str, **_kwargs: Any) -> DummyTokenizer:
            return DummyTokenizer()

    monkeypatch.setattr(data_module, "AutoTokenizer", Loader)

    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    np.arange(0, 256, dtype=np.uint16).tofile(train_path)
    np.arange(0, 128, dtype=np.uint16).tofile(val_path)

    cfg = dsl.DataConfig(
        tokenizer="hf-internal-testing/tiny-random-gpt2",
        seq_len=8,
        batch_size=2,
        workers=0,
        shards=[],
        packed=True,
        packed_train_path=str(train_path),
        packed_val_path=str(val_path),
        packed_split="train",
    )
    module = DataModule(cfg)
    batch = next(module.batches(max_tokens=16))
    assert batch.input_ids.shape == (2, 8)
