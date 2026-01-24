"""Prepare packed-token streams for NanoGPT-style benchmarking."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import typer
from datasets import load_dataset
from transformers import AutoTokenizer

app = typer.Typer(help="Build packed-token train/val files from a text dataset.")


def _iter_texts(dataset: Iterable[dict]) -> Iterable[str]:
    for sample in dataset:
        text = (
            sample.get("text")
            or sample.get("content")
            or sample.get("question")
            or ""
        )
        if text:
            yield text


def _write_stream(
    *,
    texts: Iterable[str],
    tokenizer: AutoTokenizer,
    fh,
    dtype: np.dtype,
    append_eos: bool,
    max_tokens: int | None,
    max_samples: int | None,
    rng: random.Random,
    split_tag: str,
) -> int:
    eos_id = tokenizer.eos_token_id
    total_tokens = 0
    for idx, text in enumerate(texts, start=1):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if append_eos and eos_id is not None:
            ids.append(int(eos_id))
        if not ids:
            continue
        arr = np.asarray(ids, dtype=dtype)
        arr.tofile(fh)
        total_tokens += int(arr.size)
        if idx % 1000 == 0:
            typer.echo(f"[{split_tag}] processed {idx} docs, tokens={total_tokens}")
        if max_samples is not None and idx >= max_samples:
            break
        if max_tokens is not None and total_tokens >= max_tokens:
            break
    return total_tokens


@app.command()
def main(
    dataset: str = typer.Option(..., help="HF dataset name (e.g., openwebtext)."),
    out_dir: Path = typer.Option(Path("runs/packed"), help="Output directory."),
    dataset_config: str | None = typer.Option(None, help="Optional dataset config."),
    train_split: str = typer.Option("train", help="Train split name."),
    val_split: str | None = typer.Option(None, help="Optional validation split name."),
    val_fraction: float = typer.Option(0.0005, help="Holdout fraction if val split missing."),
    tokenizer_name: str = typer.Option("gpt2", help="Tokenizer name."),
    hf_revision: str = typer.Option("main", help="Dataset revision."),
    streaming: bool = typer.Option(True, help="Use streaming dataset loader."),
    seed: int = typer.Option(1234, help="RNG seed for train/val split."),
    dtype: str = typer.Option("uint16", help="Output dtype: uint16 or int32."),
    append_eos: bool = typer.Option(True, help="Append EOS token after each document."),
    max_samples: int | None = typer.Option(None, help="Limit number of docs per split."),
    max_train_tokens: int | None = typer.Option(None, help="Cap train tokens for smoke runs."),
    max_val_tokens: int | None = typer.Option(None, help="Cap val tokens for smoke runs."),
) -> None:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    dtype_map = {"uint16": np.uint16, "int32": np.int32}
    if dtype not in dtype_map:
        raise typer.BadParameter("dtype must be uint16 or int32")
    np_dtype = dtype_map[dtype]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=hf_revision)
    if tokenizer.eos_token_id is None:
        typer.echo("Tokenizer missing eos_token_id; append_eos will be ignored.")

    rng = random.Random(int(seed))

    def load_split(split: str):
        return load_dataset(  # nosec B615 - revision pinned via config
            dataset,
            dataset_config,
            split=split,
            streaming=streaming,
            revision=hf_revision,
        )

    typer.echo("Loading train split...")
    train_dataset = load_split(train_split)

    start = time.time()
    train_tokens = 0
    val_tokens = 0

    if val_split:
        typer.echo("Loading val split...")
        val_dataset = load_split(val_split)
        with train_path.open("wb") as train_fh:
            train_tokens = _write_stream(
                texts=_iter_texts(train_dataset),
                tokenizer=tokenizer,
                fh=train_fh,
                dtype=np_dtype,
                append_eos=append_eos,
                max_tokens=max_train_tokens,
                max_samples=max_samples,
                rng=rng,
                split_tag="train",
            )
        with val_path.open("wb") as val_fh:
            val_tokens = _write_stream(
                texts=_iter_texts(val_dataset),
                tokenizer=tokenizer,
                fh=val_fh,
                dtype=np_dtype,
                append_eos=append_eos,
                max_tokens=max_val_tokens,
                max_samples=max_samples,
                rng=rng,
                split_tag="val",
            )
    else:
        typer.echo("No val split provided; creating holdout with val_fraction.")
        with train_path.open("wb") as train_fh, val_path.open("wb") as val_fh:
            for idx, text in enumerate(_iter_texts(train_dataset), start=1):
                ids = tokenizer.encode(text, add_special_tokens=False)
                eos_id = tokenizer.eos_token_id
                if append_eos and eos_id is not None:
                    ids.append(int(eos_id))
                if not ids:
                    continue
                arr = np.asarray(ids, dtype=np_dtype)
                if rng.random() < val_fraction:
                    arr.tofile(val_fh)
                    val_tokens += int(arr.size)
                    if max_val_tokens is not None and val_tokens >= max_val_tokens:
                        break
                else:
                    arr.tofile(train_fh)
                    train_tokens += int(arr.size)
                    if max_train_tokens is not None and train_tokens >= max_train_tokens:
                        break
                if idx % 1000 == 0:
                    typer.echo(
                        f"[split] docs={idx} train_tokens={train_tokens} val_tokens={val_tokens}"
                    )
                if max_samples is not None and idx >= max_samples:
                    break

    elapsed = time.time() - start
    metadata = {
        "dataset": dataset,
        "dataset_config": dataset_config,
        "train_split": train_split,
        "val_split": val_split,
        "val_fraction": val_fraction,
        "tokenizer": tokenizer_name,
        "revision": hf_revision,
        "dtype": dtype,
        "append_eos": append_eos,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": elapsed,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    typer.echo(f"Wrote {train_path} and {val_path} (train_tokens={train_tokens})")


if __name__ == "__main__":
    app()
