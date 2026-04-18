"""Create smaller Parameter Golf shards for fast scout searches."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

app = typer.Typer(help="Prepare smaller Parameter Golf scout shards.", no_args_is_help=True)

HEADER_INTS = 256
HEADER_DTYPE = np.dtype("<i4")
TOKEN_DTYPE = np.dtype("<u2")
HEADER_BYTES = HEADER_INTS * HEADER_DTYPE.itemsize
MAGIC = 20240520
VERSION = 1


def _sorted_glob(pattern: str) -> list[Path]:
    return [Path(item) for item in sorted(glob.glob(pattern))]


def _read_token_count(path: Path) -> int:
    header = np.fromfile(path, dtype=HEADER_DTYPE, count=HEADER_INTS)
    if header.size != HEADER_INTS:
        raise ValueError(f"Incomplete header: {path}")
    if int(header[0]) != MAGIC or int(header[1]) != VERSION:
        raise ValueError(f"Unexpected shard header: {path}")
    return int(header[2])


def _copy_prefix_tokens(src_paths: list[Path], dst_dir: Path, prefix: str, token_budget: int) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    remaining = max(int(token_budget), 0)
    written: list[Path] = []
    shard_idx = 0
    for src_path in src_paths:
        if remaining <= 0:
            break
        available = _read_token_count(src_path)
        take = min(available, remaining)
        if take <= 0:
            continue
        tokens = np.memmap(
            src_path,
            dtype=TOKEN_DTYPE,
            mode="r",
            offset=HEADER_BYTES,
            shape=(take,),
        )
        out_path = dst_dir / f"{prefix}_{shard_idx:06d}.bin"
        header = np.zeros(HEADER_INTS, dtype=HEADER_DTYPE)
        header[0] = MAGIC
        header[1] = VERSION
        header[2] = take
        with out_path.open("wb") as handle:
            header.tofile(handle)
            np.asarray(tokens, dtype=TOKEN_DTYPE).tofile(handle)
        written.append(out_path)
        remaining -= take
        shard_idx += 1
    return written


@app.command("prepare")
def prepare_cmd(
    train_glob: Annotated[str, typer.Option(help="Input train shard glob.")],
    val_glob: Annotated[str, typer.Option(help="Input validation shard glob.")],
    out_dir: Annotated[Path, typer.Option(help="Output directory for smaller scout shards.")],
    train_tokens: Annotated[int, typer.Option(help="Total train tokens to keep.")] = 8_388_608,
    val_tokens: Annotated[int, typer.Option(help="Total validation tokens to keep.")] = 1_048_576,
) -> None:
    """Write smaller train/validation shard prefixes for scout runs."""
    train_paths = _sorted_glob(train_glob)
    val_paths = _sorted_glob(val_glob)
    if not train_paths:
        raise FileNotFoundError(f"No train shards found for {train_glob}")
    if not val_paths:
        raise FileNotFoundError(f"No validation shards found for {val_glob}")

    written_train = _copy_prefix_tokens(train_paths, out_dir, "train", train_tokens)
    written_val = _copy_prefix_tokens(val_paths, out_dir, "val", val_tokens)
    typer.echo(
        f"Wrote {len(written_train)} train shard(s) and {len(written_val)} val shard(s) to {out_dir}"
    )


if __name__ == "__main__":
    app()
