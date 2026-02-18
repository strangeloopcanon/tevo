from __future__ import annotations

from transformer_evolution_llm.ablation import apply_ablation
from transformer_evolution_llm.dsl import ArchitectureSpec, MoEFFNConfig, RetroConfig, SSMConfig


def _base_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        model={
            "name": "ablation-test",
            "emb": {"dim": 32, "vocab": 256},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 2, "head_dim": 16, "kv_groups": 2},
                    "ffn": {"type": "dense", "hidden": 64},
                }
            ],
            "head": {"tie_embeddings": True, "vocab": 256},
        },
        train={
            "lr": 1e-3,
            "warmup": 10,
            "clip": 1.0,
            "bf16": True,
            "grad_ckpt": False,
            "max_tokens": 128,
        },
        data={
            "tokenizer": "hf-internal-testing/tiny-random-gpt2",
            "seq_len": 16,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "lighteval/mgsm", "split": "train", "weight": 1.0}],
        },
    )


def test_ablation_retro_off_removes_retro() -> None:
    spec = _base_spec()
    block = spec.model.blocks[0]
    block.extras.append(
        RetroConfig(memory_tokens=8, stride=2, aggregator="gate", gating_weight=0.25)
    )
    mutated = apply_ablation(spec, "retro_off")
    assert all(getattr(extra, "type", None) != "retro" for extra in mutated.model.blocks[0].extras)


def test_ablation_kv_groups_to_dense_sets_kv_groups_one() -> None:
    spec = _base_spec()
    mutated = apply_ablation(spec, "kv_groups_to_dense")
    attn = mutated.model.blocks[0].attn
    assert attn is not None
    assert attn.kv_groups == 1


def test_ablation_remove_ssm_drops_ssm_branch() -> None:
    spec = _base_spec()
    spec.model.blocks[0].ssm = SSMConfig(
        kind="mamba2", d_state=8, d_conv=4, dt_rank=4, chunk=16, gate=0.1
    )
    mutated = apply_ablation(spec, "remove_ssm")
    assert mutated.model.blocks[0].ssm is None


def test_ablation_moe_to_dense_replaces_moe_ffn() -> None:
    spec = _base_spec()
    dim = spec.model.emb.dim
    spec.model.blocks[0].ffn = MoEFFNConfig(
        hidden=dim * 4,
        n_experts=4,
        k=2,
        capacity_factor=1.2,
        balance=0.05,
        shared=1,
    )
    mutated = apply_ablation(spec, "moe_to_dense")
    ffn = mutated.model.blocks[0].ffn
    assert getattr(ffn, "type", "") == "dense"


def test_ablation_mask_off_sets_filter_mode_none() -> None:
    spec = _base_spec()
    spec.train.optimizer.update_filter.mode = "topk"
    mutated = apply_ablation(spec, "mask_off")
    assert mutated.train.optimizer.update_filter.mode == "none"


def test_ablation_mask_ratio_one_sets_keep_ratio_to_one() -> None:
    spec = _base_spec()
    spec.train.optimizer.update_filter.keep_ratio = 0.4
    mutated = apply_ablation(spec, "mask_ratio_one")
    assert mutated.train.optimizer.update_filter.keep_ratio == 1.0


def test_ablation_optimizer_to_adamw_forces_family() -> None:
    spec = _base_spec()
    spec.train.optimizer.name = "muon"
    spec.train.optimizer.muon_momentum = 0.9
    mutated = apply_ablation(spec, "optimizer_to_adamw")
    assert mutated.train.optimizer.name == "adamw"
    assert mutated.train.optimizer.muon_momentum is None
