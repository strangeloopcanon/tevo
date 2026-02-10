import torch

from transformer_evolution_llm.dsl import (
    ArchitectureSpec,
    BranchRouterConfig,
    ChunkMemoryConfig,
    CustomModuleConfig,
    GatedModuleConfig,
    HyperConnectionsConfig,
    LayerScaleConfig,
    LookupMemoryConfig,
    MemoryTokensConfig,
    MoEFFNConfig,
    RetroConfig,
    SSMConfig,
)
from transformer_evolution_llm.models import EvolutionModel


def test_evolution_model_forward(tiny_spec: ArchitectureSpec) -> None:
    model = EvolutionModel(tiny_spec.model)
    input_ids = torch.randint(0, tiny_spec.model.head.vocab, (2, tiny_spec.data.seq_len))
    logits = model(input_ids)
    assert logits.shape == (2, tiny_spec.data.seq_len, tiny_spec.model.head.vocab)


def test_block_with_moe_ssm_and_extras(tiny_spec: ArchitectureSpec) -> None:
    spec = ArchitectureSpec(**tiny_spec.model_dump(mode="python"))
    block = spec.model.blocks[0]
    block.ffn = MoEFFNConfig(
        type="moe",
        hidden=512,
        n_experts=2,
        k=1,
        capacity_factor=1.1,
        balance=0.05,
        shared=1,
    )
    block.ssm = SSMConfig(kind="mamba2", d_state=8, d_conv=3, dt_rank=4, chunk=16, gate=0.1)
    block.extras = [
        RetroConfig(memory_tokens=32, stride=8, aggregator="gate", gating_weight=0.3),
        MemoryTokensConfig(tokens=8, heads=2, head_dim=32, gating_weight=0.1),
        ChunkMemoryConfig(chunk_size=8, stride=4, heads=2, head_dim=32, gating_weight=0.1),
        LookupMemoryConfig(entries=64, topk=4, key_dim=128, value_dim=256, gating_weight=0.1),
        CustomModuleConfig(name="custom", params={"dim": 64}),
        GatedModuleConfig(targets=["attn", "ffn"], init_weight=0.2, learnable=True),
        LayerScaleConfig(targets=["attn", "ffn", "memory"], init=1e-5, learnable=True),
        BranchRouterConfig(
            targets=["attn", "ffn", "ssm", "memory"],
            router_type="token",
            hidden=64,
            dropout=0.0,
            temperature=1.0,
        ),
    ]
    model = EvolutionModel(spec.model)
    input_ids = torch.randint(0, spec.model.head.vocab, (1, spec.data.seq_len))
    logits = model(input_ids)
    assert torch.isfinite(logits).all()


def test_multihead_attention_with_rope() -> None:
    cfg = ArchitectureSpec(
        model={
            "name": "rope-test",
            "emb": {"dim": 128, "vocab": 32},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 4, "head_dim": 32, "rope": "yarn"},
                    "ffn": {"type": "dense", "hidden": 256},
                }
            ],
            "head": {"vocab": 32, "tie_embeddings": True},
        },
        train={"lr": 1e-3, "warmup": 1, "clip": 1.0},
        data={
            "tokenizer": "gpt2",
            "seq_len": 16,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
        },
    )
    model = EvolutionModel(cfg.model)
    x = torch.randint(0, 32, (2, 16))
    out = model(x)
    assert out.shape == (2, 16, 32)


def test_evolution_model_forward_with_hyper_connections() -> None:
    cfg = ArchitectureSpec(
        model={
            "name": "hyper-test",
            "emb": {"dim": 32, "vocab": 64},
            "hyper": {"streams": 3, "diag_bias": 4.0, "noise_std": 1e-3, "update_scale": 3.0},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 2, "head_dim": 16},
                    "ffn": {"type": "dense", "hidden": 64},
                },
                {
                    "attn": {"kind": "GQA", "heads": 2, "head_dim": 16},
                    "ffn": {"type": "dense", "hidden": 64},
                },
                {
                    "attn": {"kind": "GQA", "heads": 2, "head_dim": 16},
                    "ffn": {"type": "dense", "hidden": 64},
                },
                {
                    "attn": {"kind": "GQA", "heads": 2, "head_dim": 16},
                    "ffn": {"type": "dense", "hidden": 64},
                },
            ],
            "recurrences": [{"start": 1, "end": 4, "adapter": "gated", "concat_prelude": True}],
            "head": {"vocab": 64, "tie_embeddings": True},
        },
        train={"lr": 1e-3, "warmup": 1, "clip": 1.0, "grad_ckpt": False},
        data={
            "tokenizer": "gpt2",
            "seq_len": 8,
            "batch_size": 2,
            "workers": 0,
            "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
        },
    )
    assert isinstance(cfg.model.hyper, HyperConnectionsConfig)
    model = EvolutionModel(cfg.model)
    x = torch.randint(0, 64, (2, cfg.data.seq_len))
    out = model(x)
    assert out.shape == (2, cfg.data.seq_len, 64)
    assert torch.isfinite(out).all()


def test_embedding_conditioned_ffn_forward() -> None:
    cfg = ArchitectureSpec(
        model={
            "name": "embed-ffn-test",
            "emb": {"dim": 32, "vocab": 64},
            "blocks": [
                {
                    "attn": {"kind": "MHA", "heads": 2, "head_dim": 16},
                    "ffn": {
                        "type": "dense",
                        "input_source": "embedding",
                        "hidden": 64,
                        "activation": "swiglu",
                    },
                }
            ],
            "head": {"vocab": 64, "tie_embeddings": True},
        },
        train={"lr": 1e-3, "warmup": 1, "clip": 1.0, "grad_ckpt": False},
        data={
            "tokenizer": "gpt2",
            "seq_len": 8,
            "batch_size": 2,
            "workers": 0,
            "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
        },
    )
    model = EvolutionModel(cfg.model)
    x = torch.randint(0, 64, (2, cfg.data.seq_len))
    out = model(x)
    assert out.shape == (2, cfg.data.seq_len, 64)
    assert torch.isfinite(out).all()


def test_embedding_ffn_branch_forward() -> None:
    cfg = ArchitectureSpec(
        model={
            "name": "embed-ffn-branch-test",
            "emb": {"dim": 32, "vocab": 64},
            "blocks": [
                {
                    "attn": {"kind": "MHA", "heads": 2, "head_dim": 16},
                    "ffn": {"type": "dense", "hidden": 64},
                    "ffn_memory": {
                        "type": "dense",
                        "input_source": "embedding",
                        "hidden": 64,
                        "activation": "swiglu",
                    },
                }
            ],
            "head": {"vocab": 64, "tie_embeddings": True},
        },
        train={"lr": 1e-3, "warmup": 1, "clip": 1.0, "grad_ckpt": False},
        data={
            "tokenizer": "gpt2",
            "seq_len": 8,
            "batch_size": 2,
            "workers": 0,
            "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
        },
    )
    model = EvolutionModel(cfg.model)
    x = torch.randint(0, 64, (2, cfg.data.seq_len))
    out = model(x)
    assert out.shape == (2, cfg.data.seq_len, 64)
    assert torch.isfinite(out).all()
