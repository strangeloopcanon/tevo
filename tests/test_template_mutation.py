import random

import yaml

from transformer_evolution_llm import template_mutation as tm
from transformer_evolution_llm.dsl import ArchitectureSpec, LookupMemoryConfig
from transformer_evolution_llm.template_mutation import (
    MutationTemplate,
    _generate_random_template,
    apply_template_mutation,
    apply_template_mutation_named_with_name,
    apply_template_mutation_with_name,
    configure_template_learning,
    record_template_result,
)


def _parameter_golf_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        model={
            "name": "pg-template-test",
            "emb": {"dim": 512, "vocab": 1024},
            "blocks": [
                {
                    "attn": {
                        "kind": "GQA",
                        "heads": 8,
                        "head_dim": 64,
                        "kv_groups": 4,
                        "softmax": {"qk_norm": "rms", "softcap": 16.0},
                    },
                    "ffn": {"type": "dense", "hidden": 1024, "activation": "swiglu"},
                }
                for _ in range(6)
            ],
            "head": {"vocab": 1024, "tie_embeddings": True},
        },
        train={"lr": 1e-3, "warmup": 32, "clip": 1.0},
        data={
            "tokenizer": "parameter-golf",
            "seq_len": 512,
            "batch_size": 8,
            "workers": 0,
            "shards": [{"name": "parameter-golf", "split": "train", "weight": 1.0}],
        },
        parameter_golf={
            "train_shards_glob": "runs/parameter_golf/train_*.bin",
            "val_shards_glob": "runs/parameter_golf/val_*.bin",
            "tokenizer_path": "runs/parameter_golf/sp1024.model",
        },
    )


def test_template_mutation_changes_blocks(monkeypatch):
    spec = ArchitectureSpec(
        model={
            "name": "template-test",
            "emb": {"dim": 128, "vocab": 100},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 4, "head_dim": 32},
                    "ffn": {"type": "dense", "hidden": 512},
                }
            ],
            "head": {"vocab": 100, "tie_embeddings": True},
        },
        train={"lr": 1e-3, "warmup": 1, "clip": 1.0},
        data={
            "tokenizer": "gpt2",
            "seq_len": 64,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
        },
    )

    def fake_templates():
        return [
            MutationTemplate(
                name="test-add-extra",
                weight=1.0,
                conditions={},
                actions=[
                    {
                        "add_extra": {
                            "selector": "random",
                            "extra_type": "gated",
                            "params": {"targets": ["attn"], "init_weight": 0.3},
                        }
                    }
                ],
            )
        ]

    monkeypatch.setattr(tm, "load_templates", fake_templates)

    rng = random.Random(0)  # noqa: S311 - deterministic test RNG
    mutated = apply_template_mutation(spec, rng)
    assert mutated.model.blocks[0].extras, "expected template mutation to add an extra module"


def test_template_mutation_can_add_lookup_memory(monkeypatch):
    spec = ArchitectureSpec(
        model={
            "name": "template-test-lookup",
            "emb": {"dim": 128, "vocab": 100},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 4, "head_dim": 32},
                    "ffn": {"type": "dense", "hidden": 512},
                }
            ],
            "head": {"vocab": 100, "tie_embeddings": True},
        },
        train={"lr": 1e-3, "warmup": 1, "clip": 1.0},
        data={
            "tokenizer": "gpt2",
            "seq_len": 64,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
        },
    )

    def fake_templates():
        return [
            MutationTemplate(
                name="test-add-lookup",
                weight=1.0,
                conditions={},
                actions=[
                    {
                        "add_extra": {
                            "selector": "random",
                            "extra_type": "lookup_memory",
                            "params": {"entries": 64, "topk": 4, "key_dim": 32, "value_dim": 64},
                        }
                    }
                ],
            )
        ]

    monkeypatch.setattr(tm, "load_templates", fake_templates)

    rng = random.Random(0)  # noqa: S311 - deterministic test RNG
    mutated = apply_template_mutation(spec, rng)
    assert any(isinstance(extra, LookupMemoryConfig) for extra in mutated.model.blocks[0].extras)


def test_generate_random_template_produces_actions():
    spec = ArchitectureSpec(
        model={
            "name": "auto-template",
            "emb": {"dim": 64, "vocab": 50},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 2, "head_dim": 32},
                    "ffn": {"type": "dense", "hidden": 256},
                }
            ],
            "head": {"vocab": 50, "tie_embeddings": True},
        },
        train={"lr": 1e-3, "warmup": 1, "clip": 1.0},
        data={
            "tokenizer": "gpt2",
            "seq_len": 32,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
        },
    )
    rng = random.Random(42)  # noqa: S311 - deterministic test RNG
    template = _generate_random_template(spec, rng)
    assert template.actions, "auto-generated template should have at least one action"


def test_template_learning_updates_weight_and_persists(tmp_path):
    templates_path = tmp_path / "mutation_templates.yaml"
    templates_path.write_text(
        yaml.safe_dump(
            {
                "templates": [
                    {
                        "name": "test-template",
                        "weight": 1.0,
                        "conditions": {},
                        "actions": [
                            {
                                "add_extra": {
                                    "selector": "random",
                                    "extra_type": "gated",
                                    "params": {"targets": ["attn"], "init_weight": 0.3},
                                }
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        )
    )
    configure_template_learning(enabled=True, path=templates_path, save_every=1)
    try:
        spec = ArchitectureSpec(
            model={
                "name": "template-learn-test",
                "emb": {"dim": 128, "vocab": 100},
                "blocks": [
                    {
                        "attn": {"kind": "GQA", "heads": 4, "head_dim": 32},
                        "ffn": {"type": "dense", "hidden": 512},
                    }
                ],
                "head": {"vocab": 100, "tie_embeddings": True},
            },
            train={"lr": 1e-3, "warmup": 1, "clip": 1.0},
            data={
                "tokenizer": "gpt2",
                "seq_len": 64,
                "batch_size": 1,
                "workers": 0,
                "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
            },
        )
        rng = random.Random(0)  # noqa: S311 - deterministic test RNG
        template_name, mutated = apply_template_mutation_with_name(spec, rng)
        assert template_name == "test-template"
        assert mutated.model.blocks[0].extras
        record_template_result(template_name, delta=1.0)
        payload = yaml.safe_load(templates_path.read_text())
        weight = float(payload["templates"][0]["weight"])
        assert weight > 1.0
    finally:
        configure_template_learning(enabled=False)


def test_template_learning_promotes_dynamic_template(tmp_path, monkeypatch):
    templates_path = tmp_path / "mutation_templates.yaml"
    templates_path.write_text(yaml.safe_dump({"templates": []}, sort_keys=False))
    configure_template_learning(enabled=True, path=templates_path, save_every=1)
    try:
        template = MutationTemplate(
            name="auto-1",
            weight=1.0,
            conditions={},
            actions=[
                {
                    "add_extra": {
                        "selector": "random",
                        "extra_type": "gated",
                        "params": {"targets": ["ffn"], "init_weight": 0.2},
                    }
                }
            ],
        )
        monkeypatch.setattr(tm, "_generate_random_template", lambda spec, rng: template)

        spec = ArchitectureSpec(
            model={
                "name": "template-promote-test",
                "emb": {"dim": 64, "vocab": 50},
                "blocks": [
                    {
                        "attn": {"kind": "GQA", "heads": 2, "head_dim": 32},
                        "ffn": {"type": "dense", "hidden": 256},
                    }
                ],
                "head": {"vocab": 50, "tie_embeddings": True},
            },
            train={"lr": 1e-3, "warmup": 1, "clip": 1.0},
            data={
                "tokenizer": "gpt2",
                "seq_len": 32,
                "batch_size": 1,
                "workers": 0,
                "shards": [{"name": "ag_news", "split": "train", "weight": 1.0}],
            },
        )
        rng = random.Random(0)  # noqa: S311 - deterministic test RNG
        template_name, _ = apply_template_mutation_with_name(spec, rng)
        assert template_name == "auto-1"
        record_template_result(template_name, delta=1.0)
        payload = yaml.safe_load(templates_path.read_text())
        names = [row["name"] for row in payload["templates"]]
        assert "auto-1" in names
    finally:
        configure_template_learning(enabled=False)


def test_parameter_golf_random_template_uses_pg_actions():
    spec = _parameter_golf_spec()
    rng = random.Random(42)  # noqa: S311 - deterministic test RNG
    template = _generate_random_template(spec, rng)
    action_name = next(iter(template.actions[0]))
    assert template.conditions == {"parameter_golf_only": True}
    assert action_name in {"tune_attn", "tune_ffn", "tune_optimizer", "set_recurrence"}


def test_parameter_golf_template_can_tune_attention(monkeypatch):
    spec = _parameter_golf_spec()

    def fake_templates():
        return [
            MutationTemplate(
                name="pg-attn",
                weight=1.0,
                conditions={"parameter_golf_only": True},
                actions=[
                    {
                        "tune_attn": {
                            "selector": "random_dense",
                            "kind": "MLA",
                            "kv_latent_dim": 96,
                            "value_glu": True,
                            "qk_norm": "rms",
                            "softcap": 12.0,
                        }
                    }
                ],
            )
        ]

    monkeypatch.setattr(tm, "load_templates", fake_templates)
    rng = random.Random(0)  # noqa: S311 - deterministic test input
    template_name, mutated = apply_template_mutation_named_with_name(spec, rng, "pg-attn")
    assert template_name == "pg-attn"
    assert any(block.attn and block.attn.kind == "MLA" for block in mutated.model.blocks)
    assert any(
        block.attn and block.attn.softmax and block.attn.softmax.softcap == 12.0
        for block in mutated.model.blocks
    )


def test_parameter_golf_template_can_tune_optimizer(monkeypatch):
    spec = _parameter_golf_spec()

    def fake_templates():
        return [
            MutationTemplate(
                name="pg-opt",
                weight=1.0,
                conditions={"parameter_golf_only": True},
                actions=[
                    {
                        "tune_optimizer": {
                            "optimizer_name": "adamw",
                            "lr": 8.5e-4,
                            "gradient_transform_mode": "normalize",
                            "gradient_transform_ns_steps": 3,
                            "gradient_transform_eps": 1.0e-7,
                            "update_filter_mode": "topk",
                            "update_filter_keep_ratio": 0.6,
                            "update_filter_granularity": "block",
                            "update_filter_block_size": 64,
                            "update_filter_momentum_blend": 0.25,
                            "warmup": 48,
                            "clip": 0.75,
                        }
                    }
                ],
            )
        ]

    monkeypatch.setattr(tm, "load_templates", fake_templates)
    rng = random.Random(0)  # noqa: S311 - deterministic test input
    template_name, mutated = apply_template_mutation_named_with_name(spec, rng, "pg-opt")
    assert template_name == "pg-opt"
    assert mutated.train.optimizer.name == "adamw"
    assert mutated.train.lr == 8.5e-4
    assert mutated.train.warmup == 48
    assert mutated.train.clip == 0.75
    assert mutated.train.optimizer.gradient_transform.mode == "normalize"
    assert mutated.train.optimizer.update_filter.mode == "topk"
    assert mutated.train.optimizer.update_filter.keep_ratio == 0.6


def test_parameter_golf_template_can_set_recurrence(monkeypatch):
    spec = _parameter_golf_spec()

    def fake_templates():
        return [
            MutationTemplate(
                name="pg-rec",
                weight=1.0,
                conditions={"parameter_golf_only": True, "requires_no_recurrence": True},
                actions=[
                    {
                        "set_recurrence": {
                            "tail_blocks": 3,
                            "adapter": "gated",
                            "concat_prelude": True,
                            "train_recurrence": 2,
                            "max_train_recurrence": 4,
                            "test_recurrences": [1, 2, 4],
                        }
                    }
                ],
            )
        ]

    monkeypatch.setattr(tm, "load_templates", fake_templates)
    rng = random.Random(0)  # noqa: S311 - deterministic test input
    template_name, mutated = apply_template_mutation_named_with_name(spec, rng, "pg-rec")
    assert template_name == "pg-rec"
    assert len(mutated.model.recurrences) == 1
    recurrence = mutated.model.recurrences[0]
    assert recurrence.start == 3
    assert recurrence.end == 6
    assert recurrence.adapter == "gated"
    assert recurrence.train_recurrence == 2
    assert recurrence.max_train_recurrence == 4
