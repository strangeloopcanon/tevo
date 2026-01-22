import random

import yaml

from transformer_evolution_llm import template_mutation as tm
from transformer_evolution_llm.dsl import ArchitectureSpec
from transformer_evolution_llm.template_mutation import (
    MutationTemplate,
    _generate_random_template,
    apply_template_mutation,
    apply_template_mutation_with_name,
    configure_template_learning,
    record_template_result,
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
