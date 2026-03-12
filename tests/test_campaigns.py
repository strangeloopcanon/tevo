from __future__ import annotations

from pathlib import Path

import ujson as json

from transformer_evolution_llm.api import save_spec
from transformer_evolution_llm.campaigns import (
    AggregateReport,
    CampaignBudget,
    CampaignLane,
    CampaignManifest,
    aggregate_campaign_submissions,
    build_shortlist_from_report,
    build_submission_bundle,
    config_fingerprint,
    load_campaign_manifest,
    save_campaign_manifest,
    spec_fingerprint,
)
from transformer_evolution_llm.dsl import CustomModuleConfig


def _bridgeable_spec(tiny_spec):
    spec = tiny_spec.model_copy(deep=True)
    spec.model.blocks[0].attn.heads = 4
    spec.model.blocks[0].attn.head_dim = 128
    spec.evolution.objectives = {"ppl_code": "min"}
    return spec


def _campaign_manifest(config_path: Path) -> CampaignManifest:
    return CampaignManifest(
        campaign_id="campaign-001",
        title="Tiny campaign",
        summary="Optional additive campaign test harness.",
        base_config=str(config_path),
        config_fingerprint=config_fingerprint(config_path),
        primary_metric="ppl_code",
        objectives={"ppl_code": "min"},
        budget=CampaignBudget(generations=2, steps=40, eval_batches=2),
        tokenizer="hf-internal-testing/tiny-random-gpt2",
        seq_len=64,
        lanes=[
            CampaignLane(
                lane_id="lane-a",
                title="Lane A",
                focus="Run the shared base config with seed 11.",
                seed=11,
            ),
            CampaignLane(
                lane_id="lane-b",
                title="Lane B",
                focus="Run the shared base config with seed 17.",
                seed=17,
            ),
        ],
    )


def _write_run_artifacts(
    *,
    run_root: Path,
    config_path: Path,
    frontier_entries: list[dict[str, object]],
    seed: int,
    write_manifest: bool = True,
) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    frontier_path = run_root / "frontier.json"
    frontier_path.write_text(json.dumps(frontier_entries, indent=2))
    lineage_path = run_root / "frontier_lineage.json"
    lineage_payload = {
        "nodes": [
            {
                "id": entry["id"],
                "status": entry.get("status", "completed"),
                "mutation_trace": entry.get("mutation_trace", []),
            }
            for entry in frontier_entries
        ],
        "novelty_archive": [],
        "generation_idx": 2,
    }
    lineage_path.write_text(json.dumps(lineage_payload, indent=2))
    if write_manifest:
        manifest_path = run_root / "frontier.manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "config": str(config_path),
                    "generations": 2,
                    "steps": 40,
                    "eval_batches": 2,
                    "seed": seed,
                    "frontier": str(frontier_path),
                    "lineage": str(lineage_path),
                },
                indent=2,
            )
        )
    return run_root


def test_spec_fingerprint_ignores_lineage_metadata(tiny_spec) -> None:
    spec_a = tiny_spec.model_copy(deep=True)
    spec_b = tiny_spec.model_copy(deep=True)
    spec_a.model.blocks[0].origin_id = "origin-a"
    spec_a.model.blocks[0].parent_origin = "root-a"
    spec_b.model.blocks[0].origin_id = "origin-b"
    spec_b.model.blocks[0].parent_origin = "root-b"

    assert spec_fingerprint(spec_a) == spec_fingerprint(spec_b)


def test_build_submission_bundle_uses_existing_run_artifacts(tmp_path: Path, tiny_spec) -> None:
    spec = _bridgeable_spec(tiny_spec)
    config_path = tmp_path / "base.yaml"
    save_spec(spec, config_path)

    campaign_path = tmp_path / "campaign.yaml"
    save_campaign_manifest(_campaign_manifest(config_path), campaign_path)

    better = spec.model_copy(deep=True)
    worse = spec.model_copy(deep=True)
    frontier_entries = [
        {
            "id": "cand-best",
            "parent": None,
            "rung": 2,
            "status": "completed",
            "metrics": {"ppl_code": 4.0, "throughput": 1.2},
            "spec": better.model_dump(mode="python"),
            "mutation_trace": ["mut_a"],
        },
        {
            "id": "cand-worse",
            "parent": "cand-best",
            "rung": 2,
            "status": "completed",
            "metrics": {"ppl_code": 6.0, "throughput": 1.0},
            "spec": worse.model_dump(mode="python"),
            "mutation_trace": ["mut_b"],
        },
    ]
    run_root = _write_run_artifacts(
        run_root=tmp_path / "runs" / "lane-a",
        config_path=config_path,
        frontier_entries=frontier_entries,
        seed=11,
    )
    out_dir = tmp_path / "artifacts" / "lane-a"
    manifest, summary = build_submission_bundle(
        campaign_manifest_path=campaign_path,
        lane_id="lane-a",
        run_path=run_root,
        output_dir=out_dir,
    )

    assert manifest.config_fingerprint == config_fingerprint(config_path)
    assert summary.champion.candidate_id == "cand-best"
    assert summary.champion.bridge_compatible is True
    assert summary.lineage.node_count == 2
    assert summary.lineage.completed_count == 2
    assert summary.bridge_candidate_ids == ["cand-best", "cand-worse"]
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "frontier_top.json").exists()
    assert (out_dir / "lineage_summary.json").exists()
    assert (out_dir / "champion_spec.yaml").exists()
    assert (out_dir / "champion.train_recipe.yaml").exists()


def test_build_submission_bundle_requires_explicit_config_without_run_manifest(
    tmp_path: Path, tiny_spec
) -> None:
    spec = _bridgeable_spec(tiny_spec)
    config_path = tmp_path / "configs" / "base.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    save_spec(spec, config_path)

    campaign_path = tmp_path / "campaigns" / "demo" / "manifest.yaml"
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _campaign_manifest(config_path)
    manifest.base_config = "../../configs/base.yaml"
    save_campaign_manifest(manifest, campaign_path)

    run_root = _write_run_artifacts(
        run_root=tmp_path / "runs" / "lane-a",
        config_path=config_path,
        frontier_entries=[
            {
                "id": "cand-best",
                "parent": None,
                "rung": 2,
                "status": "completed",
                "metrics": {"ppl_code": 4.0},
                "spec": spec.model_dump(mode="python"),
                "mutation_trace": ["mut_a"],
            }
        ],
        seed=11,
        write_manifest=False,
    )

    try:
        build_submission_bundle(
            campaign_manifest_path=campaign_path,
            lane_id="lane-a",
            run_path=run_root,
            output_dir=tmp_path / "artifacts" / "lane-a",
        )
    except ValueError as exc:
        assert "--config" in str(exc)
    else:
        raise AssertionError("expected explicit --config requirement without run manifest")

    submission_manifest, _ = build_submission_bundle(
        campaign_manifest_path=campaign_path,
        lane_id="lane-a",
        run_path=run_root,
        output_dir=tmp_path / "artifacts" / "lane-a",
        config_path=config_path,
    )

    assert submission_manifest.config_path == str(config_path.resolve())


def test_aggregate_campaign_submissions_rejects_mismatched_fingerprint(
    tmp_path: Path, tiny_spec
) -> None:
    spec = _bridgeable_spec(tiny_spec)
    config_path = tmp_path / "base.yaml"
    save_spec(spec, config_path)

    campaign_path = tmp_path / "campaign.yaml"
    save_campaign_manifest(_campaign_manifest(config_path), campaign_path)

    frontier_entries = [
        {
            "id": "cand-one",
            "parent": None,
            "rung": 2,
            "status": "completed",
            "metrics": {"ppl_code": 4.0},
            "spec": spec.model_dump(mode="python"),
            "mutation_trace": ["mut_a"],
        }
    ]
    run_root = _write_run_artifacts(
        run_root=tmp_path / "runs" / "lane-a",
        config_path=config_path,
        frontier_entries=frontier_entries,
        seed=11,
    )
    artifacts_root = tmp_path / "artifacts"
    build_submission_bundle(
        campaign_manifest_path=campaign_path,
        lane_id="lane-a",
        run_path=run_root,
        output_dir=artifacts_root / "lane-a",
    )

    bad_dir = artifacts_root / "lane-b"
    bad_dir.mkdir(parents=True, exist_ok=True)
    copied_manifest = json.loads((artifacts_root / "lane-a" / "manifest.json").read_text())
    copied_manifest["lane_id"] = "lane-b"
    copied_manifest["config_fingerprint"] = "deadbeef"
    (bad_dir / "manifest.json").write_text(json.dumps(copied_manifest, indent=2))
    (bad_dir / "summary.json").write_text((artifacts_root / "lane-a" / "summary.json").read_text())
    (bad_dir / "frontier_top.json").write_text(
        (artifacts_root / "lane-a" / "frontier_top.json").read_text()
    )

    report = aggregate_campaign_submissions(
        campaign_manifest_path=campaign_path,
        artifacts_root=artifacts_root,
        output_dir=artifacts_root / "_aggregate",
    )

    assert report.submitted_lanes == 1
    assert report.rejected_submissions
    assert "config_fingerprint" in report.rejected_submissions[0].reason


def test_build_shortlist_prefers_bridgeable_candidates(tmp_path: Path, tiny_spec) -> None:
    spec = _bridgeable_spec(tiny_spec)
    config_path = tmp_path / "base.yaml"
    save_spec(spec, config_path)

    campaign_path = tmp_path / "campaign.yaml"
    save_campaign_manifest(_campaign_manifest(config_path), campaign_path)
    artifacts_root = tmp_path / "artifacts"

    incompatible = spec.model_copy(deep=True)
    incompatible.model.blocks[0].extras.append(CustomModuleConfig(type="custom", name="probe"))
    compatible = spec.model_copy(deep=True)

    lane_a_run = _write_run_artifacts(
        run_root=tmp_path / "runs" / "lane-a",
        config_path=config_path,
        frontier_entries=[
            {
                "id": "cand-incompatible",
                "parent": None,
                "rung": 2,
                "status": "completed",
                "metrics": {"ppl_code": 3.0},
                "spec": incompatible.model_dump(mode="python"),
                "mutation_trace": ["mut_a"],
            }
        ],
        seed=11,
    )
    build_submission_bundle(
        campaign_manifest_path=campaign_path,
        lane_id="lane-a",
        run_path=lane_a_run,
        output_dir=artifacts_root / "lane-a",
    )

    lane_b_run = _write_run_artifacts(
        run_root=tmp_path / "runs" / "lane-b",
        config_path=config_path,
        frontier_entries=[
            {
                "id": "cand-compatible",
                "parent": None,
                "rung": 2,
                "status": "completed",
                "metrics": {"ppl_code": 4.0},
                "spec": compatible.model_dump(mode="python"),
                "mutation_trace": ["mut_b"],
            }
        ],
        seed=17,
    )
    build_submission_bundle(
        campaign_manifest_path=campaign_path,
        lane_id="lane-b",
        run_path=lane_b_run,
        output_dir=artifacts_root / "lane-b",
    )

    aggregate_campaign_submissions(
        campaign_manifest_path=campaign_path,
        artifacts_root=artifacts_root,
        output_dir=artifacts_root / "_aggregate",
    )
    report_path = artifacts_root / "_aggregate" / "aggregate_report.json"
    assert AggregateReport(**json.loads(report_path.read_text())).submitted_lanes == 2

    shortlist = build_shortlist_from_report(
        aggregate_report_path=report_path,
        output_path=artifacts_root / "_aggregate" / "shortlist.json",
        top_k=2,
        bridge_only=True,
    )

    assert len(shortlist) == 1
    assert shortlist[0].candidate_id == "cand-compatible"
    assert shortlist[0].lane_id == "lane-b"


def test_real_campaign_manifest_matches_repo_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "campaigns" / "campaign-001-d20-nanochat" / "manifest.yaml"
    manifest = load_campaign_manifest(manifest_path)
    config_path = repo_root / manifest.base_config

    assert config_path.exists()
    assert config_fingerprint(config_path) == manifest.config_fingerprint
