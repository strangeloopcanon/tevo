"""Evolution loop orchestration.

Sub-modules split for readability:
- scoring.py: structural_distance, graph_entropy, prior_distance, composite metrics
"""

from __future__ import annotations

import math
import random
import uuid
from hashlib import sha256
from json import dumps as json_dumps
from pathlib import Path
from typing import Any

import ujson as json
from rich.console import Console
from rich.table import Table

from .candidates import Candidate, ObjectiveDirection, ParetoFrontier
from .crossover import (
    ParentKey,
    aligned_splice_blocks,
    merge_checkpoints_with_report,
)
from .data import DataModule
from .dsl import ArchitectureSpec, CompositeMetricConfig, EvolutionConfig
from .evaluation import StaticChecker, estimate_params
from .mutations import (
    load_mutation_plugins,
    mutate_with_trace,
    mutation_names,
    register_template_mutations,
)
from .parameter_golf import (
    ParameterGolfDataModule,
    estimate_artifact_total_bytes_for_spec,
    estimate_calibrated_artifact_total_bytes_for_spec,
)
from .parameter_golf_export import official_submission_metrics
from .parameter_golf_seeded import (
    incubator_promotion_summary,
    motif_signature,
    seed_lane_metadata,
)
from .scoring import (
    archive_novelty,
    artifact_budget_edge_score,
    artifact_budget_fill_score,
    artifact_budget_utilization,
    behavioral_descriptor,
    complexity_score,
    compute_composite,
    default_composites,
    default_objectives,
    graph_entropy,
    merge_composites,
    prior_distance,
    structural_distance,
)
from .simulators import evaluator_for_mode
from .template_mutation import (
    configure_template_learning,
    flush_template_learning,
    record_template_result,
)
from .trainer import FullWeightTrainer

console = Console()

ObjectiveDir = dict[str, ObjectiveDirection]


class EvolutionRunner:
    """Coordinates mutation, evaluation, and frontier tracking."""

    @staticmethod
    def _strip_lineage_metadata(payload: Any) -> Any:
        if isinstance(payload, dict):
            return {
                key: EvolutionRunner._strip_lineage_metadata(value)
                for key, value in payload.items()
                if key not in {"origin_id", "parent_origin"}
            }
        if isinstance(payload, list):
            return [EvolutionRunner._strip_lineage_metadata(value) for value in payload]
        return payload

    @staticmethod
    def _spec_fingerprint(spec: ArchitectureSpec) -> str:
        payload = spec.model_dump(mode="json")
        payload = EvolutionRunner._strip_lineage_metadata(payload)
        blob = json_dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _empty_device_cache() -> None:
        try:
            import torch
        except Exception:
            return
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    @staticmethod
    def _is_resource_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                "out of memory",
                "cuda error",
                "mps backend out of memory",
                "resource exhausted",
                "allocation failed",
            )
        )

    def __init__(
        self,
        base_spec: ArchitectureSpec,
        evolution_cfg: EvolutionConfig,
        mode: str = "simulate",
        objective_dir: ObjectiveDir | None = None,
        seed: int = 0,
        score_weight_overrides: dict[str, float] | None = None,
    ) -> None:
        self.base_spec = base_spec
        self.cfg = evolution_cfg
        self.mode = mode
        config_objectives = getattr(self.cfg, "objectives", None)
        if objective_dir is not None:
            self.objective_dir = objective_dir
        elif config_objectives is not None:
            self.objective_dir = config_objectives
        else:
            pareto = getattr(self.cfg, "pareto_objectives", None) or []
            if isinstance(pareto, list) and pareto:
                defaults = default_objectives()
                self.objective_dir = {name: defaults.get(name, "max") for name in pareto}
            else:
                self.objective_dir = default_objectives()
        self.score_weights = {
            k: (1.0 if v == "max" else -1.0)
            * (score_weight_overrides.get(k, 1.0) if score_weight_overrides else 1.0)
            for k, v in self.objective_dir.items()
        }
        load_mutation_plugins(list(getattr(self.cfg, "mutation_plugins", []) or []))
        if bool(getattr(self.cfg, "register_template_entries", True)):
            register_template_mutations()
        registry_names = set(mutation_names())
        raw_allowlist = list(getattr(self.cfg, "mutation_allowlist", []) or [])
        parsed_allowlist: list[str] = []
        invalid_allowlist: list[str] = []
        for raw_name in raw_allowlist:
            name = str(raw_name).strip()
            if not name:
                continue
            if name in registry_names:
                parsed_allowlist.append(name)
            else:
                invalid_allowlist.append(name)
        if invalid_allowlist:
            console.print(
                "[yellow]Warning:[/] ignoring unknown mutation_allowlist entries: "
                + ", ".join(sorted(set(invalid_allowlist)))
            )
        self._mutation_allowlist: list[str] | None = None
        if parsed_allowlist:
            self._mutation_allowlist = sorted(set(parsed_allowlist))
        elif raw_allowlist:
            msg = "evolution.mutation_allowlist resolved to an empty registered set."
            raise ValueError(msg)

        self.mutation_weights: dict[str, float] | None = None
        raw_weights = getattr(self.cfg, "mutation_weights", None)
        if isinstance(raw_weights, dict):
            parsed_weights: dict[str, float] = {}
            invalid_weight_names: list[str] = []
            for raw_name, raw_value in raw_weights.items():
                name = str(raw_name).strip()
                if not name:
                    continue
                if name not in registry_names:
                    invalid_weight_names.append(name)
                    continue
                if self._mutation_allowlist is not None and name not in self._mutation_allowlist:
                    continue
                parsed_weights[name] = float(raw_value)
            if invalid_weight_names:
                console.print(
                    "[yellow]Warning:[/] ignoring unknown mutation_weights entries: "
                    + ", ".join(sorted(set(invalid_weight_names)))
                )
            if parsed_weights:
                self.mutation_weights = parsed_weights
        if self._mutation_allowlist is not None and self.mutation_weights is None:
            self.mutation_weights = dict.fromkeys(self._mutation_allowlist, 1.0)
        self.mutation_steps: int = int(getattr(self.cfg, "mutation_steps", 1) or 1)
        self._adaptive_mutation = bool(getattr(self.cfg, "adaptive_mutation", False))
        self._adaptive_mutation_eta = float(getattr(self.cfg, "adaptive_mutation_eta", 0.1) or 0.1)
        self._adaptive_mutation_min = float(
            getattr(self.cfg, "adaptive_mutation_min_weight", 0.05) or 0.05
        )
        self._adaptive_mutation_max = float(
            getattr(self.cfg, "adaptive_mutation_max_weight", 5.0) or 5.0
        )
        self._mutation_success: dict[str, float] = {}
        self._mutation_counts: dict[str, int] = {}
        self._motif_families: dict[str, set[str]] = {}
        self._novelty_archive: list[list[float]] = []
        self._novelty_k = int(getattr(self.cfg, "novelty_archive_k", 15) or 15)
        self._novelty_archive_max = int(getattr(self.cfg, "novelty_archive_max", 500) or 500)
        self._generation_idx = 0
        self._active_rung0_thresholds: dict[str, float] = dict(
            getattr(self.cfg, "rung0_thresholds", {}) or {}
        )
        self._origin_counter = 0
        self.archive: dict[str, Candidate] = {}
        self.archive_max_elites = int(getattr(self.cfg, "archive_max_elites", 0) or 0)
        if (
            getattr(self.cfg, "parent_selection", "weighted") == "map_elites"
            and self.archive_max_elites <= 0
        ):
            self.archive_max_elites = max(1, int(getattr(self.cfg, "population", 12) or 12))
        # Structural elite retention to keep deeper/MoE-rich candidates alive
        self.structural_elite_k = int(getattr(self.cfg, "structural_elite_k", 2) or 0)
        self.structural_elite_weights: dict[str, float] = {
            "layers": 1.0,
            "moe_blocks": 3.0,
            "selector_blocks": 2.0,
            "embedding_ffn_blocks": 2.0,
        }
        cfg_elite_weights = getattr(self.cfg, "structural_elite_weights", None)
        if isinstance(cfg_elite_weights, dict):
            self.structural_elite_weights.update(
                {
                    k: float(v)
                    for k, v in cfg_elite_weights.items()
                    if k in self.structural_elite_weights
                }
            )
        self.frontier = ParetoFrontier(self.objective_dir)
        self.rng = random.Random(seed)  # noqa: S311  # nosec B311 - seeded per run
        thresholds = self._active_thresholds()

        def _threshold(key: str, default: float) -> float:
            raw = thresholds.get(key)
            if raw is None:
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        self.checker = StaticChecker(
            max_params=_threshold("max_params", 8.0e9),
            max_kv_bytes=_threshold("max_kv_bytes_per_token", 64_000.0),
            min_throughput=_threshold("min_throughput_proxy", 0.5),
        )
        self.trainer = FullWeightTrainer() if mode == "live" else None
        self.data_module: DataModule | ParameterGolfDataModule | None = None
        if mode == "live":
            seed_value = int(getattr(base_spec.train, "seed", 0) or 0)
            try:
                if base_spec.parameter_golf is not None:
                    self.data_module = ParameterGolfDataModule(
                        base_spec.parameter_golf,
                        seq_len=base_spec.data.seq_len,
                        batch_size=base_spec.data.batch_size,
                        seed=seed_value,
                    )
                else:
                    self.data_module = DataModule(base_spec.data, seed=seed_value)
            except TypeError:
                # Allow tests or external shims that provide a simplified DataModule signature.
                if base_spec.parameter_golf is not None:
                    self.data_module = ParameterGolfDataModule(
                        base_spec.parameter_golf,
                        seq_len=base_spec.data.seq_len,
                        batch_size=base_spec.data.batch_size,
                    )
                else:
                    self.data_module = DataModule(base_spec.data)
        self.evaluator = (
            None if mode == "live" else evaluator_for_mode(mode, checker=self.checker, seed=seed)
        )
        configured_composites = getattr(self.cfg, "composite_metrics", []) or []
        self._composite_metrics = self._merge_composites(
            configured_composites, self._default_composites()
        )
        self._seen_spec_fingerprints: set[str] = set()
        self.pool: list[Candidate] = []
        self.counter = 0
        self.checkpoint_dir = Path("runs/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        init_ckpt = getattr(base_spec.train, "init_checkpoint", None)
        self._init_checkpoint: Path | None = Path(init_ckpt) if init_ckpt else None
        self.weight_inheritance = str(getattr(self.cfg, "weight_inheritance", "parent") or "parent")
        # lineage tracking: candidate id -> list of parent ids
        self._parents: dict[str, list[str]] = {}
        self._history: list[Candidate] = []
        # rung schedule ratios relative to configured trainer.steps
        self._rung1_ratio = 0.2
        self._rung2_ratio = 1.0
        self.ppl_stop_threshold = base_spec.train.ppl_stop_threshold
        # Promotion heuristics (live mode) for high-budget candidates
        self._promotion_prob = float(getattr(self.cfg, "promotion_prob", 0.0) or 0.0)
        self._promotion_min_layers = int(getattr(self.cfg, "promotion_min_layers", 0) or 0)
        self._promotion_min_moe_blocks = int(getattr(self.cfg, "promotion_min_moe_blocks", 0) or 0)
        self._promotion_steps_multiplier = float(
            getattr(self.cfg, "promotion_steps_multiplier", 1.0) or 1.0
        )
        self._promotion_tokens_multiplier = float(
            getattr(self.cfg, "promotion_tokens_multiplier", 1.0) or 1.0
        )
        self._promotion_min_router_entropy = float(
            getattr(self.cfg, "promotion_min_router_entropy", 0.0) or 0.0
        )
        self._promotion_min_recurrence_gain = float(
            getattr(self.cfg, "promotion_min_recurrence_gain", 0.0) or 0.0
        )
        self._promotion_max_instability = getattr(self.cfg, "promotion_max_instability", None)
        self._template_learning = bool(getattr(self.cfg, "template_learning", False))
        if self._template_learning:
            save_path_raw = getattr(self.cfg, "template_learning_save_path", None)
            save_path = Path(str(save_path_raw or "configs/mutation_templates.yaml"))
            seed_path_raw = getattr(self.cfg, "template_learning_seed_path", None)
            if seed_path_raw:
                seed_path = Path(str(seed_path_raw))
                if seed_path.exists() and not save_path.exists():
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_path.write_text(seed_path.read_text())
            configure_template_learning(
                enabled=True,
                path=save_path,
                eta=float(getattr(self.cfg, "template_learning_eta", 0.2) or 0.2),
                min_weight=float(getattr(self.cfg, "template_learning_min_weight", 0.05) or 0.05),
                max_weight=float(getattr(self.cfg, "template_learning_max_weight", 5.0) or 5.0),
                max_templates=int(getattr(self.cfg, "template_learning_max_templates", 128) or 128),
                save_every=int(getattr(self.cfg, "template_learning_save_every", 20) or 20),
                promote_min_delta=float(
                    getattr(self.cfg, "template_learning_promote_min_delta", 0.0) or 0.0
                ),
            )
        else:
            configure_template_learning(enabled=False)
        self._assign_missing_origin_ids(self.base_spec)
        self._set_generation(0)

    def _remember_spec(self, spec: ArchitectureSpec) -> None:
        self._seen_spec_fingerprints.add(self._spec_fingerprint(spec))

    def _rebuild_seen_specs(self) -> None:
        self._seen_spec_fingerprints = {
            self._spec_fingerprint(c.spec) for c in self.pool + self._history
        }

    def _new_origin_id(self) -> str:
        self._origin_counter += 1
        return f"o{self._origin_counter:08d}"

    def _assign_missing_origin_ids(self, spec: ArchitectureSpec) -> None:
        for block in spec.model.blocks:
            if getattr(block, "origin_id", None):
                continue
            block.origin_id = self._new_origin_id()
            if getattr(block, "parent_origin", None) is None:
                block.parent_origin = None

    def _resolve_thresholds_for_generation(self, generation: int) -> dict[str, float]:
        thresholds = dict(getattr(self.cfg, "rung0_thresholds", {}) or {})
        schedule_entries: list[tuple[int, dict[str, Any]]] = []
        for step in getattr(self.cfg, "gate_schedule", []) or []:
            if isinstance(step, dict):
                raw_generation = step.get("generation", 0)
                raw_thresholds = step.get("thresholds", {})
            else:
                raw_generation = getattr(step, "generation", 0)
                raw_thresholds = getattr(step, "thresholds", {})
            try:
                resolved_generation = int(raw_generation)
            except (TypeError, ValueError):
                continue
            if not isinstance(raw_thresholds, dict):
                continue
            schedule_entries.append((resolved_generation, raw_thresholds))

        for step_generation, step_thresholds in sorted(schedule_entries, key=lambda item: item[0]):
            if step_generation > int(generation):
                break
            for key, value in step_thresholds.items():
                try:
                    thresholds[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        return thresholds

    def _set_generation(self, generation: int) -> None:
        self._generation_idx = int(max(0, generation))
        self._active_rung0_thresholds = self._resolve_thresholds_for_generation(
            self._generation_idx
        )
        if self.checker is None:
            return
        max_params = float(self._active_rung0_thresholds.get("max_params", self.checker.max_params))
        max_kv = float(
            self._active_rung0_thresholds.get("max_kv_bytes_per_token", self.checker.max_kv_bytes)
        )
        min_tp = float(
            self._active_rung0_thresholds.get("min_throughput_proxy", self.checker.min_throughput)
        )
        self.checker.max_params = max_params
        self.checker.max_kv_bytes = max_kv
        self.checker.min_throughput = min_tp

    def _active_thresholds(self) -> dict[str, float]:
        return dict(self._active_rung0_thresholds)

    def _update_novelty_archive(self, descriptor: list[float]) -> None:
        if not descriptor:
            return
        max_size = max(1, int(self._novelty_archive_max))
        if len(self._novelty_archive) < max_size:
            self._novelty_archive.append(list(descriptor))
            return
        idx = self.rng.randrange(max_size)
        self._novelty_archive[idx] = list(descriptor)

    def _cleanup_seed_state(self, candidate: Candidate) -> None:
        """Remove intermediate crossover seed checkpoints when no longer needed."""
        seed_state_path = candidate.seed_state_path
        if seed_state_path is None:
            return
        if self._init_checkpoint is not None:
            try:
                if seed_state_path.resolve() == self._init_checkpoint.resolve():
                    candidate.seed_state_path = None
                    return
            except OSError:
                pass
        try:
            seed_state_path.unlink(missing_ok=True)
        except OSError:
            pass
        candidate.seed_state_path = None

    def _thresholds_ok(self, candidate: Candidate, *, require_metrics: bool) -> bool:
        if float(candidate.metrics.get("nan_seen", 0.0) or 0.0) > 0.0:
            return False
        stop_reason = candidate.metrics.get("stop_reason_code")
        if stop_reason is not None:
            try:
                stop_code = float(stop_reason)
            except (TypeError, ValueError):
                return False
            if stop_code in (1.0, 2.0):  # high_grad / low_entropy
                return False
        thresholds = self._active_thresholds()
        for key, limit in thresholds.items():
            if not isinstance(key, str):
                continue
            if key.startswith("max_"):
                metric_name = key[len("max_") :]
                metric_val = candidate.metrics.get(metric_name)
                if metric_val is None:
                    if require_metrics:
                        return False
                    continue
                try:
                    metric_f = float(metric_val)
                    limit_f = float(limit)
                except (TypeError, ValueError):
                    if require_metrics:
                        return False
                    continue
                if metric_f > limit_f:
                    return False
                continue
            if not key.startswith("min_"):
                continue
            metric_name = key[len("min_") :]
            if metric_name == "throughput_proxy":
                # StaticChecker already enforces this gate.
                continue
            metric_val = candidate.metrics.get(metric_name)
            if metric_val is None:
                if require_metrics:
                    return False
                continue
            try:
                metric_f = float(metric_val)
                limit_f = float(limit)
            except (TypeError, ValueError):
                if require_metrics:
                    return False
                continue
            if metric_f < limit_f:
                return False
        return True

    def _objective_metrics_ok(self, candidate: Candidate) -> bool:
        if not self._thresholds_ok(candidate, require_metrics=True):
            return False
        for name in self.objective_dir:
            val = candidate.metrics.get(name)
            if val is None:
                return False
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(val_f):
                return False
        return True

    def _score_weight_overrides(self) -> dict[str, float]:
        """Recover unsigned weight overrides from signed score weights."""
        overrides: dict[str, float] = {}
        for metric, direction in self.objective_dir.items():
            weight = self.score_weights.get(metric)
            if weight is None:
                continue
            sign = 1.0 if direction == "max" else -1.0
            overrides[metric] = float(weight) / sign
        return overrides

    def _annotate_official_submission_metrics(self, candidate: Candidate) -> None:
        if candidate.spec.parameter_golf is None:
            return
        edge_kwargs = self._artifact_budget_edge_kwargs(candidate.spec)
        budget_bytes = float(candidate.spec.parameter_golf.artifact_budget_bytes)
        artifact_zlib_bytes = candidate.metrics.get("artifact_zlib_bytes")
        try:
            zlib_bytes = float(artifact_zlib_bytes) if artifact_zlib_bytes is not None else None
        except (TypeError, ValueError):
            zlib_bytes = None
        metrics = official_submission_metrics(
            candidate.spec,
            artifact_zlib_bytes=zlib_bytes,
        )
        if str(candidate.spec.parameter_golf.lane_kind or "exportable") == "incubator":
            metrics["main_track_eligible_est"] = 0.0
            if "main_track_eligible_exact" in metrics:
                metrics["main_track_eligible_exact"] = 0.0
        if (
            str(getattr(candidate.spec.parameter_golf, "eval_protocol", "mid_fidelity"))
            == "scout_fast"
        ):
            metrics["main_track_eligible_est"] = 0.0
            if "main_track_eligible_exact" in metrics:
                metrics["main_track_eligible_exact"] = 0.0
        total_bytes_est = metrics.get("official_submission_total_bytes_est")
        metrics["official_submission_budget_utilization_est"] = artifact_budget_utilization(
            total_bytes_est,
            budget_bytes,
        )
        metrics["official_submission_budget_fill_score_est"] = artifact_budget_fill_score(
            total_bytes_est,
            budget_bytes,
        )
        metrics["official_submission_budget_edge_score_est"] = artifact_budget_edge_score(
            total_bytes_est,
            budget_bytes,
            **edge_kwargs,
        )
        total_bytes_exact = candidate.metrics.get("artifact_total_bytes")
        if total_bytes_exact is not None:
            metrics["artifact_budget_utilization"] = artifact_budget_utilization(
                total_bytes_exact,
                budget_bytes,
            )
            metrics["artifact_budget_fill_score"] = artifact_budget_fill_score(
                total_bytes_exact,
                budget_bytes,
            )
            metrics["artifact_budget_edge_score"] = artifact_budget_edge_score(
                total_bytes_exact,
                budget_bytes,
                **edge_kwargs,
            )
            metrics["official_submission_budget_utilization"] = artifact_budget_utilization(
                total_bytes_exact,
                budget_bytes,
            )
            metrics["official_submission_budget_fill_score"] = artifact_budget_fill_score(
                total_bytes_exact,
                budget_bytes,
            )
            metrics["official_submission_budget_edge_score"] = artifact_budget_edge_score(
                total_bytes_exact,
                budget_bytes,
                **edge_kwargs,
            )
        candidate.metrics.update(metrics)

    def _artifact_budget_edge_kwargs(self, spec: ArchitectureSpec) -> dict[str, float]:
        parameter_golf = spec.parameter_golf
        if parameter_golf is None:
            return {}
        return {
            "target_utilization": float(parameter_golf.target_budget_utilization),
            "under_window": float(parameter_golf.target_budget_under_window),
            "over_window": float(parameter_golf.target_budget_over_window),
        }

    def _annotate_seed_lane_metadata(self, candidate: Candidate) -> None:
        metadata = seed_lane_metadata(candidate.spec)
        if not metadata:
            return
        metadata["motif_signature"] = motif_signature(candidate.spec)
        candidate.metadata.update(metadata)

    def _motif_appearance_count(self, candidate: Candidate) -> int:
        signature = str(candidate.metadata.get("motif_signature") or "").strip()
        if not signature:
            return 1
        families = set(self._motif_families.get(signature, set()))
        family = candidate.metadata.get("seed_family")
        if family is not None:
            families.add(str(family))
        return max(1, len(families))

    def _record_motif_family(self, candidate: Candidate) -> None:
        signature = str(candidate.metadata.get("motif_signature") or "").strip()
        family = candidate.metadata.get("seed_family")
        if not signature or family is None:
            return
        known = self._motif_families.setdefault(signature, set())
        known.add(str(family))

    def _annotate_incubator_metrics(self, candidate: Candidate) -> None:
        if candidate.spec.parameter_golf is None:
            return
        appearance_count = self._motif_appearance_count(candidate)
        summary = incubator_promotion_summary(
            candidate.spec,
            post_quant_val_bpb=candidate.metrics.get("post_quant_val_bpb"),
            appearance_count=appearance_count,
        )
        delta = summary.get("delta_bpb")
        candidate.metadata["motif_promotion"] = summary
        candidate.metrics["motif_appearance_count"] = float(appearance_count)
        candidate.metrics["motif_transfer_eligible"] = 1.0 if summary["eligible"] else 0.0
        if delta is not None:
            candidate.metrics["motif_anchor_delta_bpb"] = float(delta)

    def run(self, generations: int) -> list[Candidate]:
        survivors: list[Candidate] = []
        start_generation = self._generation_idx if (self.pool or self._history) else 0
        self._set_generation(start_generation)
        if not self.pool:
            base_candidate = Candidate(
                ident=self._new_id("seed"), spec=self.base_spec.model_copy(deep=True)
            )
            self._assign_missing_origin_ids(base_candidate.spec)
            if self._init_checkpoint is not None:
                base_candidate.seed_state_path = self._init_checkpoint
            self._parents[base_candidate.ident] = []
            try:
                self._evaluate_candidate(base_candidate)
            except Exception as exc:
                console.print(f"[red]Seed candidate {base_candidate.ident} crashed:[/] {exc}")
                base_candidate.status = "failed"
                base_candidate.checkpoint = None
                try:
                    self._cleanup_seed_state(base_candidate)
                except Exception as cleanup_exc:
                    console.print(f"[red]Seed cleanup failed:[/] {cleanup_exc}")
                try:
                    self._remove_candidate_artifacts(base_candidate)
                except Exception as cleanup_exc:
                    console.print(f"[red]Seed artifact cleanup failed:[/] {cleanup_exc}")
            if base_candidate.status == "completed":
                self._update_archive(base_candidate)
                self._record_motif_family(base_candidate)
                survivors.append(base_candidate)
            self.pool.append(base_candidate)
            self._history.append(base_candidate)
            self._remember_spec(base_candidate.spec)
        for generation_offset in range(generations):
            generation_idx = int(start_generation + generation_offset)
            self._set_generation(generation_idx)
            try:
                candidate = self._spawn_candidate()
            except Exception as exc:
                console.print(f"[red]Failed to spawn candidate:[/] {exc}")
                continue
            console.print(f"[cyan]Evaluating[/] {candidate.ident}")
            try:
                self._evaluate_candidate(candidate)
            except Exception as exc:
                console.print(
                    f"[red]Candidate {candidate.ident} crashed during evaluation:[/] {exc}"
                )
                if self._is_resource_error(exc):
                    self._empty_device_cache()
                candidate.status = "failed"
                candidate.checkpoint = None
                try:
                    self._cleanup_seed_state(candidate)
                except Exception as cleanup_exc:
                    console.print(f"[red]Candidate cleanup failed:[/] {cleanup_exc}")
                try:
                    self._remove_candidate_artifacts(candidate)
                except Exception as cleanup_exc:
                    console.print(f"[red]Candidate artifact cleanup failed:[/] {cleanup_exc}")
            if candidate.status == "completed":
                self._update_archive(candidate)
                self._maybe_update_mutation_weights(candidate)
                self._maybe_update_template_learning(candidate)
                self._record_motif_family(candidate)
            if candidate.status == "completed":
                survivors.append(candidate)
            self.pool.append(candidate)
            self._history.append(candidate)
            self._remember_spec(candidate.spec)
            try:
                self._trim_pool()
            except Exception as exc:
                console.print(f"[red]Pool trim failed:[/] {exc}")
            try:
                self._garbage_collect_checkpoints()
            except Exception as exc:
                console.print(f"[red]Checkpoint GC failed:[/] {exc}")
            self._generation_idx = generation_idx + 1
        try:
            flush_template_learning()
        except Exception as exc:
            console.print(f"[red]Template learning flush failed:[/] {exc}")
        return survivors

    def _evaluate_candidate(self, candidate: Candidate) -> None:
        self._annotate_seed_lane_metadata(candidate)
        candidate.metrics["layers"] = float(candidate.spec.model.n_layers)
        candidate.metrics["effective_depth"] = float(candidate.spec.model.n_layers)
        candidate.metrics["physical_depth"] = float(candidate.spec.model.physical_block_count())
        candidate.metrics["shared_blocks"] = float(candidate.spec.model.shared_block_count())
        candidate.metrics["moe_blocks"] = float(candidate.spec.model.moe_block_count())
        candidate.metrics["params"] = float(estimate_params(candidate.spec))
        candidate.metrics["seq_len"] = float(candidate.spec.data.seq_len)
        candidate.metrics["graph_entropy"] = self._graph_entropy(candidate.spec)
        candidate.metrics["complexity_score"] = complexity_score(candidate.spec)
        if candidate.spec.parameter_golf is not None:
            edge_kwargs = self._artifact_budget_edge_kwargs(candidate.spec)
            raw_payload_bytes, raw_total_bytes = estimate_artifact_total_bytes_for_spec(
                candidate.spec
            )
            calibrated_payload_bytes, calibrated_total_bytes = (
                estimate_calibrated_artifact_total_bytes_for_spec(candidate.spec)
            )
            candidate.metrics["artifact_payload_bytes_raw_est"] = float(raw_payload_bytes)
            candidate.metrics["artifact_total_bytes_raw_est"] = float(raw_total_bytes)
            candidate.metrics["artifact_payload_bytes_est"] = float(calibrated_payload_bytes)
            candidate.metrics["artifact_total_bytes_est"] = float(calibrated_total_bytes)
            candidate.metrics["artifact_code_bytes"] = float(
                candidate.spec.parameter_golf.code_bytes
            )
            candidate.metrics["artifact_budget_bytes"] = float(
                candidate.spec.parameter_golf.artifact_budget_bytes
            )
            candidate.metrics["artifact_budget_utilization_est"] = artifact_budget_utilization(
                calibrated_total_bytes,
                candidate.spec.parameter_golf.artifact_budget_bytes,
            )
            candidate.metrics["artifact_budget_fill_score_est"] = artifact_budget_fill_score(
                calibrated_total_bytes,
                candidate.spec.parameter_golf.artifact_budget_bytes,
            )
            candidate.metrics["artifact_budget_edge_score_est"] = artifact_budget_edge_score(
                calibrated_total_bytes,
                candidate.spec.parameter_golf.artifact_budget_bytes,
                **edge_kwargs,
            )
            candidate.metrics["eval_protocol_is_faithful"] = (
                0.0
                if str(getattr(candidate.spec.parameter_golf, "eval_protocol", "mid_fidelity"))
                == "scout_fast"
                else 1.0
            )
            self._annotate_official_submission_metrics(candidate)
        # Selector-related proxies: count blocks with selector enabled and average top-k
        selector_blocks: float = 0.0
        selector_topk_sum: float = 0.0
        selector_count: float = 0.0
        # Memory proxies: count blocks with retro extras and number of recurrences
        memory_blocks: float = 0.0
        embedding_ffn_blocks: float = 0.0
        flex_ffn_blocks: float = 0.0
        ssm_blocks: float = 0.0
        mla_blocks: float = 0.0
        linear_blocks: float = 0.0
        sparsity_blocks: float = 0.0
        qk_norm_blocks: float = 0.0
        for block in candidate.spec.model.blocks:
            attn = block.attn
            if attn and getattr(attn, "selector", "none") != "none":
                selector_blocks += 1.0
                topk_val = getattr(attn, "selector_topk", None)
                if topk_val is not None:
                    selector_topk_sum += float(topk_val)
                    selector_count += 1.0
            if attn is not None:
                kind = str(getattr(attn, "kind", "MHA") or "MHA").upper()
                if kind == "MLA":
                    mla_blocks += 1.0
                elif kind == "LINEAR":
                    linear_blocks += 1.0
                sparsity = str(getattr(attn, "sparsity", "none") or "none").lower()
                if sparsity != "none" or getattr(attn, "sw", None) is not None:
                    sparsity_blocks += 1.0
                if getattr(attn, "qk_norm_max", None) is not None:
                    qk_norm_blocks += 1.0
            if block.ssm is not None:
                ssm_blocks += 1.0
            embed_ffn = False
            ffns = [block.ffn, getattr(block, "ffn_memory", None)]
            for ffn in ffns:
                if ffn is None:
                    continue
                if str(getattr(ffn, "input_source", "residual") or "residual") == "embedding":
                    embed_ffn = True
            if embed_ffn:
                embedding_ffn_blocks += 1.0
            if getattr(block, "ffn_memory", None) is not None:
                flex_ffn_blocks += 1.0
            # Count memory-bearing blocks via retro extras
            for extra in block.extras:
                extra_type = getattr(extra, "type", type(extra).__name__).lower()
                if extra_type in {"retro", "assoc_memory", "memory_tokens", "chunk_memory"}:
                    memory_blocks += 1.0
                    break
        candidate.metrics["selector_blocks"] = selector_blocks
        candidate.metrics["selector_topk_avg"] = (
            selector_topk_sum / selector_count if selector_count > 0 else 0.0
        )
        candidate.metrics["memory_blocks"] = memory_blocks
        candidate.metrics["embedding_ffn_blocks"] = embedding_ffn_blocks
        candidate.metrics["flex_ffn_blocks"] = flex_ffn_blocks
        candidate.metrics["ssm_blocks"] = ssm_blocks
        candidate.metrics["mla_blocks"] = mla_blocks
        candidate.metrics["linear_blocks"] = linear_blocks
        candidate.metrics["sparsity_blocks"] = sparsity_blocks
        candidate.metrics["qk_norm_blocks"] = qk_norm_blocks
        candidate.metrics["recurrences"] = float(len(candidate.spec.model.recurrences))
        opt_cfg = candidate.spec.train.optimizer
        opt_name = str(getattr(opt_cfg, "name", "adamw") or "adamw").lower()
        opt_family_map = {"adamw": 0.0, "lion": 1.0, "muon": 2.0}
        candidate.metrics["optimizer_family"] = float(opt_family_map.get(opt_name, -1.0))
        grad_transform = getattr(opt_cfg, "gradient_transform", None)
        if grad_transform is not None:
            grad_mode = str(getattr(grad_transform, "mode", "identity") or "identity").lower()
            grad_mode_map = {
                "identity": 0.0,
                "sign": 1.0,
                "normalize": 2.0,
                "orthogonalize_2d": 3.0,
                "sign_orthogonalize_2d": 4.0,
            }
            candidate.metrics["opt_grad_transform_mode"] = float(grad_mode_map.get(grad_mode, -1.0))
            candidate.metrics["opt_grad_transform_ns_steps"] = float(
                getattr(grad_transform, "ns_steps", 5) or 5
            )
            grad_eps = float(getattr(grad_transform, "eps", 1e-8) or 1e-8)
            candidate.metrics["opt_grad_transform_eps_log10"] = float(
                math.log10(max(1e-12, grad_eps))
            )
        update_filter = getattr(opt_cfg, "update_filter", None)
        if update_filter is not None:
            mode = str(getattr(update_filter, "mode", "none") or "none").lower()
            gran = str(getattr(update_filter, "granularity", "element") or "element").lower()
            mode_map = {"none": 0.0, "bernoulli": 1.0, "topk": 2.0}
            gran_map = {"element": 0.0, "block": 1.0}
            candidate.metrics["opt_mask_mode"] = float(mode_map.get(mode, -1.0))
            candidate.metrics["opt_mask_keep_ratio"] = float(
                getattr(update_filter, "keep_ratio", 1.0) or 1.0
            )
            candidate.metrics["opt_mask_granularity"] = float(gran_map.get(gran, -1.0))
            candidate.metrics["opt_mask_momentum_blend"] = float(
                getattr(update_filter, "momentum_blend", 0.0) or 0.0
            )
        # novelty diagnostics and archive-based novelty objective
        ref = None
        if candidate.parent:
            parent = next((c for c in self.pool if c.ident == candidate.parent), None)
            ref = parent.spec if parent else self.base_spec
        else:
            ref = self.base_spec
        parent_novelty = float(self._structural_distance(ref, candidate.spec))
        descriptor = behavioral_descriptor(candidate.spec)
        candidate.metrics["parent_novelty"] = parent_novelty
        candidate.metrics["novelty"] = archive_novelty(
            descriptor,
            self._novelty_archive,
            k=max(1, int(self._novelty_k)),
        )
        self._update_novelty_archive(descriptor)
        if not self._thresholds_ok(candidate, require_metrics=False):
            candidate.status = "failed"
            self._cleanup_seed_state(candidate)
            return
        if self.mode == "live":
            static = self.checker.run(candidate.spec)
            candidate.metrics.update(static.metrics)
            if not self._thresholds_ok(candidate, require_metrics=False):
                candidate.status = "failed"
                self._cleanup_seed_state(candidate)
                return
            if not static.ok:
                candidate.status = "failed"
                self._cleanup_seed_state(candidate)
                return
            if self.trainer is None or self.data_module is None:
                raise RuntimeError("Live mode requires trainer and data module.")
            # Prior-aware token budget
            params = float(estimate_params(candidate.spec))
            tokens_budget = int(candidate.spec.priors.tokens_per_param * params)
            base_rung2 = max(int(self.cfg.rung2_tokens), int(candidate.spec.train.max_tokens or 0))
            base_rung1 = int(self.cfg.rung1_tokens)
            mult = 1.0
            if candidate.spec.model.n_layers >= 4:
                mult += 0.2
            if candidate.spec.model.moe_block_count() >= 1:
                mult += 0.1
            rung2_tokens = int(min(base_rung2 * mult, tokens_budget)) if tokens_budget > 0 else 0
            rung1_tokens = int(min(base_rung1 * mult, rung2_tokens)) if rung2_tokens > 0 else 0
            rung2_extra = max(0, rung2_tokens - rung1_tokens)
            # Multi-fidelity schedule: rung1 (short), possibly rung2 (full)
            base_steps = getattr(self.trainer, "steps", None)
            if not isinstance(base_steps, int):
                base_steps = 100
            # Rung 1
            rung1_steps = (
                base_steps if rung2_extra <= 0 else max(1, int(base_steps * self._rung1_ratio))
            )
            self.trainer.steps = rung1_steps
            if rung1_tokens <= 0:
                candidate.status = "failed"
                self.trainer.steps = base_steps
                return
            seed_value = int(getattr(candidate.spec.train, "seed", 0) or 0)
            if hasattr(self.data_module, "reset_rng"):
                self.data_module.reset_rng(seed_value)
            batches = self.data_module.batches(max_tokens=rung1_tokens)
            seed_state = candidate.seed_state_path or candidate.parent_checkpoint
            if self.weight_inheritance == "scratch":
                seed_state = None
            elif self.weight_inheritance == "init" and self._init_checkpoint is not None:
                seed_state = self._init_checkpoint
            try:
                metrics1, checkpoint = self.trainer.train(
                    candidate=candidate,
                    spec=candidate.spec,
                    batch_iter=batches,
                    seed_state_path=seed_state,
                )
            except Exception as exc:
                console.print(f"[red]Candidate {candidate.ident} failed during rung1:[/] {exc}")
                if self._is_resource_error(exc):
                    self._empty_device_cache()
                candidate.status = "failed"
                candidate.checkpoint = None
                self._cleanup_seed_state(candidate)
                self._remove_candidate_artifacts(candidate)
                self.trainer.steps = base_steps
                return
            candidate.metrics.update(metrics1)
            self._annotate_official_submission_metrics(candidate)
            self._annotate_incubator_metrics(candidate)
            candidate.checkpoint = checkpoint
            # Composite metrics may be part of the objective set (e.g.,
            # efficiency_score). Compute them before objective gating.
            self._apply_composite_metrics(candidate)
            if not self._objective_metrics_ok(candidate):
                candidate.status = "failed"
                self._cleanup_seed_state(candidate)
                self._remove_candidate_artifacts(candidate)
                candidate.checkpoint = None
                self.trainer.steps = base_steps
                return
            # Early stop heuristic: clearly poor ppl
            ppl1 = float(candidate.metrics.get("ppl_code", 1e9))
            threshold = self.ppl_stop_threshold
            if threshold is not None and ppl1 > threshold:
                self._cleanup_seed_state(candidate)
                candidate.status = "completed"
                self._apply_composite_metrics(candidate)
                self.frontier.update(candidate)
                # restore trainer steps
                self.trainer.steps = base_steps
                return

            # Adaptive rung budget: check improvement rate
            adaptive_rung = bool(getattr(self.cfg, "adaptive_rung_budget", False))
            improvement_rate = float(candidate.metrics.get("improvement_rate", 0.0))

            if adaptive_rung:
                fast_promote = float(
                    getattr(self.cfg, "adaptive_rung_fast_promote_threshold", 0.15) or 0.15
                )
                slow_stop = float(
                    getattr(self.cfg, "adaptive_rung_slow_stop_threshold", 0.02) or 0.02
                )

                # Early stop if improvement rate is too low (plateau)
                if improvement_rate < slow_stop:
                    console.print(
                        f"[yellow]Early stop {candidate.ident}:[/] low improvement rate "
                        f"({improvement_rate:.3f} < {slow_stop})"
                    )
                    self._cleanup_seed_state(candidate)
                    candidate.status = "completed"
                    self._apply_composite_metrics(candidate)
                    self.frontier.update(candidate)
                    self.trainer.steps = base_steps
                    return

                # Fast promote: increase rung2 budget for rapidly improving candidates
                if improvement_rate > fast_promote:
                    console.print(
                        f"[green]Fast promote {candidate.ident}:[/] high improvement rate "
                        f"({improvement_rate:.3f} > {fast_promote})"
                    )
                    # Give 50% extra budget but never exceed total token budget.
                    rung2_extra = int(rung2_extra * 1.5)
                    rung2_extra = min(rung2_extra, max(0, int(tokens_budget) - int(rung1_tokens)))
            # Rung 2 (full)
            if rung2_extra <= 0:
                self._cleanup_seed_state(candidate)
                candidate.status = "completed"
                self._apply_composite_metrics(candidate)
                self.frontier.update(candidate)
                self.trainer.steps = base_steps
                return
            self.trainer.steps = max(1, int(base_steps * self._rung2_ratio))
            batches = self.data_module.batches(max_tokens=rung2_extra)
            try:
                # Disable speedrun metrics during continuation rungs so the
                # "tokens_to_target" objective reflects from-scratch learning
                # (rung1) rather than warm-started refinement.
                rung2_spec = candidate.spec.model_copy(deep=True)
                rung2_spec.train.speedrun_eval_interval = 0
                metrics2, checkpoint = self.trainer.train(
                    candidate=candidate,
                    spec=rung2_spec,
                    batch_iter=batches,
                    seed_state_path=checkpoint,
                )
            except Exception as exc:
                console.print(f"[red]Candidate {candidate.ident} failed during rung2:[/] {exc}")
                if self._is_resource_error(exc):
                    self._empty_device_cache()
                candidate.status = "failed"
                candidate.checkpoint = None
                self._cleanup_seed_state(candidate)
                self._remove_candidate_artifacts(candidate)
                # restore trainer steps
                self.trainer.steps = base_steps
                return
            candidate.metrics.update(metrics2)
            self._annotate_official_submission_metrics(candidate)
            self._annotate_incubator_metrics(candidate)
            candidate.checkpoint = checkpoint
            self._apply_composite_metrics(candidate)
            if not self._objective_metrics_ok(candidate):
                candidate.status = "failed"
                self._cleanup_seed_state(candidate)
                self._remove_candidate_artifacts(candidate)
                candidate.checkpoint = None
                # restore trainer steps
                self.trainer.steps = base_steps
                return
            if candidate.seed_state_path is not None:
                self._cleanup_seed_state(candidate)
            candidate.status = "completed"
            rung_tokens_used = int(rung1_tokens) + int(rung2_extra)
            # Optional promotion rung: give complex candidates extra budget
            promoted = False
            if self._should_promote(candidate, tokens_budget, rung_tokens_used):
                promoted = self._run_promotion(
                    candidate, base_steps, tokens_budget, rung_tokens_used
                )
            if promoted:
                self._apply_composite_metrics(candidate)
                if not self._objective_metrics_ok(candidate):
                    candidate.status = "failed"
                    self._cleanup_seed_state(candidate)
                    self._remove_candidate_artifacts(candidate)
                    candidate.checkpoint = None
                    self.trainer.steps = base_steps
                    return
            # Prior distance metric (not in objectives by default)
            candidate.metrics["prior_distance"] = self._prior_distance(candidate.spec)
            self._apply_composite_metrics(candidate)
            self.frontier.update(candidate)
            # Restore trainer step budget
            self.trainer.steps = base_steps
            return
        if not self.evaluator.rung0(candidate):  # type: ignore[union-attr]
            return
        candidate.rung = 1
        if not self.evaluator.rung1(candidate):  # type: ignore[union-attr]
            return
        candidate.rung = 2
        if not self.evaluator.rung2(candidate):  # type: ignore[union-attr]
            return
        self._apply_composite_metrics(candidate)
        self.frontier.update(candidate)

    def _should_promote(self, candidate: Candidate, tokens_budget: int, used_tokens: int) -> bool:
        """Decide whether to apply a high-budget promotion rung to this candidate."""
        if self._promotion_prob <= 0.0:
            return False
        if self.mode != "live" or self.trainer is None or self.data_module is None:
            return False
        if tokens_budget <= 0 or used_tokens >= tokens_budget:
            return False
        layers = candidate.spec.model.n_layers
        moe_blocks = candidate.spec.model.moe_block_count()
        if layers < self._promotion_min_layers:
            return False
        if moe_blocks < self._promotion_min_moe_blocks:
            return False
        router_entropy = candidate.metrics.get("router_entropy")
        if router_entropy is not None and router_entropy < self._promotion_min_router_entropy:
            return False
        recurrence_gain = candidate.metrics.get("recurrence_gain")
        if recurrence_gain is not None and recurrence_gain < self._promotion_min_recurrence_gain:
            return False
        if self._promotion_max_instability is not None:
            instability = candidate.metrics.get("instability")
            if instability is not None and instability > self._promotion_max_instability:
                return False
        if self.rng.random() > self._promotion_prob:
            return False
        return True

    def _run_promotion(
        self,
        candidate: Candidate,
        base_steps: int,
        tokens_budget: int,
        used_tokens: int,
    ) -> bool:
        """Apply an additional high-budget training rung to the candidate."""
        if self.trainer is None or self.data_module is None:
            return False
        extra_tokens = int(self._promotion_tokens_multiplier * max(tokens_budget - used_tokens, 0))
        if extra_tokens <= 0:
            return False
        max_tokens = min(tokens_budget, used_tokens + extra_tokens)
        promo_tokens = max(0, int(max_tokens - used_tokens))
        if promo_tokens <= 0:
            return False
        promo_steps = max(1, int(base_steps * self._promotion_steps_multiplier))
        original_steps = self.trainer.steps
        try:
            self.trainer.steps = promo_steps
            batches = self.data_module.batches(max_tokens=promo_tokens)
            seed_state = candidate.checkpoint
            try:
                promo_spec = candidate.spec.model_copy(deep=True)
                promo_spec.train.speedrun_eval_interval = 0
                metrics3, checkpoint3 = self.trainer.train(
                    candidate=candidate,
                    spec=promo_spec,
                    batch_iter=batches,
                    seed_state_path=seed_state,
                )
                candidate.metrics.update(metrics3)
                candidate.checkpoint = checkpoint3
                return True
            except Exception as exc:
                console.print(f"[yellow]Promotion rung failed for {candidate.ident}:[/] {exc}")
                if self._is_resource_error(exc):
                    self._empty_device_cache()
                return False
        finally:
            self.trainer.steps = original_steps

    @staticmethod
    def _structural_distance(a: ArchitectureSpec, b: ArchitectureSpec) -> float:
        return structural_distance(a, b)

    def _prior_distance(self, spec: ArchitectureSpec) -> float:
        return prior_distance(spec)

    @staticmethod
    def _graph_entropy(spec: ArchitectureSpec) -> float:
        return graph_entropy(spec)

    def _apply_composite_metrics(self, candidate: Candidate) -> None:
        if not self._composite_metrics:
            return
        for comp in self._composite_metrics:
            value = self._compute_composite(comp, candidate.metrics)
            if value is None:
                continue
            candidate.metrics[comp.name] = value

    @staticmethod
    def _compute_composite(comp: CompositeMetricConfig, metrics: dict[str, float]) -> float | None:
        return compute_composite(comp, metrics)

    @staticmethod
    def _default_composites() -> list[CompositeMetricConfig]:
        return default_composites()

    @staticmethod
    def _merge_composites(
        primary: list[CompositeMetricConfig], defaults: list[CompositeMetricConfig]
    ) -> list[CompositeMetricConfig]:
        return merge_composites(primary, defaults)

    def _select_parent(self) -> Candidate:
        strategy = getattr(self.cfg, "parent_selection", "weighted")
        if strategy == "map_elites" and self.archive:
            return self.rng.choice(list(self.archive.values()))
        if strategy == "pareto_uniform" and self.frontier.entries:
            return self.rng.choice(self.frontier.entries)
        # Optionally restrict the parent pool to a top-k fraction to tune exploration/exploitation.
        candidates = list(self.pool)
        topk_keep = float(getattr(self.cfg, "topk_keep", 1.0) or 1.0)
        if candidates and 0.0 < topk_keep < 1.0:
            # Always keep the structural elites eligible as parents.
            elite_ids: set[str] = set()
            if self.structural_elite_k > 0:
                scored = [(self._structural_score(c), c) for c in candidates]
                scored.sort(key=lambda t: t[0], reverse=True)
                elite_ids = {c.ident for _, c in scored[: self.structural_elite_k]}
            scored_by_score = sorted(
                candidates,
                key=lambda cand: cand.score(self.score_weights),
                reverse=True,
            )
            keep_n = max(1, int(round(topk_keep * len(scored_by_score))))
            keep_ids = {c.ident for c in scored_by_score[:keep_n]}
            keep_ids.update(elite_ids)
            candidates = [c for c in candidates if c.ident in keep_ids]

        if strategy == "epsilon_lexicase" and candidates:
            # Epsilon-lexicase: like lexicase but with tolerance bands for near-ties.
            # Candidates within epsilon of the best are all considered "tied".
            return self._select_epsilon_lexicase(candidates)

        if strategy == "lexicase" and candidates:
            # Lexicase: randomly order objectives; progressively filter to the best on each
            objectives = list(self.objective_dir.keys())
            self.rng.shuffle(objectives)
            for obj in objectives:
                if not candidates:
                    break
                direction = self.objective_dir[obj]
                best_val = None
                for cand in candidates:
                    val = cand.metrics.get(obj, 0.0)
                    if best_val is None:
                        best_val = val
                    else:
                        if (direction == "max" and val > best_val) or (
                            direction == "min" and val < best_val
                        ):
                            best_val = val
                # keep top 10% (or all equal) on this objective
                if best_val is not None:
                    tol = 1e-9
                    filtered = []
                    # compute threshold for top-10%
                    values = [c.metrics.get(obj, 0.0) for c in candidates]
                    if direction == "max":
                        values_sorted = sorted(values, reverse=True)
                    else:
                        values_sorted = sorted(values)
                    cutoff_idx = max(0, int(0.1 * (len(values_sorted) - 1)))
                    threshold = values_sorted[cutoff_idx]
                    for cand in candidates:
                        val = cand.metrics.get(obj, 0.0)
                        if direction == "max":
                            if val + tol >= threshold:
                                filtered.append(cand)
                        else:
                            if val - tol <= threshold:
                                filtered.append(cand)
                    candidates = filtered
            if candidates:
                return self.rng.choice(candidates)
        # Default weighted tournament among 3
        contenders = self.rng.sample(
            candidates or self.pool, k=min(3, len(candidates or self.pool))
        )
        return max(contenders, key=lambda cand: cand.score(self.score_weights))

    def _select_epsilon_lexicase(self, candidates: list[Candidate]) -> Candidate:
        """Epsilon-lexicase selection with tolerance bands.

        Unlike strict lexicase, candidates within epsilon of the best value
        on each objective are all considered "tied" and pass to the next round.
        This prevents good candidates from being eliminated due to noise.

        The epsilon is computed as a fraction of the objective's range in the
        current candidate pool.
        """
        epsilon_frac = float(getattr(self.cfg, "epsilon_lexicase_epsilon", 0.05) or 0.05)
        objectives = list(self.objective_dir.keys())
        self.rng.shuffle(objectives)
        remaining = list(candidates)

        for obj in objectives:
            if len(remaining) <= 1:
                break

            direction = self.objective_dir[obj]
            values = [c.metrics.get(obj, 0.0) for c in remaining]

            # Compute range for epsilon calculation
            min_val = min(values)
            max_val = max(values)
            obj_range = max_val - min_val
            epsilon = epsilon_frac * obj_range if obj_range > 0 else 1e-9

            # Find the best value
            if direction == "max":
                best_val = max(values)
                threshold = best_val - epsilon
                filtered = [c for c in remaining if c.metrics.get(obj, 0.0) >= threshold]
            else:
                best_val = min(values)
                threshold = best_val + epsilon
                filtered = [c for c in remaining if c.metrics.get(obj, 0.0) <= threshold]

            if filtered:
                remaining = filtered

        return self.rng.choice(remaining) if remaining else self.rng.choice(candidates)

    def _update_archive(self, candidate: Candidate) -> None:
        if self.archive_max_elites <= 0:
            return
        layers = int(candidate.metrics.get("layers") or candidate.spec.model.n_layers)
        moe_blocks = int(
            candidate.metrics.get("moe_blocks") or candidate.spec.model.moe_block_count()
        )
        selector_blocks = int(candidate.metrics.get("selector_blocks") or 0)
        memory_blocks = int(candidate.metrics.get("memory_blocks") or 0)
        embedding_ffn_blocks = int(candidate.metrics.get("embedding_ffn_blocks") or 0)
        recurrences = int(candidate.metrics.get("recurrences") or 0)
        mla_blocks = int(candidate.metrics.get("mla_blocks") or 0)
        ssm_blocks = int(candidate.metrics.get("ssm_blocks") or 0)
        sparsity_blocks = int(candidate.metrics.get("sparsity_blocks") or 0)
        qk_norm_blocks = int(candidate.metrics.get("qk_norm_blocks") or 0)
        linear_blocks = int(candidate.metrics.get("linear_blocks") or 0)

        key = (
            f"L{min(layers, 64)}"
            f"_E{min(moe_blocks, 16)}"
            f"_S{min(selector_blocks, 16)}"
            f"_M{min(memory_blocks, 16)}"
            f"_F{min(embedding_ffn_blocks, 16)}"
            f"_R{min(recurrences, 8)}"
            f"_A{min(mla_blocks, 16)}"
            f"_X{min(ssm_blocks, 16)}"
            f"_P{min(sparsity_blocks, 16)}"
            f"_Q{min(qk_norm_blocks, 16)}"
            f"_N{min(linear_blocks, 16)}"
        )
        if bool(getattr(self.cfg, "map_elites_complexity_band", False)):
            width = float(getattr(self.cfg, "complexity_band_width", 4.0) or 4.0)
            width = max(1.0, width)
            extras_total = sum(len(block.extras) for block in candidate.spec.model.blocks)
            complexity = (
                float(layers)
                + 1.5 * float(moe_blocks)
                + 1.0 * float(memory_blocks)
                + 0.75 * float(recurrences)
                + 0.5 * float(extras_total)
            )
            band = int(max(0, math.floor(complexity / width)))
            key = f"{key}_C{band}"
        existing = self.archive.get(key)
        if existing is None or candidate.score(self.score_weights) > existing.score(
            self.score_weights
        ):
            self.archive[key] = candidate
        if len(self.archive) > self.archive_max_elites:
            worst_key = min(
                self.archive,
                key=lambda k: self.archive[k].score(self.score_weights),
            )
            self.archive.pop(worst_key, None)

    def _maybe_update_mutation_weights(self, candidate: Candidate) -> None:
        """Update mutation weights based on improvement magnitude, not just success/failure.

        Instead of binary success (did it reach frontier?), we track the actual
        improvement delta and weight mutations by how much they improve objectives.
        This allows mutations that produce large improvements to be favored over
        those that produce small ones.
        """
        if not self._adaptive_mutation or not candidate.parent:
            return
        parent = next((c for c in self.pool if c.ident == candidate.parent), None)
        if parent is None:
            parent = next((c for c in self._history if c.ident == candidate.parent), None)
        if parent is None:
            return

        # Calculate improvement delta
        delta = float(candidate.score(self.score_weights) - parent.score(self.score_weights))

        # Track running average of improvement magnitudes for normalization
        if not hasattr(self, "_improvement_baseline"):
            self._improvement_baseline = 0.1  # Initial baseline
        if not hasattr(self, "_improvement_count"):
            self._improvement_count = 0

        # Update baseline with exponential moving average of absolute deltas
        self._improvement_count += 1
        baseline_eta = 0.05  # Slow adaptation for baseline
        self._improvement_baseline = (
            1.0 - baseline_eta
        ) * self._improvement_baseline + baseline_eta * abs(delta)
        baseline = max(self._improvement_baseline, 1e-6)

        # Compute magnitude-weighted reward:
        # - Positive delta: reward = 0.5 + 0.5 * tanh(delta / baseline)
        # - Negative delta: reward = 0.5 + 0.5 * tanh(delta / baseline)
        # This gives a smooth reward in [0, 1] that scales with improvement magnitude
        import math

        normalized_delta = delta / baseline
        reward = 0.5 + 0.5 * math.tanh(normalized_delta)

        registry_names = set(mutation_names())
        names = [name for name in candidate.mutation_trace if name in registry_names]
        if not names:
            # Backward compatibility with older saved states.
            label = candidate.ident.rsplit("-", 2)[0]
            names = [name for name in label.split("+") if name in registry_names]
        if not names:
            return
        if self.mutation_weights is None:
            self.mutation_weights = dict.fromkeys(registry_names, 1.0)

        # Track improvement deltas per mutation for analysis
        if not hasattr(self, "_mutation_deltas"):
            self._mutation_deltas: dict[str, list[float]] = {}

        eta = max(1e-6, min(1.0, self._adaptive_mutation_eta))
        for name in names:
            self._mutation_counts[name] = int(self._mutation_counts.get(name, 0)) + 1

            # Track raw deltas for analysis
            if name not in self._mutation_deltas:
                self._mutation_deltas[name] = []
            self._mutation_deltas[name].append(delta)
            # Keep only last 100 deltas per mutation
            if len(self._mutation_deltas[name]) > 100:
                self._mutation_deltas[name] = self._mutation_deltas[name][-100:]

            prev = float(self._mutation_success.get(name, 0.5))
            updated = (1.0 - eta) * prev + eta * reward
            self._mutation_success[name] = updated
            lo = max(1e-6, self._adaptive_mutation_min)
            hi = max(lo, self._adaptive_mutation_max)
            self.mutation_weights[name] = lo + (hi - lo) * updated

    def _maybe_update_template_learning(self, candidate: Candidate) -> None:
        if not self._template_learning or not candidate.parent:
            return
        parent = next((c for c in self.pool if c.ident == candidate.parent), None)
        if parent is None:
            parent = next((c for c in self._history if c.ident == candidate.parent), None)
        if parent is None:
            return
        delta = float(candidate.score(self.score_weights) - parent.score(self.score_weights))
        segments = list(candidate.mutation_trace)
        if not segments:
            label = candidate.ident.rsplit("-", 2)[0]
            segments = label.split("+")
        for segment in segments:
            template_name: str | None = None
            if segment.startswith("tpl::"):
                template_name = segment.split("::", 1)[1]
            elif segment.startswith("tpl_"):
                template_name = segment[4:]
            if template_name and template_name != "none":
                record_template_result(template_name, delta)
        if bool(getattr(self.cfg, "register_template_entries", True)):
            register_template_mutations()

    def _trim_pool(self) -> None:
        if len(self.pool) <= self.cfg.population:
            return
        frontier_ids = {cand.ident for cand in self.frontier.entries}
        archive_ids = {cand.ident for cand in self.archive.values()}
        # Protect structural elites from trimming (keep them in the parent pool).
        elite_ids: set[str] = set()
        if self.structural_elite_k > 0 and self.pool:
            scored = [(self._structural_score(c), c) for c in self.pool]
            scored.sort(key=lambda t: t[0], reverse=True)
            elite_ids = {c.ident for _, c in scored[: self.structural_elite_k]}
        excess = len(self.pool) - self.cfg.population
        removable = [cand for cand in self.pool if cand.ident not in elite_ids]
        removable.sort(key=lambda cand: cand.score(self.score_weights))
        to_remove = {cand.ident for cand in removable[: max(0, excess)]}
        removed = [cand for cand in self.pool if cand.ident in to_remove]
        self.pool = [cand for cand in self.pool if cand.ident not in to_remove]
        for candidate in removed:
            # Keep checkpoint artifacts for Pareto-frontier entries even if removed from pool.
            if candidate.ident in frontier_ids or candidate.ident in archive_ids:
                continue
            self._remove_candidate_artifacts(candidate)

    def _garbage_collect_checkpoints(self) -> None:
        """Remove orphaned checkpoint files to keep disk usage bounded.

        During the run we may temporarily keep checkpoints for candidates that
        were on the frontier when they were trimmed from the pool. If they later
        fall off the frontier and are not retained as archive elites, their
        checkpoints become unreachable. This GC keeps only checkpoints needed to
        resume: pool + frontier + archive (+ any transient seed_state_path).
        """

        if not self.checkpoint_dir.exists():
            return

        keep: set[Path] = set()

        def _add(path: Path | None) -> None:
            if path is None:
                return
            try:
                keep.add(Path(path).resolve())
            except Exception:
                return

        _add(self._init_checkpoint)
        for cand in self.pool:
            _add(cand.checkpoint)
            _add(cand.seed_state_path)
        for cand in self.frontier.entries:
            _add(cand.checkpoint)
        for cand in self.archive.values():
            _add(cand.checkpoint)
            _add(cand.seed_state_path)

        for file_path in self.checkpoint_dir.glob("*.pt"):
            try:
                if file_path.resolve() in keep:
                    continue
                file_path.unlink(missing_ok=True)
            except OSError:
                continue

    def _spawn_candidate(self) -> Candidate:
        if (self.pool or self._history) and not self._seen_spec_fingerprints:
            # Helpful for tests / manual runner construction.
            self._rebuild_seen_specs()
        seen = self._seen_spec_fingerprints
        max_attempts = 32

        if (
            self.mode == "live"
            and len(self.pool) >= 2
            and self.rng.random() < self.cfg.crossover_prob
        ):
            parent_pool: list[Candidate] = []
            strategy = getattr(self.cfg, "parent_selection", "weighted")
            if strategy == "map_elites" and len(self.archive) >= 2:
                parent_pool = list(self.archive.values())
            else:
                parent_pool = [c for c in self.pool if c.status == "completed"]
                topk_keep = float(getattr(self.cfg, "topk_keep", 1.0) or 1.0)
                if parent_pool and 0.0 < topk_keep < 1.0:
                    scored = sorted(
                        parent_pool,
                        key=lambda cand: cand.score(self.score_weights),
                        reverse=True,
                    )
                    keep_n = max(2, int(round(topk_keep * len(scored))))
                    parent_pool = scored[:keep_n]
            if len(parent_pool) < 2:
                parent_pool = []
            for _ in range(max_attempts):
                if not parent_pool:
                    break
                parent_a, parent_b = self.rng.sample(parent_pool, 2)
                spec: ArchitectureSpec | None = None
                crossover_report: dict[str, Any] = {}
                prefer_parent: ParentKey = (
                    "a"
                    if parent_a.score(self.score_weights) >= parent_b.score(self.score_weights)
                    else "b"
                )
                try:
                    plan = aligned_splice_blocks(
                        parent_a.spec,
                        parent_b.spec,
                        self.rng,
                        preferred_parent=prefer_parent,
                    )
                    base_spec = parent_a.spec if prefer_parent == "a" else parent_b.spec
                    spec_data = base_spec.model_dump(mode="python")
                    spec_data["model"]["blocks"] = [
                        block.model_dump(mode="python") for block in plan.blocks
                    ]
                    spec = ArchitectureSpec(**spec_data)
                    self._assign_missing_origin_ids(spec)
                    crossover_report = dict(plan.report)
                except Exception:
                    spec = None
                if spec is None:
                    continue
                if self._spec_fingerprint(spec) in seen:
                    continue
                child_id = self._new_id("xover")
                seed_path: Path | None = None
                if self.weight_inheritance == "parent":
                    seed_path = self.checkpoint_dir / f"{child_id}_seed.pt"
                    merged_path, merge_report = merge_checkpoints_with_report(
                        child_spec=spec,
                        parent_a_ckpt=parent_a.checkpoint,
                        parent_b_ckpt=parent_b.checkpoint,
                        out_path=seed_path,
                        source_map=plan.source_map,
                        preferred_parent=prefer_parent,
                    )
                    seed_path = merged_path
                    crossover_report["checkpoint_merge"] = merge_report
                self._parents[child_id] = [parent_a.ident, parent_b.ident]
                return Candidate(
                    ident=child_id,
                    spec=spec,
                    parent=None,
                    seed_state_path=seed_path,
                    mutation_trace=["xover::aligned"],
                    crossover_report=crossover_report,
                )
            console.print(
                "[yellow]Warning:[/] crossover only produced already-seen specs; "
                "falling back to mutation."
            )
        parent = self._select_parent()
        # Get context-aware mutation weights if enabled
        mutation_weights = self._context_aware_mutation_weights(parent)
        if mutation_weights is None:
            mutation_weights = self.mutation_weights
        parent_fp = self._spec_fingerprint(parent.spec)
        last: tuple[str, ArchitectureSpec, list[str]] | None = None
        for _ in range(max_attempts):
            result: tuple[str, ArchitectureSpec, list[str]] | None = None
            try:
                mutate_kwargs: dict[str, Any] = {"steps": getattr(self, "mutation_steps", 1)}
                if self._mutation_allowlist is not None:
                    mutate_kwargs["allowed_names"] = self._mutation_allowlist
                result = mutate_with_trace(
                    parent.spec,
                    self.rng,
                    mutation_weights,
                    **mutate_kwargs,
                )
            except Exception:
                result = None
            if result is None:
                continue
            name, spec, trace = result
            self._assign_missing_origin_ids(spec)
            last = (name, spec, trace)
            spec_fp = self._spec_fingerprint(spec)
            if spec_fp == parent_fp or spec_fp in seen:
                continue
            child = Candidate(
                ident=self._new_id(name),
                spec=spec,
                parent=parent.ident,
                parent_checkpoint=parent.checkpoint,
                mutation_trace=trace,
            )
            self._parents[child.ident] = [parent.ident]
            return child
        if last is None:
            msg = "mutation produced no candidates"
            raise RuntimeError(msg)
        name, spec, trace = last
        console.print(
            "[yellow]Warning:[/] mutation only produced already-seen/no-op specs; "
            f"using last sample ({name})."
        )
        child = Candidate(
            ident=self._new_id(name),
            spec=spec,
            parent=parent.ident,
            parent_checkpoint=parent.checkpoint,
            mutation_trace=trace,
        )
        self._parents[child.ident] = [parent.ident]
        return child

    def _context_aware_mutation_weights(self, parent: Candidate) -> dict[str, float] | None:
        """Compute context-aware mutation weights based on candidate architecture state.

        Analyzes the parent's structure and metrics to prioritize mutations that
        address weaknesses or explore underutilized components.

        Returns:
            Mutation weight overrides, or None to use default weights.
        """
        if not bool(getattr(self.cfg, "context_aware_mutation", False)):
            return None

        spec = parent.spec
        metrics = parent.metrics

        # Start with uniform weights
        weights = dict.fromkeys(self._mutation_allowlist or mutation_names(), 1.0)

        # Analyze architecture state
        n_layers = spec.model.n_layers
        moe_blocks = int(metrics.get("moe_blocks", 0))
        memory_blocks = int(metrics.get("memory_blocks", 0))
        selector_blocks = int(metrics.get("selector_blocks", 0))
        router_entropy = float(metrics.get("router_entropy", 1.0))
        ssm_blocks = int(metrics.get("ssm_blocks", 0))

        # Context 1: Shallow stack -> prioritize depth mutations
        if n_layers < 6:
            if "duplicate_block_span" in weights:
                weights["duplicate_block_span"] = 3.0
            if "add_recurrence" in weights:
                weights["add_recurrence"] = 2.0
            if "add_additional_recurrence" in weights:
                weights["add_additional_recurrence"] = 2.0

        # Context 2: Many MoE blocks but low router entropy -> prioritize router tuning
        if moe_blocks >= 2 and router_entropy < 0.8:
            for key, value in (
                ("tune_router", 3.0),
                ("tune_router_coeffs", 3.0),
                ("tune_experts", 2.0),
            ):
                if key in weights:
                    weights[key] = value

        # Context 3: No memory modules -> prioritize memory insertion
        if memory_blocks == 0:
            for key, value in (
                ("insert_retro_module", 3.0),
                ("insert_assoc_memory", 2.5),
                ("insert_memory_tokens", 2.0),
                ("insert_chunk_memory", 2.0),
                ("insert_lookup_memory", 2.0),
            ):
                if key in weights:
                    weights[key] = value

        # Context 4: No selectors -> prioritize selector mutations
        if selector_blocks == 0 and n_layers >= 4:
            if "toggle_selector" in weights:
                weights["toggle_selector"] = 2.5

        # Context 5: No SSM blocks -> consider adding
        if ssm_blocks == 0 and n_layers >= 4:
            if "toggle_ssm" in weights:
                weights["toggle_ssm"] = 1.5

        # Context 6: High instability -> prioritize stability mutations
        instability = float(metrics.get("instability", 0.0))
        if instability > 2.0:
            for key, value in (
                ("toggle_qk_norm", 2.5),
                ("insert_layer_scale", 2.0),
                ("tune_layer_scale", 2.0),
            ):
                if key in weights:
                    weights[key] = value

        # Context 7: Poor long-recall -> prioritize memory/recurrence
        long_recall = float(metrics.get("long_recall", 0.5))
        if long_recall < 0.5:
            for key, value in (
                ("insert_retro_module", 2.0),
                ("add_recurrence", 2.0),
                ("insert_chunk_memory", 2.0),
            ):
                if key in weights:
                    weights[key] = value

        # Merge with existing adaptive weights if present
        if self.mutation_weights:
            for name, weight in self.mutation_weights.items():
                if name in weights:
                    # Average context-aware and adaptive weights
                    weights[name] = (weights[name] + weight) / 2

        return weights

    def _structural_score(self, cand: Candidate) -> float:
        """Score structural richness to keep depth/MoE/selector candidates alive."""
        metrics = cand.metrics
        layers = float(metrics.get("layers") or cand.spec.model.n_layers)
        moe_blocks = float(metrics.get("moe_blocks") or 0.0)
        selector_blocks = float(metrics.get("selector_blocks") or 0.0)
        w = self.structural_elite_weights
        return (
            w.get("layers", 0.0) * layers
            + w.get("moe_blocks", 0.0) * moe_blocks
            + w.get("selector_blocks", 0.0) * selector_blocks
        )

    def frontier_table(self) -> Table:
        table = Table(title="Pareto Frontier")
        table.add_column("ID")
        table.add_column("ppl_code")
        table.add_column("long_recall")
        table.add_column("throughput")
        for cand in self.frontier.entries:
            table.add_row(
                cand.ident,
                f"{cand.metrics.get('ppl_code', 0.0):.2f}",
                f"{cand.metrics.get('long_recall', 0.0):.2f}",
                f"{cand.metrics.get('throughput', 0.0):.2f}",
            )
        return table

    def save_frontier(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.frontier.to_json(), indent=2))
        console.print(f"Frontier saved to {path}")

    def save_state(self, path: Path) -> None:
        """Persist runner state for resuming later."""
        path.parent.mkdir(parents=True, exist_ok=True)
        seed_val: int | None = None
        try:
            # random.Random does not expose the seed value directly.
            seed_val = None
        except Exception:
            seed_val = None
        state: dict[str, Any] = {
            "seed": seed_val,
            "rng_state": self.rng.getstate(),
            "counter": self.counter,
            "checkpoint_dir": str(self.checkpoint_dir),
            "init_checkpoint": str(self._init_checkpoint) if self._init_checkpoint else None,
            "objective_dir": self.objective_dir,
            "score_weights": self.score_weights,
            "score_weight_overrides": self._score_weight_overrides(),
            "pool": [c.serialize() for c in self.pool],
            "frontier": [c.ident for c in self.frontier.entries],
            "parents": self._parents,
            "history": [c.serialize() for c in self._history],
            "mutation_weights": self.mutation_weights,
            "mutation_allowlist": self._mutation_allowlist,
            "mutation_steps": self.mutation_steps,
            "archive_max_elites": self.archive_max_elites,
            "archive": {k: v.ident for k, v in self.archive.items()},
            "mutation_success": self._mutation_success,
            "mutation_counts": self._mutation_counts,
            "novelty_archive": self._novelty_archive,
            "structural_elite": {
                "k": self.structural_elite_k,
                "weights": self.structural_elite_weights,
            },
            "generation_idx": self._generation_idx,
            "active_rung0_thresholds": self._active_rung0_thresholds,
            "origin_counter": self._origin_counter,
        }
        path.write_text(json.dumps(state, indent=2))
        console.print(f"State saved to {path}")

    @classmethod
    def load_state(
        cls,
        path: Path,
        mode: str = "simulate",
        score_weight_overrides: dict[str, float] | None = None,
    ) -> EvolutionRunner:
        """Rehydrate a runner from a saved state manifest."""
        data = json.loads(path.read_text())
        checkpoint_dir = Path(data.get("checkpoint_dir", "runs/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Rebuild candidates
        pool: list[Candidate] = []
        history: list[Candidate] = []
        for item in data.get("pool", []):
            pool.append(Candidate.from_json(item))
        for item in data.get("history", []):
            history.append(Candidate.from_json(item))
        # Base spec: use the first candidate's spec as base
        if not history and not pool:
            raise ValueError("State file contains no candidates.")
        base_spec = (history or pool)[0].spec.model_copy(deep=True)
        evo_cfg = base_spec.evolution
        seed_val = data.get("seed")
        if not isinstance(seed_val, int):
            seed_val = 0
        resolved_overrides = score_weight_overrides
        if resolved_overrides is None:
            saved_overrides = data.get("score_weight_overrides")
            if isinstance(saved_overrides, dict):
                resolved_overrides = {
                    str(k): float(v)
                    for k, v in saved_overrides.items()
                    if isinstance(v, int | float)
                }
            else:
                saved_signed = data.get("score_weights")
                saved_objective_dir = data.get("objective_dir")
                if isinstance(saved_signed, dict) and isinstance(saved_objective_dir, dict):
                    decoded: dict[str, float] = {}
                    for metric, direction in saved_objective_dir.items():
                        value = saved_signed.get(metric)
                        if not isinstance(value, int | float):
                            continue
                        sign = 1.0 if direction == "max" else -1.0
                        decoded[str(metric)] = float(value) / sign
                    resolved_overrides = decoded
        runner = cls(
            base_spec=base_spec,
            evolution_cfg=evo_cfg,
            mode=mode,
            objective_dir=data.get("objective_dir"),
            seed=seed_val,
            score_weight_overrides=resolved_overrides,
        )
        mutation_allowlist = data.get("mutation_allowlist")
        if isinstance(mutation_allowlist, list):
            names = set(mutation_names())
            resolved = [str(name) for name in mutation_allowlist if str(name) in names]
            runner._mutation_allowlist = resolved or None
        mutation_weights = data.get("mutation_weights")
        if isinstance(mutation_weights, dict):
            parsed_weights = {
                str(key): float(value)
                for key, value in mutation_weights.items()
                if isinstance(value, int | float)
            }
            if runner._mutation_allowlist is not None:
                allow = set(runner._mutation_allowlist)
                parsed_weights = {
                    key: value for key, value in parsed_weights.items() if key in allow
                }
            runner.mutation_weights = parsed_weights
        if "mutation_steps" in data:
            try:
                runner.mutation_steps = int(data["mutation_steps"])
            except (TypeError, ValueError):
                console.print("[yellow]Warning:[/] invalid mutation_steps in state; using default.")
        elite_cfg = data.get("structural_elite") or {}
        k_raw = elite_cfg.get("k")
        if k_raw is not None:
            try:
                runner.structural_elite_k = int(k_raw)
            except (TypeError, ValueError):
                console.print("[yellow]Warning:[/] invalid structural_elite.k; keeping default.")
        weights = elite_cfg.get("weights")
        if isinstance(weights, dict):
            for key, value in weights.items():
                if key not in runner.structural_elite_weights:
                    continue
                try:
                    runner.structural_elite_weights[key] = float(value)
                except (TypeError, ValueError):
                    console.print(
                        f"[yellow]Warning:[/] invalid structural_elite weight for {key}; "
                        "keeping default."
                    )
        runner.checkpoint_dir = checkpoint_dir
        init_ckpt = data.get("init_checkpoint")
        runner._init_checkpoint = Path(init_ckpt) if init_ckpt else None
        runner.counter = int(data.get("counter", 0))
        runner.pool = pool
        runner._history = history
        runner._parents = data.get("parents", {})
        # Rebuild frontier entries by id lookup
        id_to_candidate = {c.ident: c for c in pool + history}
        frontier_ids = data.get("frontier", [])
        runner.frontier._entries = [
            id_to_candidate[cid] for cid in frontier_ids if cid in id_to_candidate
        ]
        archive_cfg = data.get("archive")
        if isinstance(archive_cfg, dict):
            runner.archive = {
                str(bin_key): id_to_candidate[cid]
                for bin_key, cid in archive_cfg.items()
                if cid in id_to_candidate
            }
        max_elites = data.get("archive_max_elites")
        if max_elites is not None:
            try:
                runner.archive_max_elites = max(0, int(max_elites))
            except (TypeError, ValueError):
                pass
        mutation_success = data.get("mutation_success")
        if isinstance(mutation_success, dict):
            runner._mutation_success = {
                str(k): float(v) for k, v in mutation_success.items() if isinstance(v, int | float)
            }
        mutation_counts = data.get("mutation_counts")
        if isinstance(mutation_counts, dict):
            runner._mutation_counts = {
                str(k): int(v) for k, v in mutation_counts.items() if isinstance(v, int | float)
            }
        novelty_archive = data.get("novelty_archive")
        if isinstance(novelty_archive, list):
            parsed_archive: list[list[float]] = []
            for descriptor in novelty_archive:
                if not isinstance(descriptor, list):
                    continue
                parsed_archive.append(
                    [float(value) for value in descriptor if isinstance(value, int | float)]
                )
            runner._novelty_archive = parsed_archive
        generation_idx = data.get("generation_idx")
        if isinstance(generation_idx, int | float):
            runner._generation_idx = int(generation_idx)
        active_thresholds = data.get("active_rung0_thresholds")
        if isinstance(active_thresholds, dict):
            runner._active_rung0_thresholds = {
                str(key): float(value)
                for key, value in active_thresholds.items()
                if isinstance(value, int | float)
            }
        origin_counter = data.get("origin_counter")
        if isinstance(origin_counter, int | float):
            runner._origin_counter = int(origin_counter)
        for cand in runner.pool + runner._history:
            runner._assign_missing_origin_ids(cand.spec)
        runner._set_generation(runner._generation_idx)
        runner._rebuild_seen_specs()
        # Restore RNG
        rng_state = data.get("rng_state")
        if rng_state:
            # Best-effort restore; ignore if incompatible
            if isinstance(rng_state, list | tuple):
                try:
                    runner.rng.setstate(tuple(rng_state))
                except Exception:
                    console.print(
                        "[yellow]Warning:[/] failed to restore RNG state; continuing with new seed."
                    )
        return runner

    def _new_id(self, prefix: str) -> str:
        self.counter += 1
        safe = "".join(ch if (ch.isalnum() or ch in {"-", "_", "+"}) else "_" for ch in prefix)
        return f"{safe}-{self.counter}-{uuid.uuid4().hex[:4]}"

    def _remove_candidate_artifacts(self, candidate: Candidate) -> None:
        for file_path in (candidate.checkpoint, candidate.seed_state_path):
            if file_path is None:
                continue
            if self._init_checkpoint is not None:
                try:
                    if Path(file_path).resolve() == self._init_checkpoint.resolve():
                        continue
                except OSError:
                    pass
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError:
                pass

    def save_lineage(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        nodes = []
        for cand in self._history:
            node = {
                "id": cand.ident,
                "parents": self._parents.get(cand.ident, []),
                "status": cand.status,
                "rung": cand.rung,
                "metrics": cand.metrics,
                "metadata": cand.metadata,
                "spec": cand.spec.model_dump(mode="python"),
                "mutation_trace": cand.mutation_trace,
                "crossover_report": cand.crossover_report,
            }
            nodes.append(node)
        payload = {
            "nodes": nodes,
            "novelty_archive": self._novelty_archive,
            "generation_idx": self._generation_idx,
        }
        path.write_text(json.dumps(payload, indent=2))
        console.print(f"Lineage saved to {path}")
