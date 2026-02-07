"""Torch modules assembled from the DSL for live training.

Sub-modules are split for readability:
- attention.py: RoPE, MultiHeadSelfAttention, ALiBi, activations, RMSNorm
- memory.py: RetroModule, MemoryTokensModule, ChunkMemoryModule, LookupMemoryModule,
             AssociativeMemoryModule
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from .attention import (
    ActivationLookup,
    MultiHeadSelfAttention,
    RMSNorm,
    RotaryPositionalEncoding,
    _build_alibi_bias,
    _norm_layer,
    _relu_squared,
    _swiglu,
)
from .dsl import (
    AssociativeMemoryConfig,
    BlockConfig,
    BranchRouterConfig,
    ChunkMemoryConfig,
    CustomModuleConfig,
    GatedModuleConfig,
    HyperConnectionsConfig,
    LayerScaleConfig,
    LookupMemoryConfig,
    MemoryTokensConfig,
    ModelConfig,
    MoECustomExpertConfig,
    MoEDenseExpertConfig,
    MoEFFNConfig,
    MoESSMExpertConfig,
    RecurrenceConfig,
    RetroConfig,
    SSMConfig,
)
from .memory import (
    AssociativeMemoryModule,
    ChunkMemoryModule,
    LookupMemoryModule,
    MemoryTokensModule,
    RetroModule,
)
from .plugins import get_component

# Re-export for backward compatibility
__all__ = [
    "ActivationLookup",
    "AssociativeMemoryModule",
    "BranchRouter",
    "ChunkMemoryModule",
    "CustomModule",
    "DenseFFN",
    "EvolutionBlock",
    "EvolutionModel",
    "Expert",
    "GatedModule",
    "HyperConnections",
    "LookupMemoryModule",
    "MemoryTokensModule",
    "MoELayer",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RecurrenceAdapter",
    "RetroModule",
    "RotaryPositionalEncoding",
    "SSMLayer",
    "_build_alibi_bias",
    "_norm_layer",
    "_relu_squared",
    "_swiglu",
    "count_parameters",
]


class DenseFFN(nn.Module):
    def __init__(self, dim: int, hidden: int, activation: str, dropout: float = 0.0):
        super().__init__()
        inner = hidden * 2 if activation == "swiglu" else hidden
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner if activation != "swiglu" else hidden, dim)
        self.activation_name = activation
        self.activation: Callable[[Tensor], Tensor] = ActivationLookup.get(activation) or F.silu
        self.dropout_p = float(dropout or 0.0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.activation(out)
        if self.dropout_p > 0.0:
            out = cast(Tensor, F.dropout(out, p=self.dropout_p, training=self.training))
        return cast(Tensor, self.fc2(out))


class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int, activation: str, hops: int = 1):
        super().__init__()
        self.net = DenseFFN(dim, hidden, activation, dropout=0.0)
        self.hops = max(1, hops)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.hops):
            x = cast(Tensor, self.net(x))
        return x


class MoELayer(nn.Module):
    def __init__(self, dim: int, cfg: MoEFFNConfig):
        super().__init__()
        self.cfg = cfg
        self.router_type = getattr(cfg, "router_type", "softmax")
        self.router_bias_detached = bool(getattr(cfg, "router_bias_detached", False))
        self.drop_policy = getattr(cfg, "drop_policy", "none") or "none"
        self.capacity_factor = float(getattr(cfg, "capacity_factor", 1.0) or 1.0)
        self.shared_expert_count = max(
            int(getattr(cfg, "shared", 0) or 0), 1 if getattr(cfg, "shared_expert", False) else 0
        )
        self.router = nn.Linear(dim, cfg.n_experts)
        self.experts = nn.ModuleList()
        self.shared_expert = (
            Expert(dim, cfg.hidden, activation="swiglu") if self.shared_expert_count > 0 else None
        )
        if cfg.experts:
            for idx in range(cfg.n_experts):
                # If fewer expert configs than n_experts, repeat the last one.
                if idx < len(cfg.experts):
                    ecfg = cfg.experts[idx]
                else:
                    ecfg = cfg.experts[-1]
                if isinstance(ecfg, MoEDenseExpertConfig):
                    hidden = ecfg.hidden or cfg.hidden
                    hops = ecfg.hops
                    self.experts.append(Expert(dim, hidden, activation=ecfg.activation, hops=hops))
                elif isinstance(ecfg, MoESSMExpertConfig):
                    # Wrap SSMLayer as an expert, possibly with multiple hops.
                    hops = ecfg.hops
                    ssm_layer = SSMLayer(ecfg.ssm, dim)

                    class _SSMExpert(nn.Module):
                        def __init__(self, layer: SSMLayer, num_hops: int) -> None:
                            super().__init__()
                            self.layer = layer
                            self.hops = max(1, num_hops)

                        def forward(self, x: Tensor) -> Tensor:
                            for _ in range(self.hops):
                                x = self.layer(x)
                            return x

                    self.experts.append(_SSMExpert(ssm_layer, hops))
                elif isinstance(ecfg, MoECustomExpertConfig):
                    # Use CustomModule to build a custom expert if possible.
                    custom_cfg = CustomModuleConfig(name=ecfg.name, params=ecfg.params)
                    self.experts.append(CustomModule(custom_cfg, dim))
                else:
                    # Fallback to a standard dense expert.
                    self.experts.append(Expert(dim, cfg.hidden, activation="swiglu"))
        else:
            self.experts = nn.ModuleList(
                Expert(dim, cfg.hidden, activation="swiglu") for _ in range(cfg.n_experts)
            )

    def forward(self, x: Tensor) -> Tensor:
        temp = float(self.cfg.router_temperature) if self.cfg.router_temperature else 1.0
        bias = self.router.bias.detach() if self.router_bias_detached else self.router.bias
        logits = F.linear(x, self.router.weight, bias) / temp
        topk_val, topk_idx = torch.topk(logits, k=self.cfg.k, dim=-1)
        if self.router_type == "sigmoid":
            weights = torch.sigmoid(topk_val)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weights = weights / denom
        else:
            weights = torch.softmax(topk_val, dim=-1)
        outputs = torch.zeros_like(x)
        overflow_count = 0
        capacity = None
        if self.drop_policy != "none" and self.cfg.n_experts > 0:
            # Approximate Switch-style capacity per expert.
            bsz, seq_len, _ = x.shape
            assignments = max(1, int(bsz * seq_len * self.cfg.k))
            capacity = max(1, int(self.capacity_factor * (assignments / self.cfg.n_experts)))
        # Track simple routing stats for aux losses/metrics
        # Entropy over top-k weights (normalized to [0,1] by log(k))
        eps = 1e-8
        entropy = -(weights * (weights + eps).log()).sum(dim=-1).mean()
        self.last_entropy = entropy
        # Load-balance proxy: selection frequency deviation from uniform
        with torch.no_grad():
            b, t, _ = x.shape
            total = max(1, b * t * self.cfg.k)
            counts = torch.zeros(self.cfg.n_experts, device=x.device)
            for expert_pos in range(self.cfg.k):
                idx = topk_idx[..., expert_pos].reshape(-1)
                counts += torch.bincount(idx, minlength=self.cfg.n_experts).float()
            freq = counts / total
            uniform = 1.0 / self.cfg.n_experts
            lb = ((freq - uniform) ** 2).mean()
        self.last_lb = lb
        # Persist the last routing frequency histogram for tooling/metrics.
        self.last_load = freq
        for expert_pos in range(self.cfg.k):
            idx = topk_idx[..., expert_pos]
            weight = weights[..., expert_pos].unsqueeze(-1)
            expert_outputs = torch.zeros_like(x)
            for expert_id in range(self.cfg.n_experts):
                mask = idx == expert_id
                if mask.any():
                    if capacity is not None and self.drop_policy == "greedy":
                        flat = mask.reshape(-1)
                        positions = flat.nonzero(as_tuple=False).squeeze(-1)
                        if positions.numel() > capacity:
                            overflow_count += int(positions.numel() - capacity)
                            kept = positions[:capacity]
                            flat = torch.zeros_like(flat)
                            flat[kept] = True
                            mask = flat.view_as(mask)
                    expert_out = self.experts[expert_id](x[mask])
                    expert_outputs = expert_outputs.index_put((mask,), expert_out)
            outputs = outputs + expert_outputs * weight
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x)
            outputs = outputs + shared_out
        total_assignments = max(1, int(x.shape[0] * x.shape[1] * self.cfg.k))
        self.last_overflow = float(overflow_count) / float(total_assignments)
        return outputs

    def sort_experts(self) -> None:
        norms = []
        for expert in self.experts:
            if isinstance(expert, Expert):
                norms.append(expert.net.fc1.weight.norm().item())
            else:
                norms.append(0.0)
        permutation = sorted(range(len(norms)), key=lambda i: norms[i], reverse=True)
        self.reorder(permutation)

    def reorder(self, permutation: list[int]) -> None:
        self.experts = nn.ModuleList([self.experts[i] for i in permutation])
        with torch.no_grad():
            self.router.weight[:] = self.router.weight[permutation]
            self.router.bias[:] = self.router.bias[permutation]


class SSMLayer(nn.Module):
    def __init__(self, cfg: SSMConfig, dim: int):
        super().__init__()
        self.cfg = cfg
        inner = int(getattr(cfg, "d_state", dim) or dim)
        inner = max(1, inner)
        self.in_proj = nn.Linear(dim, inner)
        self.conv = nn.Conv1d(
            in_channels=inner,
            out_channels=inner,
            kernel_size=cfg.d_conv,
            padding=0,
            groups=1,
        )
        self.out_proj = nn.Linear(inner, dim)
        self._conv_left_pad = max(0, int(cfg.d_conv) - 1)

    def forward(self, x: Tensor) -> Tensor:
        gate = float(getattr(self.cfg, "gate", 1.0) or 1.0)
        h = self.in_proj(x)
        seq_in = h.transpose(1, 2)
        if self._conv_left_pad:
            seq_in = F.pad(seq_in, (self._conv_left_pad, 0))
        seq = self.conv(seq_in).transpose(1, 2)
        out = self.out_proj(cast(Tensor, seq))
        return cast(Tensor, out * gate)


class BranchRouter(nn.Module):
    def __init__(self, cfg: BranchRouterConfig, dim: int):
        super().__init__()
        self.cfg = cfg
        self.targets = list(cfg.targets or [])
        self.dropout = nn.Dropout(float(getattr(cfg, "dropout", 0.0) or 0.0))
        self.temperature = float(getattr(cfg, "temperature", 1.0) or 1.0)
        self.last_entropy: Tensor | None = None
        self.net: nn.Module
        n_targets = len(self.targets)
        hidden = getattr(cfg, "hidden", None)
        if hidden:
            hidden_dim = int(hidden)
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, n_targets),
            )
        else:
            self.net = nn.Linear(dim, n_targets)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        if not self.targets:
            return x.new_zeros(x.shape[0], x.shape[1], 0)
        x_in = self.dropout(x)
        temp = self.temperature if self.temperature > 0.0 else 1.0
        if getattr(self.cfg, "router_type", "token") == "sequence":
            logits = self.net(x_in.mean(dim=1, keepdim=True)) / temp
            weights = torch.softmax(logits, dim=-1).expand(-1, x.shape[1], -1)
        else:
            logits = self.net(x_in) / temp
            weights = torch.softmax(logits, dim=-1)
        eps = 1e-8
        entropy = -(weights * (weights + eps).log()).sum(dim=-1).mean()
        self.last_entropy = entropy
        return cast(Tensor, weights)


class CustomModule(nn.Module):
    def __init__(self, cfg: CustomModuleConfig, dim: int):
        super().__init__()
        inner = cfg.params.get("dim", dim)
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.SiLU(),
            nn.Linear(inner, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class GatedModule(nn.Module):
    def __init__(self, cfg: GatedModuleConfig):
        super().__init__()
        weight = torch.tensor(cfg.init_weight, dtype=torch.float32)
        if cfg.learnable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


class EvolutionBlock(nn.Module):
    def __init__(self, dim: int, cfg: BlockConfig, norm_type: str = "layernorm"):
        super().__init__()
        self.attn = MultiHeadSelfAttention(cfg.attn, dim) if cfg.attn else None
        if cfg.ffn is None:
            self.ffn: DenseFFN | MoELayer | None = None
        elif cfg.ffn.type == "moe":
            if not isinstance(cfg.ffn, MoEFFNConfig):
                msg = "MoE block requires MoEFFNConfig."
                raise TypeError(msg)
            self.ffn = MoELayer(dim, cfg.ffn)
        else:
            self.ffn = DenseFFN(
                dim,
                cfg.ffn.hidden,
                getattr(cfg.ffn, "activation", "silu"),
                dropout=float(getattr(cfg.ffn, "dropout", 0.0) or 0.0),
            )
        self.ssm = SSMLayer(cfg.ssm, dim) if cfg.ssm else None
        self.norm = _norm_layer(norm_type, dim)
        self.router: BranchRouter | None = None
        self.memory_extras = nn.ModuleList()
        self.extras = nn.ModuleList()
        self._gate_params = nn.ParameterDict()
        self._gate_buffers: dict[str, str] = {}
        self._layer_scale_params = nn.ParameterDict()
        self._layer_scale_buffers: dict[str, str] = {}
        gate_cfgs: list[GatedModuleConfig] = []
        layerscale_cfgs: list[LayerScaleConfig] = []
        router_cfgs: list[BranchRouterConfig] = []
        for extra in cfg.extras:
            if isinstance(extra, RetroConfig):
                self.memory_extras.append(RetroModule(extra))
            elif isinstance(extra, GatedModuleConfig):
                gate_cfgs.append(extra)
            elif isinstance(extra, AssociativeMemoryConfig):
                self.memory_extras.append(AssociativeMemoryModule(extra, dim))
            elif isinstance(extra, MemoryTokensConfig):
                self.memory_extras.append(MemoryTokensModule(extra, dim))
            elif isinstance(extra, ChunkMemoryConfig):
                self.memory_extras.append(ChunkMemoryModule(extra, dim))
            elif isinstance(extra, LookupMemoryConfig):
                self.memory_extras.append(LookupMemoryModule(extra, dim))
            elif isinstance(extra, BranchRouterConfig):
                router_cfgs.append(extra)
            elif isinstance(extra, LayerScaleConfig):
                layerscale_cfgs.append(extra)
            elif isinstance(extra, CustomModuleConfig):
                builder = get_component(extra.name)
                if builder is not None:
                    self.extras.append(builder(extra, dim))
                else:
                    self.extras.append(CustomModule(extra, dim))
        if router_cfgs:
            self.router = BranchRouter(router_cfgs[-1], dim)
        if gate_cfgs:
            learnable = any(cfg.learnable for cfg in gate_cfgs)
            init_by_target: dict[str, float] = {}
            ordered_targets: list[str] = []
            for gate_cfg in gate_cfgs:
                for target in gate_cfg.targets:
                    name = str(target)
                    init_by_target[name] = float(gate_cfg.init_weight)
                    if name not in ordered_targets:
                        ordered_targets.append(name)
            eps = 1e-6
            for target in ordered_targets:
                init = init_by_target.get(target, 1.0)
                init = max(eps, min(1.0 - eps, float(init)))
                logit = math.log(init / (1.0 - init))
                if learnable:
                    self._gate_params[target] = nn.Parameter(
                        torch.tensor(logit, dtype=torch.float32)
                    )
                else:
                    buf_name = f"gate_{target}_logit"
                    self.register_buffer(buf_name, torch.tensor(logit, dtype=torch.float32))
                    self._gate_buffers[target] = buf_name
        if layerscale_cfgs:
            ls_init_by_target: dict[str, float] = {}
            learnable_by_target: dict[str, bool] = {}
            ls_ordered_targets: list[str] = []
            for ls_cfg in layerscale_cfgs:
                init = float(getattr(ls_cfg, "init", 1e-5) or 1e-5)
                learnable = bool(getattr(ls_cfg, "learnable", True))
                for target in ls_cfg.targets:
                    name = str(target)
                    ls_init_by_target[name] = init
                    learnable_by_target[name] = learnable_by_target.get(name, False) or learnable
                    if name not in ls_ordered_targets:
                        ls_ordered_targets.append(name)
            for target in ls_ordered_targets:
                init = ls_init_by_target.get(target, 1e-5)
                vec = torch.full((dim,), float(init), dtype=torch.float32)
                if learnable_by_target.get(target, True):
                    self._layer_scale_params[target] = nn.Parameter(vec)
                else:
                    buf_name = f"layer_scale_{target}"
                    self.register_buffer(buf_name, vec)
                    self._layer_scale_buffers[target] = buf_name

    def _gate_scale(self, name: str, x: Tensor) -> Tensor:
        if name in self._gate_params:
            return torch.sigmoid(self._gate_params[name]).to(dtype=x.dtype, device=x.device)
        buf_name = self._gate_buffers.get(name)
        if buf_name:
            buf = getattr(self, buf_name)
            if isinstance(buf, torch.Tensor):
                return torch.sigmoid(buf).to(dtype=x.dtype, device=x.device)
        return x.new_tensor(1.0)

    def _layer_scale(self, name: str, x: Tensor) -> Tensor:
        if name in self._layer_scale_params:
            return cast(Tensor, self._layer_scale_params[name]).to(dtype=x.dtype, device=x.device)
        buf_name = self._layer_scale_buffers.get(name)
        if buf_name:
            buf = getattr(self, buf_name)
            if isinstance(buf, torch.Tensor):
                return buf.to(dtype=x.dtype, device=x.device)
        return x.new_tensor(1.0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.delta(x)

    def delta(self, x: Tensor) -> Tensor:
        """Return the block's residual update (x_out - x_in)."""
        x_in = x
        if self.router is None:
            if self.attn:
                out = self.attn(self.norm(x))
                out = out * self._layer_scale("attn", out)
                x = x + out * self._gate_scale("attn", out)
            if self.ssm:
                out = self.ssm(self.norm(x))
                out = out * self._layer_scale("ssm", out)
                x = x + out * self._gate_scale("ssm", out)
            if self.ffn:
                out = self.ffn(self.norm(x))
                out = out * self._layer_scale("ffn", out)
                x = x + out * self._gate_scale("ffn", out)
            for memory_module in self.memory_extras:
                out = memory_module(self.norm(x))
                out = out * self._layer_scale("memory", out)
                x = x + out * self._gate_scale("memory", out)
            for extra in self.extras:
                x = x + extra(self.norm(x))  # nosec B610 - pure tensor gating, no SQL context
            return cast(Tensor, x - x_in)

        x_norm = self.norm(x)
        weights = self.router(x_norm)
        mixed = torch.zeros_like(x)
        for idx, target in enumerate(self.router.targets):
            out = None
            if target == "attn" and self.attn is not None:
                out = self.attn(x_norm)
            elif target == "ssm" and self.ssm is not None:
                out = self.ssm(x_norm)
            elif target == "ffn" and self.ffn is not None:
                out = self.ffn(x_norm)
            elif target == "memory" and len(self.memory_extras) > 0:
                out = sum(mem(x_norm) for mem in self.memory_extras)
            if out is None:
                out = torch.zeros_like(x)
            out = out * self._layer_scale(target, out)
            out = out * self._gate_scale(target, out)
            w = weights[..., idx].unsqueeze(-1).to(dtype=out.dtype)
            mixed = mixed + out * w
        x = x + mixed
        for extra in self.extras:
            x = x + extra(self.norm(x))  # nosec B610 - pure tensor gating, no SQL context
        return cast(Tensor, x - x_in)


class HyperConnections(nn.Module):
    """Hyper-residual controller: N residual lanes + per-layer mixing."""

    def __init__(self, cfg: HyperConnectionsConfig, *, layers: int, dim: int):
        super().__init__()
        self.streams = int(getattr(cfg, "streams", 1) or 1)
        self.layers = int(layers)
        self.dim = int(dim)
        self.update_scale = float(getattr(cfg, "update_scale", 1.0) or 1.0)
        self.diag_bias = float(getattr(cfg, "diag_bias", 4.0) or 4.0)
        self.noise_std = float(getattr(cfg, "noise_std", 0.0) or 0.0)

        self.pre_logits = nn.Parameter(torch.zeros(self.layers, self.streams))
        self.post_logits = nn.Parameter(torch.zeros(self.layers, self.streams))
        self.res_logits = nn.Parameter(torch.zeros(self.layers, self.streams, self.streams))
        self.out_logits = nn.Parameter(torch.zeros(self.streams))

        self._init_logits()

    def _init_logits(self) -> None:
        if self.layers <= 0 or self.streams <= 1:
            return
        with torch.no_grad():
            self.pre_logits.zero_()
            self.post_logits.zero_()
            self.out_logits.zero_()
            self.res_logits.zero_()
            eye = torch.eye(self.streams)
            self.res_logits.add_(eye[None, :, :] * self.diag_bias)
            if self.noise_std > 0.0:
                self.pre_logits.add_(torch.randn_like(self.pre_logits) * self.noise_std)
                self.post_logits.add_(torch.randn_like(self.post_logits) * self.noise_std)
                self.out_logits.add_(torch.randn_like(self.out_logits) * self.noise_std)
                self.res_logits.add_(torch.randn_like(self.res_logits) * self.noise_std)

    def enabled(self) -> bool:
        return self.streams > 1

    def init_streams(self, x: Tensor) -> Tensor:
        # x: (B, T, D) -> (B, T, N, D)
        return x.unsqueeze(2).repeat(1, 1, self.streams, 1)

    def _pre(self, layer_idx: int, *, ref: Tensor) -> Tensor:
        weights = torch.softmax(self.pre_logits[layer_idx], dim=-1)
        return weights.to(dtype=ref.dtype, device=ref.device)

    def _post(self, layer_idx: int, *, ref: Tensor) -> Tensor:
        weights = torch.softmax(self.post_logits[layer_idx], dim=-1)
        return weights.to(dtype=ref.dtype, device=ref.device)

    def _res(self, layer_idx: int, *, ref: Tensor) -> Tensor:
        # Row-normalized mixing (each row sums to 1).
        weights = torch.softmax(self.res_logits[layer_idx], dim=-1)
        return weights.to(dtype=ref.dtype, device=ref.device)

    def readout(self, streams: Tensor) -> Tensor:
        # streams: (B, T, N, D) -> (B, T, D)
        weights = torch.softmax(self.out_logits, dim=-1).to(
            dtype=streams.dtype, device=streams.device
        )
        return cast(Tensor, (streams * weights.view(1, 1, -1, 1)).sum(dim=2))


class EvolutionModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        vocab = cfg.emb.vocab or cfg.head.vocab
        if vocab is None:
            raise ValueError("Vocabulary must be specified in emb or head config.")
        self.embed = nn.Embedding(vocab, cfg.emb.dim)
        init_std = float(getattr(cfg.emb, "init_std", 0.02) or 0.02)
        nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)
        self.emb_dropout = nn.Dropout(float(getattr(cfg.emb, "dropout", 0.0) or 0.0))
        self.blocks = nn.ModuleList(
            [EvolutionBlock(cfg.emb.dim, block, norm_type=cfg.norm) for block in cfg.blocks]
        )
        self.hyper: HyperConnections | None = None
        hyper_cfg = getattr(cfg, "hyper", None)
        if isinstance(hyper_cfg, HyperConnectionsConfig) and hyper_cfg.streams > 1:
            self.hyper = HyperConnections(hyper_cfg, layers=len(cfg.blocks), dim=cfg.emb.dim)
        self.norm = _norm_layer(cfg.norm, cfg.emb.dim)
        self.lm_head = nn.Linear(cfg.emb.dim, cfg.head.vocab)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)
        if getattr(cfg.head, "tie_embeddings", True):
            if cfg.head.vocab != vocab:
                raise ValueError("tie_embeddings requires emb.vocab == head.vocab")
            self.lm_head.weight = self.embed.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=init_std)
        self._grad_checkpointing = False
        # Recurrence setup
        self._recurrence_order: list[tuple[int, RecurrenceConfig]] = sorted(
            enumerate(cfg.recurrences), key=lambda item: item[1].start
        )
        self.recurrence_adapters = nn.ModuleList()
        self.recurrence_steps: dict[int, int] = {}
        recurrence_dim = cfg.emb.dim
        if self.hyper is not None and self.hyper.enabled():
            recurrence_dim = cfg.emb.dim * self.hyper.streams
        for idx, rec_cfg in self._recurrence_order:
            concat_dim = recurrence_dim * 2 if rec_cfg.concat_prelude else recurrence_dim
            adapter = RecurrenceAdapter(
                input_dim=concat_dim,
                output_dim=recurrence_dim,
                kind=rec_cfg.adapter,
            )
            self.recurrence_adapters.append(adapter)
            self.recurrence_steps[idx] = rec_cfg.train_recurrence

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        x = cast(Tensor, self.emb_dropout(x))
        if self.hyper is not None and self.hyper.enabled():
            streams = self.hyper.init_streams(x)
            if not self._recurrence_order:
                for idx, block in enumerate(self.blocks):
                    streams = self._run_hyper_block(idx, cast(EvolutionBlock, block), streams)
            else:
                idx = 0
                order_pos = 0
                while idx < len(self.blocks):
                    if (
                        order_pos < len(self._recurrence_order)
                        and idx == self._recurrence_order[order_pos][1].start
                    ):
                        spec_idx, rec_cfg = self._recurrence_order[order_pos]
                        streams = self._run_recurrence_hyper(spec_idx, rec_cfg, streams)
                        idx = rec_cfg.end
                        order_pos += 1
                        continue
                    streams = self._run_hyper_block(
                        idx, cast(EvolutionBlock, self.blocks[idx]), streams
                    )
                    idx += 1
            x = self.hyper.readout(streams)
            x = self.norm(x)
            return cast(Tensor, self.lm_head(x))
        if not self._recurrence_order:
            for block in self.blocks:
                x = self._run_block(block, x)
        else:
            idx = 0
            order_pos = 0
            while idx < len(self.blocks):
                if (
                    order_pos < len(self._recurrence_order)
                    and idx == self._recurrence_order[order_pos][1].start
                ):
                    spec_idx, rec_cfg = self._recurrence_order[order_pos]
                    x = self._run_recurrence(spec_idx, rec_cfg, x)
                    idx = rec_cfg.end
                    order_pos += 1
                    continue
                x = self._run_block(self.blocks[idx], x)
                idx += 1
        x = self.norm(x)
        return cast(Tensor, self.lm_head(x))

    def set_grad_checkpointing(self, enabled: bool) -> None:
        self._grad_checkpointing = bool(enabled)

    def _run_block(self, block: nn.Module, x: Tensor) -> Tensor:
        if self._grad_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            try:
                return cast(Tensor, checkpoint(block, x, use_reentrant=False))
            except TypeError:
                # Older torch versions do not support use_reentrant.
                return cast(Tensor, checkpoint(block, x))
        return cast(Tensor, block(x))

    def _run_hyper_block(self, layer_idx: int, block: EvolutionBlock, streams: Tensor) -> Tensor:
        if self.hyper is None:
            raise RuntimeError("hyper block requested without hyper configuration")
        if self._grad_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            def _fn(state: Tensor) -> Tensor:
                return self._hyper_step(layer_idx, block, state)

            try:
                return cast(Tensor, checkpoint(_fn, streams, use_reentrant=False))
            except TypeError:
                return cast(Tensor, checkpoint(_fn, streams))
        return self._hyper_step(layer_idx, block, streams)

    def _hyper_step(self, layer_idx: int, block: EvolutionBlock, streams: Tensor) -> Tensor:
        if self.hyper is None:
            raise RuntimeError("hyper step requested without hyper configuration")
        pre = self.hyper._pre(layer_idx, ref=streams)
        x_pre = cast(Tensor, (streams * pre.view(1, 1, -1, 1)).sum(dim=2))
        delta = block.delta(x_pre)
        post = self.hyper._post(layer_idx, ref=streams)
        update = delta.unsqueeze(2) * post.view(1, 1, -1, 1) * float(self.hyper.update_scale)
        res = self.hyper._res(layer_idx, ref=streams)
        mixed = torch.einsum("ij,btjd->btid", res, streams)
        return cast(Tensor, mixed + update)

    def set_recurrence_steps(self, steps: dict[int, int]) -> None:
        for idx, value in steps.items():
            self.recurrence_steps[idx] = max(1, int(value))

    def _run_recurrence(
        self,
        spec_idx: int,
        rec_cfg: RecurrenceConfig,
        prelude_output: Tensor,
    ) -> Tensor:
        # Determine current loop count
        end_idx = min(rec_cfg.end, len(self.blocks))
        start_idx = min(rec_cfg.start, max(0, end_idx - 1))
        if end_idx - start_idx <= 1:
            return prelude_output
        current_steps = max(1, self.recurrence_steps.get(spec_idx, rec_cfg.train_recurrence))
        state = self._init_recurrence_state(rec_cfg, prelude_output)
        adapter = self.recurrence_adapters[self._adapter_position(spec_idx)]
        for _ in range(current_steps):
            h = state
            for block_idx in range(start_idx, end_idx):
                h = self._run_block(self.blocks[block_idx], h)
            adapter_input = torch.cat([prelude_output, h], dim=-1) if rec_cfg.concat_prelude else h
            state = adapter(adapter_input, h)
        return state

    def _run_recurrence_hyper(
        self,
        spec_idx: int,
        rec_cfg: RecurrenceConfig,
        prelude_output: Tensor,
    ) -> Tensor:
        if self.hyper is None or not self.hyper.enabled():
            raise RuntimeError("hyper recurrence requested without hyper configuration")
        end_idx = min(rec_cfg.end, len(self.blocks))
        start_idx = min(rec_cfg.start, max(0, end_idx - 1))
        if end_idx - start_idx <= 1:
            return prelude_output
        current_steps = max(1, self.recurrence_steps.get(spec_idx, rec_cfg.train_recurrence))
        state = self._init_recurrence_state(rec_cfg, prelude_output)
        adapter = self.recurrence_adapters[self._adapter_position(spec_idx)]
        b, t, n, d = prelude_output.shape
        flat_dim = n * d
        prelude_flat = prelude_output.reshape(b, t, flat_dim)
        for _ in range(current_steps):
            h = state
            for block_idx in range(start_idx, end_idx):
                h = self._run_hyper_block(
                    block_idx, cast(EvolutionBlock, self.blocks[block_idx]), h
                )
            h_flat = h.reshape(b, t, flat_dim)
            adapter_input = (
                torch.cat([prelude_flat, h_flat], dim=-1) if rec_cfg.concat_prelude else h_flat
            )
            state_flat = adapter(adapter_input, h_flat)
            state = state_flat.view(b, t, n, d)
        return state

    def _init_recurrence_state(self, cfg: RecurrenceConfig, reference: Tensor) -> Tensor:
        if cfg.init_state == "noise":
            return cast(Tensor, torch.randn_like(reference) * cfg.noise_std)
        return cast(Tensor, torch.zeros_like(reference))

    def _adapter_position(self, spec_idx: int) -> int:
        for pos, (cfg_idx, _) in enumerate(self._recurrence_order):
            if cfg_idx == spec_idx:
                return pos
        return 0


class RecurrenceAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kind: str = "linear"):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.proj = nn.Linear(input_dim, output_dim)
        elif kind == "gated":
            self.val = nn.Linear(input_dim, output_dim)
            self.gate = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported adapter kind '{kind}'")

    def forward(self, adapter_input: Tensor, residual: Tensor) -> Tensor:
        if self.kind == "linear":
            return cast(Tensor, self.proj(adapter_input))
        gate = torch.sigmoid(self.gate(adapter_input))
        value = torch.tanh(self.val(adapter_input))
        return cast(Tensor, gate * value + (1 - gate) * residual)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


## RMSNorm, _norm_layer -> moved to attention.py (imported above)
