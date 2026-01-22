"""Torch modules assembled from the DSL for live training."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from .dsl import (
    AssociativeMemoryConfig,
    AttentionConfig,
    BlockConfig,
    BranchRouterConfig,
    ChunkMemoryConfig,
    CustomModuleConfig,
    GatedModuleConfig,
    HyperConnectionsConfig,
    LayerScaleConfig,
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
from .plugins import get_component


def _swiglu(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return F.silu(x1) * x2


ActivationLookup: dict[str, Callable[[Tensor], Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "swiglu": _swiglu,
}


class RotaryPositionalEncoding(nn.Module):
    """Minimal RoPE implementation for experimentation."""

    inv_freq: Tensor

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        *,
        rope_type: str = "standard",
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.rope_type = (rope_type or "standard").lower()
        self.scale_factor = float(scale_factor or 1.0)

        effective_base = self.base
        if self.rope_type in {"ntk", "yarn"} and self.scale_factor != 1.0:
            denom = max(1.0, float(dim - 2))
            exponent = float(dim) / denom
            effective_base = effective_base * (self.scale_factor**exponent)

        inv_freq = 1.0 / (effective_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.inv_freq = cast(Tensor, self.inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> Tensor:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        if self.rope_type in {"linear", "yarn"} and self.scale_factor != 1.0:
            t = t / self.scale_factor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

    @staticmethod
    def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
        # x: (B, T, H, D)
        seq_len = x.size(1)
        rope = rope[:seq_len, :]
        rot_dim = min(rope.size(-1), x.size(-1))
        cos = rope.cos()[None, :, None, :rot_dim]
        sin = rope.sin()[None, :, None, :rot_dim]
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]
        cos_part = cos[..., ::2]
        sin_part = sin[..., ::2]
        rotated = torch.stack(
            (x1 * cos_part - x2 * sin_part, x1 * sin_part + x2 * cos_part),
            dim=-1,
        ).flatten(-2)
        if x_pass.numel():
            rotated = torch.cat([rotated, x_pass], dim=-1)
        return rotated


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig, dim: int):
        super().__init__()
        self.cfg = cfg
        self.kind = str(getattr(cfg, "kind", "MHA") or "MHA").upper()
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.kv_groups = cfg.kv_groups or 1
        self.n_kv_heads = max(1, self.heads // self.kv_groups)
        self.dropout_p = float(getattr(cfg, "dropout", 0.0) or 0.0)

        if self.kind == "MLA":
            latent_dim = int(getattr(cfg, "kv_latent_dim", 0) or 0)
            if latent_dim <= 0:
                latent_dim = int(self.n_kv_heads * self.head_dim)
            self.q_proj = nn.Linear(dim, self.heads * self.head_dim, bias=True)
            self.kv_down = nn.Linear(dim, latent_dim, bias=True)
            self.kv_up = nn.Linear(latent_dim, 2 * self.n_kv_heads * self.head_dim, bias=True)
            self.c_proj = nn.Linear(self.heads * self.head_dim, dim, bias=True)
        else:
            self.c_attn = nn.Linear(
                dim,
                (self.heads + 2 * self.n_kv_heads) * self.head_dim,
                bias=True,
            )
            self.c_proj = nn.Linear(self.heads * self.head_dim, dim, bias=True)

        self.gating_pos = getattr(cfg, "gating_pos", "none")
        self.gating_op = getattr(cfg, "gating_op", "dense")
        # Selector-based sparsity (content-dependent top-k)
        self.selector_mode = getattr(cfg, "selector", "none") or "none"
        self.selector_topk = getattr(cfg, "selector_topk", None)
        self.selector_heads = getattr(cfg, "selector_heads", None)
        self.selector_dim = getattr(cfg, "selector_dim", None)
        self.selector_rope = getattr(cfg, "selector_rope", "none") or "none"
        self.selector_detach = bool(getattr(cfg, "selector_detach", False))

        if self.gating_pos != "none":
            # Head-specific gating: G_h = Sigmoid(Op(Q_h))
            if self.gating_op == "dense":
                # Weights: (heads, head_dim, head_dim)
                self.gate_weight = nn.Parameter(
                    torch.empty(self.heads, self.head_dim, self.head_dim)
                )
                nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
            else:
                # Diagonal gating weights: (heads, head_dim)
                self.gate_weight = nn.Parameter(torch.empty(self.heads, self.head_dim))
                nn.init.uniform_(self.gate_weight, -0.1, 0.1)

            self.gate_bias = nn.Parameter(torch.zeros(self.heads, self.head_dim))

        self.rope: RotaryPositionalEncoding | None
        rope_mode = str(getattr(cfg, "rope", "") or "").lower()
        if rope_mode and rope_mode not in {"none", "off", "false", "0"}:
            self.rope = RotaryPositionalEncoding(
                cfg.head_dim,
                float(cfg.rope_theta or 10000.0),
                rope_type=rope_mode,
                scale_factor=float(getattr(cfg, "rope_factor", None) or 1.0),
            )
        else:
            self.rope = None
        self._impl_logged = False

    def _build_selector_mask(self, q: Tensor, k: Tensor, *, causal: bool) -> Tensor:
        """Return a float mask (0 / -inf) for selector sparsity.

        q, k are expected in (B, H, T, D) layout.
        """
        b, h, t, d = q.shape
        topk_raw = int(self.selector_topk or 0)
        keep = max(1, min(topk_raw if topk_raw > 0 else 64, t))
        sel_dim_raw = int(self.selector_dim or 0)
        sel_dim = max(1, min(sel_dim_raw if sel_dim_raw > 0 else d, d))
        h_sel_raw = int(self.selector_heads or 0)
        h_sel = max(1, min(h_sel_raw if h_sel_raw > 0 else 1, h))

        q_sel = q[:, :h_sel, :, :sel_dim].to(dtype=torch.float32)
        k_sel = k[:, :h_sel, :, :sel_dim].to(dtype=torch.float32)

        ctx: contextlib.AbstractContextManager[None]
        ctx = torch.no_grad() if self.selector_detach else contextlib.nullcontext()
        with ctx:
            scores = torch.matmul(q_sel, k_sel.transpose(-1, -2)) / math.sqrt(max(1, sel_dim))
            if causal:
                future = torch.triu(
                    torch.ones(t, t, device=scores.device, dtype=torch.bool), diagonal=1
                )
                scores = scores.masked_fill(future, float("-inf"))

            indices = scores.topk(k=keep, dim=-1).indices  # (B, H_sel, T, K)
            selected = torch.zeros((b, h_sel, t, t), device=scores.device, dtype=torch.bool)
            selected.scatter_(
                dim=-1,
                index=indices,
                src=torch.ones_like(indices, dtype=torch.bool),
            )
            diag = torch.arange(t, device=scores.device)
            selected[:, :, diag, diag] = True
            if causal:
                causal_allowed = torch.tril(
                    torch.ones(t, t, device=scores.device, dtype=torch.bool)
                )
                selected = selected & causal_allowed

            if h_sel < h:
                selected = selected.any(dim=1, keepdim=True)
            attn_mask = torch.where(selected, 0.0, float("-inf")).to(dtype=torch.float32)
            return cast(Tensor, attn_mask)

    def forward(self, x: Tensor) -> Tensor:
        # Optional input norm clamp to stabilize Q/K magnitudes
        if getattr(self.cfg, "qk_norm_max", None):
            max_norm = float(self.cfg.qk_norm_max)  # type: ignore[arg-type]
            eps = 1e-6
            norms = x.norm(dim=-1, keepdim=True).clamp_min(eps)
            scale = (max_norm / norms).clamp(max=1.0)
            x = x * scale

        b, t, d = x.shape
        if self.kind == "MLA":
            q = self.q_proj(x)
            kv_latent = self.kv_down(x)
            kv = self.kv_up(kv_latent)
            q = q.view(b, t, self.heads, self.head_dim)
            kv_size = self.n_kv_heads * self.head_dim
            k_raw, v_raw = torch.split(kv, [kv_size, kv_size], dim=-1)
            k = k_raw.view(b, t, self.n_kv_heads, self.head_dim)
            v = v_raw.view(b, t, self.n_kv_heads, self.head_dim)
        else:
            qkv = self.c_attn(x)
            q_size = self.heads * self.head_dim
            kv_size = self.n_kv_heads * self.head_dim
            q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
            q = q.view(b, t, self.heads, self.head_dim)
            k = k.view(b, t, self.n_kv_heads, self.head_dim)
            v = v.view(b, t, self.n_kv_heads, self.head_dim)

        selector_active = self.selector_mode != "none"
        q_unrot = q
        k_unrot = k
        rope_emb = None
        if self.rope is not None:
            rope_emb = self.rope(t, x.device)
            q = RotaryPositionalEncoding.apply_rotary(q, rope_emb)
            k = RotaryPositionalEncoding.apply_rotary(k, rope_emb)

        causal = bool(getattr(self.cfg, "causal", True))
        alibi = bool(getattr(self.cfg, "alibi", False))
        sparsity = getattr(self.cfg, "sparsity", "none")
        sw = getattr(self.cfg, "sw", None)
        static_patterns = sparsity != "none" or (sparsity == "none" and sw)

        linear_enabled = self.kind == "LINEAR"
        if linear_enabled and (selector_active or sparsity != "none" or sw or alibi):
            linear_enabled = False

        attn_mask: Tensor | None = None
        if not linear_enabled:
            static_mask: Tensor | None = None
            if static_patterns:
                static_mask = torch.full((t, t), float("-inf"), device=x.device)
                if sparsity == "local_block":
                    w = int(self.cfg.sw or self.head_dim)
                    for i in range(t):
                        lo = max(0, i - w)
                        hi = i + 1 if causal else min(t, i + w + 1)
                        static_mask[i, lo:hi] = 0.0
                    bsz = int(self.cfg.block_size or w)
                    stride = int(getattr(self.cfg, "block_stride", bsz))
                    for i in range(t):
                        hi = i + 1 if causal else t
                        for start in range(0, hi, max(1, stride)):
                            end = min(hi, start + bsz)
                            if end > start:
                                static_mask[i, start:end] = 0.0
                elif sparsity == "local_global":
                    w = int(self.cfg.sw or self.head_dim)
                    gstride = int(getattr(self.cfg, "global_stride", 0) or 0)
                    for i in range(t):
                        lo = max(0, i - w)
                        hi = i + 1 if causal else min(t, i + w + 1)
                        static_mask[i, lo:hi] = 0.0
                        if gstride > 0:
                            global_idx = torch.arange(
                                0, i + 1 if causal else t, gstride, device=x.device
                            )
                            static_mask[i, global_idx] = 0.0
                        static_mask[i, 0] = 0.0
                elif sparsity == "block" and getattr(self.cfg, "block_size", None):
                    bsz = int(self.cfg.block_size or 0)
                    stride = int(getattr(self.cfg, "block_stride", self.cfg.block_size or bsz))
                    for start in range(0, t, max(1, stride)):
                        end = min(t, start + bsz)
                        for i in range(start, end):
                            hi = min(end, i + 1) if causal else end
                            static_mask[i, start:hi] = 0.0
                elif sparsity == "dilated" and getattr(self.cfg, "dilation", None):
                    dilation = max(1, int(self.cfg.dilation or 1))
                    for i in range(t):
                        for offset in range(min(dilation, t)):
                            if i % dilation != offset:
                                continue
                            idx = torch.arange(
                                offset, (i + 1) if causal else t, dilation, device=x.device
                            )
                            static_mask[i, idx] = 0.0
                else:
                    sliding_active = sparsity == "sliding" or (
                        sparsity == "none" and getattr(self.cfg, "sw", None)
                    )
                    if sliding_active:
                        w = int(self.cfg.sw or self.head_dim)
                        for i in range(t):
                            lo = max(0, i - w)
                            hi = i + 1 if causal else min(t, i + w + 1)
                            static_mask[i, lo:hi] = 0.0

            selector_mask: Tensor | None = None
            if selector_active:
                if rope_emb is None or self.selector_rope == "none":
                    q_sel = q_unrot
                    k_sel = k_unrot
                elif self.selector_rope == "full":
                    q_sel = q
                    k_sel = k
                else:
                    rot_dim = max(
                        0,
                        min(int(self.selector_dim or self.head_dim), int(self.head_dim // 2)),
                    )
                    rot_dim = (rot_dim // 2) * 2
                    if rot_dim <= 0:
                        q_sel = q_unrot
                        k_sel = k_unrot
                    else:
                        rope_slice = rope_emb[:, :rot_dim]
                        q_rot_part = RotaryPositionalEncoding.apply_rotary(
                            q_unrot[..., :rot_dim], rope_slice
                        )
                        k_rot_part = RotaryPositionalEncoding.apply_rotary(
                            k_unrot[..., :rot_dim], rope_slice
                        )
                        q_sel = torch.cat([q_rot_part, q_unrot[..., rot_dim:]], dim=-1)
                        k_sel = torch.cat([k_rot_part, k_unrot[..., rot_dim:]], dim=-1)

                # Expand selector keys to match heads (GQA repeat), mirroring attention.
                if self.n_kv_heads != self.heads:
                    repeat_factor = math.ceil(self.heads / self.n_kv_heads)
                    k_sel = k_sel.repeat_interleave(repeat_factor, dim=2)[:, :, : self.heads, :]
                # Transpose to (B, H, T, D)
                q_sel_t = q_sel.transpose(1, 2)
                k_sel_t = k_sel.transpose(1, 2)
                selector_mask = self._build_selector_mask(q_sel_t, k_sel_t, causal=causal)

            if static_mask is not None:
                attn_mask = static_mask
            if selector_mask is not None:
                attn_mask = (
                    selector_mask if attn_mask is None else torch.maximum(attn_mask, selector_mask)
                )

            if alibi:
                bias = _build_alibi_bias(self.heads, t, device=x.device, causal=causal)
                if attn_mask is None and causal:
                    causal_mask = torch.zeros((t, t), device=x.device)
                    causal_mask = causal_mask.masked_fill(
                        torch.triu(torch.ones(t, t, device=x.device), diagonal=1) == 1,
                        float("-inf"),
                    )
                    attn_mask = causal_mask
                attn_mask = bias if attn_mask is None else attn_mask + bias

        # Adjust for GQA if needed (manual repeat)
        if self.n_kv_heads != self.heads:
            repeat_factor = math.ceil(self.heads / self.n_kv_heads)
            k = k.repeat_interleave(repeat_factor, dim=2)[:, :, : self.heads, :]
            v = v.repeat_interleave(repeat_factor, dim=2)[:, :, : self.heads, :]

        # Transpose for SDPA: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate Gate if enabled
        gate = None
        if self.gating_pos != "none":
            # q is (B, H, T, D). gate_bias is (H, D)
            if self.gating_op == "dense":
                # gate_weight is (H, D, D)
                # Q[b,h,t,:] @ W[h,:,:] -> (B, H, T, D)
                g = torch.einsum("bhtd,hde->bhte", q, self.gate_weight)
            else:
                # gate_weight is (H, D)
                # Expand to broadcast against q: (1, H, 1, D)
                gw = self.gate_weight.unsqueeze(0).unsqueeze(2)
                g = q * gw

            gb = self.gate_bias.unsqueeze(0).unsqueeze(2)
            gate = torch.sigmoid(g + gb)

        # Optionally gate values before attention
        if gate is not None and self.gating_pos == "value":
            v = v * gate

        if linear_enabled:
            feature = str(getattr(self.cfg, "linear_feature_map", "elu") or "elu").lower()
            if feature == "elu":
                q_phi = F.elu(q.to(dtype=torch.float32)) + 1.0
                k_phi = F.elu(k.to(dtype=torch.float32)) + 1.0
            else:
                q_phi = q.to(dtype=torch.float32)
                k_phi = k.to(dtype=torch.float32)
            v_f = v.to(dtype=torch.float32)

            if causal:
                k_acc = k_phi.cumsum(dim=2)
                kv = torch.einsum("bhtd,bhtm->bhtdm", k_phi, v_f)
                kv_acc = kv.cumsum(dim=2)
            else:
                k_sum = k_phi.sum(dim=2, keepdim=True)
                kv_sum = torch.einsum("bhtd,bhtm->bhdm", k_phi, v_f).unsqueeze(2)
                k_acc = k_sum.expand(-1, -1, t, -1)
                kv_acc = kv_sum.expand(-1, -1, t, -1, -1)

            denom = torch.einsum("bhtd,bhtd->bht", q_phi, k_acc).unsqueeze(-1).clamp_min(1e-6)
            out = torch.einsum("bhtd,bhtdm->bhtm", q_phi, kv_acc) / denom
            out = out.to(dtype=q.dtype)
        else:
            if attn_mask is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        if gate is not None and self.gating_pos == "output":
            out = out * gate

        # Transpose back: (B, T, H, D)
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, t, self.heads * self.head_dim)
        out = cast(Tensor, self.c_proj(out))
        if self.dropout_p > 0.0:
            out = cast(Tensor, F.dropout(out, p=self.dropout_p, training=self.training))
        return out


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


def _build_alibi_bias(heads: int, seq_len: int, *, device: torch.device, causal: bool) -> Tensor:
    # Standard ALiBi slopes (head-dependent) with a simple power-of-two fallback.
    # Bias is negative for distant keys: -slope[h] * (i - j).
    def get_slopes(n: int) -> list[float]:
        # From the ALiBi paper reference implementation.
        import math

        def power_of_two_slopes(power: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(power) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(power)]

        if math.log2(n).is_integer():
            return power_of_two_slopes(n)
        closest_power = 2 ** int(math.floor(math.log2(n)))
        slopes = power_of_two_slopes(closest_power)
        extra = get_slopes(2 * closest_power)[0::2]
        slopes.extend(extra[: n - closest_power])
        return slopes

    slopes = torch.tensor(get_slopes(heads), device=device, dtype=torch.float32)
    pos = torch.arange(seq_len, device=device, dtype=torch.int64)
    dist = pos[:, None] - pos[None, :]
    if causal:
        dist = dist.clamp_min(0)
    bias = -slopes[:, None, None] * dist.to(dtype=torch.float32)[None, :, :]
    return cast(Tensor, bias)


class RetroModule(nn.Module):
    def __init__(self, cfg: RetroConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: Tensor) -> Tensor:
        # Approximate Retro memory as a long-horizon moving average. Interpret
        # (memory_tokens * stride) as an effective horizon (in tokens).
        horizon = max(1, int(self.cfg.memory_tokens) * max(1, int(self.cfg.stride)))
        window = min(horizon, x.shape[1])
        cumsum = torch.cumsum(x, dim=1)
        padding = torch.zeros_like(x[:, :window])
        shifted = torch.cat([padding, cumsum[:, :-window]], dim=1)
        avg = (cumsum - shifted) / max(1, window)

        agg = getattr(self.cfg, "aggregator", "gate")
        if agg == "mean":
            out = avg
        elif agg == "attention":
            scale = 1.0 / math.sqrt(max(1, x.shape[-1]))
            score = (x * avg).sum(dim=-1, keepdim=True) * scale
            out = torch.sigmoid(score) * avg
        else:  # "gate"
            out = avg
        return cast(Tensor, out * float(self.cfg.gating_weight))


class MemoryTokensModule(nn.Module):
    def __init__(self, cfg: MemoryTokensConfig, dim: int):
        super().__init__()
        self.cfg = cfg
        self.heads = int(cfg.heads)
        self.head_dim = int(cfg.head_dim)
        self.dropout_p = float(getattr(cfg, "dropout", 0.0) or 0.0)
        inner = self.heads * self.head_dim
        self.q_proj = nn.Linear(dim, inner, bias=True)
        self.o_proj = nn.Linear(inner, dim, bias=True)
        init_std = float(getattr(cfg, "init_std", 0.02) or 0.02)
        mem = torch.empty(int(cfg.tokens), 2 * inner, dtype=torch.float32)
        nn.init.normal_(mem, mean=0.0, std=init_std)
        self.mem_kv = nn.Parameter(mem)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        inner = self.heads * self.head_dim
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k_raw, v_raw = self.mem_kv.split(inner, dim=-1)
        k = k_raw.view(1, -1, self.heads, self.head_dim).expand(b, -1, -1, -1).transpose(1, 2)
        v = v_raw.view(1, -1, self.heads, self.head_dim).expand(b, -1, -1, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k.to(dtype=q.dtype), v.to(dtype=q.dtype))
        out = out.transpose(1, 2).contiguous().view(b, t, inner)
        out = cast(Tensor, self.o_proj(out))
        if self.dropout_p > 0.0:
            out = cast(Tensor, F.dropout(out, p=self.dropout_p, training=self.training))
        return cast(Tensor, out * float(getattr(self.cfg, "gating_weight", 0.0) or 0.0))


class ChunkMemoryModule(nn.Module):
    def __init__(self, cfg: ChunkMemoryConfig, dim: int):
        super().__init__()
        self.cfg = cfg
        self.heads = int(cfg.heads)
        self.head_dim = int(cfg.head_dim)
        self.dropout_p = float(getattr(cfg, "dropout", 0.0) or 0.0)
        inner = self.heads * self.head_dim
        self.q_proj = nn.Linear(dim, inner, bias=True)
        self.k_proj = nn.Linear(dim, inner, bias=True)
        self.v_proj = nn.Linear(dim, inner, bias=True)
        self.o_proj = nn.Linear(inner, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        chunk = max(1, int(self.cfg.chunk_size))
        stride = int(getattr(self.cfg, "stride", None) or chunk)
        stride = max(1, stride)
        ends = torch.arange(0, t, stride, device=x.device, dtype=torch.int64)
        if ends.numel() == 0:
            return x.new_zeros(b, t, x.size(-1))
        starts = (ends - chunk + 1).clamp_min(0)

        x_f = x.to(dtype=torch.float32)
        cumsum = torch.cumsum(x_f, dim=1)
        end_sum = cumsum.index_select(1, ends)
        prev_idx = (starts - 1).clamp_min(0)
        prev_sum = cumsum.index_select(1, prev_idx)
        prev_sum = prev_sum * (starts > 0).to(dtype=torch.float32).view(1, -1, 1)
        window_sum = end_sum - prev_sum
        lengths = (ends - starts + 1).to(dtype=torch.float32).view(1, -1, 1).clamp_min(1.0)
        summary = (window_sum / lengths).to(dtype=x.dtype)

        inner = self.heads * self.head_dim
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(summary).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(summary).view(b, -1, self.heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(t, device=x.device, dtype=torch.int64).view(t, 1)
        allowed = ends.view(1, -1) <= positions
        attn_mask = torch.where(allowed, 0.0, float("-inf")).to(dtype=torch.float32)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(b, t, inner)
        out = cast(Tensor, self.o_proj(out))
        if self.dropout_p > 0.0:
            out = cast(Tensor, F.dropout(out, p=self.dropout_p, training=self.training))
        return cast(Tensor, out * float(getattr(self.cfg, "gating_weight", 0.0) or 0.0))


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


class AssociativeMemoryModule(nn.Module):
    def __init__(self, cfg: AssociativeMemoryConfig, dim: int):
        super().__init__()
        self.cfg = cfg
        self.heads = int(cfg.heads)
        self.head_dim = int(cfg.head_dim)
        self.dropout_p = float(getattr(cfg, "dropout", 0.0) or 0.0)
        inner = self.heads * self.head_dim
        self.q_proj = nn.Linear(dim, inner, bias=True)
        self.k_proj = nn.Linear(dim, inner, bias=True)
        self.v_proj = nn.Linear(dim, inner, bias=True)
        self.o_proj = nn.Linear(inner, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)

        feature = str(getattr(self.cfg, "feature_map", "elu") or "elu").lower()
        if feature == "elu":
            q_phi = F.elu(q.to(dtype=torch.float32)) + 1.0
            k_phi = F.elu(k.to(dtype=torch.float32)) + 1.0
        else:
            q_phi = q.to(dtype=torch.float32)
            k_phi = k.to(dtype=torch.float32)
        v_f = v.to(dtype=torch.float32)

        k_acc = k_phi.cumsum(dim=2)
        kv = torch.einsum("bhtd,bhtm->bhtdm", k_phi, v_f)
        kv_acc = kv.cumsum(dim=2)
        denom = torch.einsum("bhtd,bhtd->bht", q_phi, k_acc).unsqueeze(-1).clamp_min(1e-6)
        out = torch.einsum("bhtd,bhtdm->bhtm", q_phi, kv_acc) / denom
        out = out.to(dtype=x.dtype)
        out = out.transpose(1, 2).contiguous().view(b, t, self.heads * self.head_dim)
        out = cast(Tensor, self.o_proj(out))
        if self.dropout_p > 0.0:
            out = cast(Tensor, F.dropout(out, p=self.dropout_p, training=self.training))
        return cast(Tensor, out * float(getattr(self.cfg, "gating_weight", 0.0) or 0.0))


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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def _norm_layer(norm_type: str, dim: int) -> nn.Module:
    if norm_type.lower() == "rmsnorm":
        return RMSNorm(dim)
    return nn.LayerNorm(dim)
