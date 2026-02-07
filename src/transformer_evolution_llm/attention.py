"""Attention primitives: RoPE, multi-head attention, ALiBi, activations, norms."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from .dsl import AttentionConfig

# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


def _swiglu(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return F.silu(x1) * x2


def _relu_squared(x: Tensor) -> Tensor:
    return F.relu(x).square()


ActivationLookup: dict[str, Callable[[Tensor], Tensor]] = {
    "relu": F.relu,
    "relu_squared": _relu_squared,
    "gelu": F.gelu,
    "silu": F.silu,
    "swiglu": _swiglu,
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Rotary Positional Encoding
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ALiBi bias
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Multi-head Self-Attention
# ---------------------------------------------------------------------------


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

        # QK normalization from softmax config (applied per-head after projection)
        softmax_cfg = getattr(cfg, "softmax", None)
        self._qk_norm_type = "none"
        self._softcap: float | None = None
        self._qk_scale_override: float | None = None
        if softmax_cfg is not None:
            self._qk_norm_type = str(getattr(softmax_cfg, "qk_norm", "none") or "none")
            self._softcap = getattr(softmax_cfg, "softcap", None)
            raw_scale = getattr(softmax_cfg, "qk_scale", None)
            if raw_scale is not None:
                if isinstance(raw_scale, str) and raw_scale == "none":
                    self._qk_scale_override = None
                else:
                    self._qk_scale_override = float(raw_scale)
        self.q_norm: nn.Module
        self.k_norm: nn.Module
        if self._qk_norm_type == "rms":
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        elif self._qk_norm_type == "layer":
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Value GLU: gate the value path with a learned projection
        self._value_glu = bool(getattr(cfg, "value_glu", False) or False)
        if self._value_glu:
            # V gate: same shape as V, projects from hidden dim
            self.v_gate_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)

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

        # Apply per-head QK norm (before RoPE, after projection)
        if self._qk_norm_type != "none":
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Value GLU: sigmoid-gate the value path
        if self._value_glu:
            v_gate = torch.sigmoid(self.v_gate_proj(x).view(b, t, self.n_kv_heads, self.head_dim))
            v = v * v_gate

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

        # Bridge: if stencil is present and low-level sparsity isn't set, use stencil
        stencil = getattr(self.cfg, "stencil", None)
        if stencil is not None and (sparsity == "none" or sparsity is None) and sw is None:
            stencil_kind = str(getattr(stencil, "kind", "full"))
            kind_map = {
                "local": "sliding",
                "dilated": "dilated",
                "block": "block",
                "ring": "block",
                "sliding": "sliding",
                "hybrid": "local_global",
            }
            if stencil_kind in kind_map:
                sparsity = kind_map[stencil_kind]
                stencil_win = getattr(stencil, "window", None)
                if stencil_win is not None:
                    sw = stencil_win
                    # Ensure cfg.sw is available for branch code that reads it
                    object.__setattr__(self.cfg, "sw", sw)
                # Map stencil block/stride/dilation/globals to cfg attrs for forward
                if getattr(stencil, "block", None) and not getattr(self.cfg, "block_size", None):
                    object.__setattr__(self.cfg, "block_size", stencil.block)
                if getattr(stencil, "stride", None) and not getattr(self.cfg, "block_stride", None):
                    object.__setattr__(self.cfg, "block_stride", stencil.stride)
                if getattr(stencil, "dilation", None) and not getattr(self.cfg, "dilation", None):
                    object.__setattr__(self.cfg, "dilation", stencil.dilation)
                if getattr(stencil, "globals", None) and not getattr(
                    self.cfg, "global_stride", None
                ):
                    object.__setattr__(self.cfg, "global_stride", stencil.globals)

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
            # Determine QK scale: override from softmax config, or default 1/sqrt(d)
            effective_scale = (
                self._qk_scale_override
                if self._qk_scale_override is not None
                else 1.0 / math.sqrt(self.head_dim)
            )
            if self._softcap is not None:
                # Manual attention with logit soft-capping (Gemma 2 style)
                scores = torch.matmul(q, k.transpose(-1, -2)) * effective_scale
                # Soft cap: tanh(scores / cap) * cap
                scores = torch.tanh(scores / self._softcap) * self._softcap
                if attn_mask is not None:
                    scores = scores + attn_mask
                elif causal:
                    causal_mask = torch.triu(
                        torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
                    )
                    scores = scores.masked_fill(causal_mask, float("-inf"))
                weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
                if self.dropout_p > 0.0 and self.training:
                    weights = F.dropout(weights, p=self.dropout_p)
                out = torch.matmul(weights, v)
            elif self._qk_scale_override is not None:
                # Use SDPA with custom scale
                if attn_mask is None:
                    out = F.scaled_dot_product_attention(
                        q, k, v, is_causal=causal, scale=effective_scale
                    )
                else:
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, scale=effective_scale
                    )
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
