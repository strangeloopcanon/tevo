"""Memory modules: Retro, MemoryTokens, ChunkMemory, LookupMemory, AssociativeMemory."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from .dsl import (
    AssociativeMemoryConfig,
    ChunkMemoryConfig,
    LookupMemoryConfig,
    MemoryTokensConfig,
    RetroConfig,
)


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


class LookupMemoryModule(nn.Module):
    """Conditional lookup from a learned key/value table."""

    def __init__(self, cfg: LookupMemoryConfig, dim: int):
        super().__init__()
        self.cfg = cfg
        self.entries = int(cfg.entries)
        self.topk = max(1, min(int(getattr(cfg, "topk", 4) or 4), self.entries))
        self.key_dim = int(getattr(cfg, "key_dim", None) or dim)
        self.value_dim = int(getattr(cfg, "value_dim", None) or dim)
        self.temperature = float(getattr(cfg, "temperature", 1.0) or 1.0)
        self.dropout_p = float(getattr(cfg, "dropout", 0.0) or 0.0)
        self.chunk_size = max(1, int(getattr(cfg, "chunk_size", 1024) or 1024))
        self.lookup_device = str(getattr(cfg, "lookup_device", "model") or "model").lower()

        init_std = 0.02
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.keys = nn.Parameter(torch.randn(self.entries, self.key_dim) * init_std)
        self.values = nn.Parameter(torch.randn(self.entries, self.value_dim) * init_std)
        self.out_proj: nn.Module
        if self.value_dim == dim:
            self.out_proj = nn.Identity()
        else:
            self.out_proj = nn.Linear(self.value_dim, dim, bias=False)

    def _apply(self, fn):
        super()._apply(fn)
        # Optional offload: keep the lookup table on CPU even when the rest of
        # the model moves devices. This makes placement a searchable knob.
        if self.lookup_device == "cpu":
            with torch.no_grad():
                self.keys.data = self.keys.data.cpu()
                self.values.data = self.values.data.cpu()
                if self.keys.grad is not None:
                    self.keys.grad.data = self.keys.grad.data.cpu()
                if self.values.grad is not None:
                    self.values.grad.data = self.values.grad.data.cpu()
        return self

    def _topk_chunked(self, q: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        """Chunked top-k retrieval over a key table.

        Args:
            q: (B, T, Dk)
            keys: (N, Dk)
        Returns:
            (scores, idx): both (B, T, K)
        """
        b, t, d = q.shape
        n = int(keys.shape[0])
        k = max(1, min(self.topk, n))
        chunk = max(1, min(self.chunk_size, n))
        q2 = q.reshape(-1, d)
        best_scores: Tensor | None = None
        best_idx: Tensor | None = None
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            keys_chunk = keys[start:end]
            scores = (q2 @ keys_chunk.t()).view(b, t, end - start)
            top_scores, top_idx = torch.topk(scores, k=min(k, end - start), dim=-1)
            top_idx = top_idx + start
            if best_scores is None or best_idx is None:
                best_scores = top_scores
                best_idx = top_idx
                continue
            cand_scores = torch.cat([best_scores, top_scores], dim=-1)
            cand_idx = torch.cat([best_idx, top_idx], dim=-1)
            best_scores, chosen = torch.topk(cand_scores, k=k, dim=-1)
            best_idx = cand_idx.gather(-1, chosen)
        if best_scores is None or best_idx is None:
            return q.new_zeros(b, t, 1), q.new_zeros(b, t, 1, dtype=torch.int64)
        return best_scores, best_idx

    def forward(self, x: Tensor) -> Tensor:
        b, t, dim = x.shape
        if self.entries <= 0:
            return x.new_zeros(b, t, dim)

        q = self.q_proj(x)
        table_device = torch.device("cpu") if self.lookup_device == "cpu" else x.device

        q_f = q.to(device=table_device, dtype=torch.float32)
        keys_f = self.keys.to(device=table_device, dtype=torch.float32)
        values_f = self.values.to(device=table_device, dtype=torch.float32)

        scores, idx = self._topk_chunked(q_f, keys_f)
        temp = self.temperature if self.temperature > 0.0 else 1.0
        weights = torch.softmax(scores / temp, dim=-1)
        selected = values_f[idx]  # (B, T, K, value_dim)
        out = (weights.unsqueeze(-1) * selected).sum(dim=-2)
        out = out.to(device=x.device, dtype=x.dtype)
        out = cast(Tensor, self.out_proj(out))
        if self.dropout_p > 0.0:
            out = cast(Tensor, F.dropout(out, p=self.dropout_p, training=self.training))
        return cast(Tensor, out * float(getattr(self.cfg, "gating_weight", 0.0) or 0.0))


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
