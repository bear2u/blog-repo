---
layout: post
title: "MiniMind 완벽 가이드 (2) - 모델 아키텍처"
date: 2025-02-04
permalink: /minimind-guide-02-architecture/
author: jingyaogong
categories: [LLM 학습, MiniMind]
tags: [MiniMind, Transformer, RoPE, RMSNorm, MoE, Architecture]
original_url: "https://github.com/jingyaogong/minimind"
excerpt: "MiniMind의 Transformer 아키텍처, RoPE, RMSNorm, MoE 구현을 분석합니다."
---

## 아키텍처 개요

MiniMind는 **Llama 스타일의 디코더 전용 Transformer** 아키텍처를 사용합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MiniMind Architecture                         │
│                                                                  │
│   Input Tokens ──▶ Embedding ──▶ [Transformer Blocks] ──▶ LM Head │
│                                                                  │
│   Transformer Block:                                            │
│   ┌─────────────────────────────────────────────────┐           │
│   │  Input                                           │           │
│   │    │                                             │           │
│   │    ▼                                             │           │
│   │  RMSNorm ──▶ Self-Attention ──▶ + (Residual)    │           │
│   │                                   │              │           │
│   │                                   ▼              │           │
│   │  RMSNorm ──▶ FFN/MoE ──▶ + (Residual)           │           │
│   │                                   │              │           │
│   │                                   ▼              │           │
│   │  Output                                          │           │
│   └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 모델 설정

```python
# model/model_minimind.py

@dataclass
class MiniMindConfig:
    dim: int = 512                    # 히든 차원
    n_layers: int = 8                 # 레이어 수
    n_heads: int = 8                  # 어텐션 헤드 수
    n_kv_heads: int = 2               # KV 헤드 수 (GQA)
    vocab_size: int = 6400            # 어휘 크기
    hidden_dim: int = None            # FFN 히든 차원
    multiple_of: int = 64             # FFN 차원 배수
    norm_eps: float = 1e-5            # 정규화 엡실론
    max_seq_len: int = 8192           # 최대 시퀀스 길이
    dropout: float = 0.0              # 드롭아웃
    use_moe: bool = False             # MoE 사용 여부
    n_experts: int = 8                # 전문가 수 (MoE)
    n_experts_per_tok: int = 2        # 토큰당 활성 전문가
```

---

## RMSNorm

LayerNorm 대신 **RMSNorm**을 사용하여 효율성을 높입니다.

```python
class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # RMS 계산: sqrt(mean(x^2))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

### LayerNorm vs RMSNorm

| 특성 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 평균 계산 | O | X |
| 연산량 | 더 많음 | 더 적음 |
| 성능 | 비슷 | 비슷 |
| 채택 | GPT | Llama, MiniMind |

---

## Rotary Position Embedding (RoPE)

위치 정보를 **회전 행렬**로 인코딩합니다.

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """RoPE를 위한 주파수 미리 계산"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # 복소수로 변환 (cos + i*sin)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """Query와 Key에 RoPE 적용"""
    # 복소수로 변환
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 회전 적용
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

### RoPE의 장점

- **상대적 위치** 인코딩
- **길이 일반화** 가능 (YaRN 등 확장 기법)
- **효율적** 계산

---

## Self-Attention

**Grouped Query Attention (GQA)**를 지원합니다.

```python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # KV 반복 횟수

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        bsz, seqlen, _ = x.shape

        # Q, K, V 계산
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # RoPE 적용
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # KV 캐시 처리 (추론 시)
        if kv_cache is not None:
            xk = torch.cat([kv_cache[0], xk], dim=1)
            xv = torch.cat([kv_cache[1], xv], dim=1)
            kv_cache = (xk, xv)

        # GQA: KV 헤드 반복
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Attention 계산
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), kv_cache


def repeat_kv(x, n_rep):
    """GQA를 위해 KV 헤드 반복"""
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
```

---

## Feed-Forward Network

**SwiGLU** 활성화 함수를 사용합니다.

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        hidden_dim = config.hidden_dim or int(8 * config.dim / 3)
        # multiple_of로 반올림
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)  # Down
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)  # Up

    def forward(self, x):
        # SwiGLU: swish(xW1) * xW3
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

## Mixture of Experts (MoE)

선택적으로 **MoE 레이어**를 사용할 수 있습니다.

```python
class MoEGate(nn.Module):
    """Top-K 라우팅 게이트"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_experts_per_tok = config.n_experts_per_tok
        self.gate = nn.Linear(config.dim, config.n_experts, bias=False)

    def forward(self, x):
        # 라우팅 점수 계산
        logits = self.gate(x)

        # Top-K 선택
        weights, indices = torch.topk(
            F.softmax(logits, dim=-1),
            self.n_experts_per_tok,
            dim=-1
        )

        # 가중치 정규화
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights, indices


class MoE(nn.Module):
    """Mixture of Experts 레이어"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.gate = MoEGate(config)
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_experts)
        ])

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)

        weights, indices = self.gate(x_flat)

        # 각 전문가 출력 수집
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # 이 전문가에 라우팅된 토큰 찾기
            mask = (indices == i).any(dim=-1)
            if mask.sum() == 0:
                continue

            # 전문가 처리
            expert_input = x_flat[mask]
            expert_output = expert(expert_input)

            # 가중치 적용
            expert_weights = weights[mask]
            expert_indices = indices[mask]
            weight_mask = (expert_indices == i).float()
            final_weight = (expert_weights * weight_mask).sum(dim=-1, keepdim=True)

            output[mask] += expert_output * final_weight

        return output.view(bsz, seqlen, dim)
```

---

## Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)

        if config.use_moe and layer_id % 2 == 1:  # 홀수 레이어만 MoE
            self.feed_forward = MoE(config)
        else:
            self.feed_forward = FeedForward(config)

        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x, freqs_cis, mask=None, kv_cache=None):
        # Self-Attention + Residual
        h, kv_cache = self.attention(
            self.attention_norm(x), freqs_cis, mask, kv_cache
        )
        x = x + h

        # FFN/MoE + Residual
        x = x + self.feed_forward(self.ffn_norm(x))

        return x, kv_cache
```

---

## 전체 모델

```python
class MiniMind(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(i, config) for i in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # RoPE 주파수 미리 계산
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2
        )

    def forward(self, tokens, targets=None):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        freqs_cis = self.freqs_cis[:seqlen].to(h.device)
        mask = self._create_causal_mask(seqlen, h.device)

        for layer in self.layers:
            h, _ = layer(h, freqs_cis, mask)

        h = self.norm(h)
        logits = self.output(h)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )
            return logits, loss

        return logits

    def _create_causal_mask(self, seqlen, device):
        mask = torch.full((seqlen, seqlen), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask
```

---

*다음 글에서는 Tokenizer 훈련을 살펴봅니다.*
