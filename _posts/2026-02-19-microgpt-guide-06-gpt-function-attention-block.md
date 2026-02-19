---
layout: post
title: "microgpt.py 가이드 (06) - gpt 함수 (어텐션 블록): QKV와 누적 KV 캐시"
date: 2026-02-19
permalink: /microgpt-guide-06-gpt-function-attention-block/
author: Andrej Karpathy
categories: ['LLM 학습', '트랜스포머']
tags: [Attention, QKV, Multi-head, KV Cache, microgpt]
original_url: "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py"
excerpt: "gpt 함수의 attention 부분을 중심으로 Q/K/V 계산, head 분할, causal 시점 누적을 상세히 설명합니다."
---

## 입력 임베딩 결합 (109~113행)

- `wte[token_id]`: 토큰 임베딩
- `wpe[pos_id]`: 위치 임베딩
- 둘을 더한 뒤 RMSNorm

이후 레이어 반복으로 들어갑니다.

---

## Q/K/V 계산 (118~122행)

```python
q = linear(x, ...attn_wq)
k = linear(x, ...attn_wk)
v = linear(x, ...attn_wv)
keys[li].append(k)
values[li].append(v)
```

핵심은 `keys/values`를 시점마다 append해서 causal context를 축적한다는 점입니다.

---

## head 단위 분리 (124~132행)

- `hs = h * head_dim`
- `q_h`, `k_h`, `v_h` 슬라이스
- 각 head에서 attention score 계산

score 식:

`score_t = (q · k_t) / sqrt(head_dim)`

그 뒤 softmax로 가중치화하고 `v_h`를 가중합합니다.

---

## attention 출력 결합

head별 출력 `head_out`을 `x_attn`으로 concat한 다음,
`attn_wo` 선형층으로 projection합니다.

```python
x = linear(x_attn, attn_wo)
x = x + residual
```

즉 standard MHA + residual 구조를 정확히 따릅니다.

---

## 구현 포인트

- 마스크 행렬을 별도로 만들지 않고, "지금까지 append된 k/v만 사용"하는 방식으로 causal 제약을 만족합니다.
- 추론 시에도 동일 로직이 자연스럽게 동작합니다(KV 누적).

다음 장에서 attention 뒤의 MLP/잔차/최종 로짓 경로를 완성합니다.
