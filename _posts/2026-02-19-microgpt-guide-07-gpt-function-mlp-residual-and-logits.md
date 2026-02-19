---
layout: post
title: "microgpt.py 가이드 (07) - gpt 함수 (MLP/Residual/Logits): 토큰 분포 만들기"
date: 2026-02-19
permalink: /microgpt-guide-07-gpt-function-mlp-residual-and-logits/
author: Andrej Karpathy
categories: ['LLM 학습', '트랜스포머']
tags: [MLP, Residual, ReLU, lm_head, microgpt]
original_url: "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py"
excerpt: "Attention 다음 MLP 블록과 최종 lm_head를 통해 다음 토큰 로짓이 만들어지는 과정을 해설합니다."
---

## MLP 블록 (136~141행)

패턴은 전형적인 Transformer FFN입니다.

1. RMSNorm
2. `fc1` 확장(4 * n_embd)
3. ReLU
4. `fc2` 축소(n_embd)
5. residual add

```python
x = linear(x, mlp_fc1)
x = [xi.relu() for xi in x]
x = linear(x, mlp_fc2)
x = [a + b for a, b in zip(x, x_residual)]
```

---

## 왜 ReLU인가

주석(93행)에도 명시되듯 GeLU 대신 ReLU를 사용해 구현을 단순화합니다.

- 미분식 단순
- 오토그라드 구현 부담 감소
- 학습 데모에는 충분한 표현력

---

## 최종 로짓 (143행)

```python
logits = linear(x, state_dict['lm_head'])
```

`lm_head`는 hidden state를 vocabulary 크기로 사상합니다.

- 출력 크기 = `vocab_size`
- softmax 이전 raw score
- 이후 target token의 확률을 손실로 사용

---

## gpt 함수의 출력 의미

`gpt(...)`는 확률이 아니라 로짓을 반환합니다.

- 학습 시: softmax -> NLL
- 추론 시: temperature scaling 후 sampling

다음 장에서 학습 루프에서 이 출력이 손실로 연결되는 흐름을 봅니다.
