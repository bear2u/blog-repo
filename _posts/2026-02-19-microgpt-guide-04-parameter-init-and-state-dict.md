---
layout: post
title: "microgpt.py 가이드 (04) - 파라미터 초기화/state_dict: tiny GPT 파라미터 해부"
date: 2026-02-19
permalink: /microgpt-guide-04-parameter-init-and-state-dict/
author: Andrej Karpathy
categories: ['LLM 학습', '모델 구조']
tags: [state_dict, Parameter Initialization, Transformer, microgpt]
original_url: "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py"
excerpt: "n_layer, n_embd, n_head 설정이 파라미터 텐서(리스트) 형태로 어떻게 펼쳐지는지 설명합니다."
---

## 하이퍼파라미터 (75~79행)

- `n_layer = 1`
- `n_embd = 16`
- `block_size = 16`
- `n_head = 4`
- `head_dim = 4`

학습 데모 목적이라 매우 작은 모델입니다.

---

## matrix 생성기 (80행)

```python
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) ... ]]
```

각 weight는 `Value` 스칼라로 구성됩니다.

즉 일반 프레임워크의 2D tensor 하나를, 여기서는 `list[list[Value]]`로 표현합니다.

---

## state_dict 키 구조 (81~88행)

공통:

- `wte`: token embedding
- `wpe`: position embedding
- `lm_head`: vocab projection

레이어별:

- `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`
- `mlp_fc1`, `mlp_fc2`

이 키 이름이 곧 모델 구조 문서 역할을 합니다.

---

## 파라미터 flatten (89행)

```python
params = [p for mat in state_dict.values() for row in mat for p in row]
```

Adam 업데이트를 단일 루프로 돌리기 위해 모든 파라미터를 1차원 리스트로 평탄화합니다.

다음 장에서 forward 계산을 구성하는 수학 유틸 3개를 먼저 분석합니다.
