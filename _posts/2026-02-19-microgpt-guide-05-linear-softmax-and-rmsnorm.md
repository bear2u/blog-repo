---
layout: post
title: "microgpt.py 가이드 (05) - linear/softmax/rmsnorm: forward 최소 유틸"
date: 2026-02-19
permalink: /microgpt-guide-05-linear-softmax-and-rmsnorm/
author: Andrej Karpathy
categories: ['LLM 학습', '수학/함수']
tags: [linear, softmax, rmsnorm, microgpt, forward pass]
original_url: "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py"
excerpt: "gpt 함수 직전에 정의된 linear, softmax, rmsnorm 세 함수의 수학적 의미와 수치 안정성 포인트를 설명합니다."
---

## linear(x, w)

```python
return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

행렬곱 `y = W x`를 순수 Python 반복문으로 구현합니다.

- `wo`: W의 한 행
- 출력 차원마다 dot product 계산

---

## softmax(logits)

```python
max_val = max(val.data for val in logits)
exps = [(val - max_val).exp() for val in logits]
```

`max`를 빼는 이유는 오버플로우 방지입니다.

- 안정화된 exp 계산
- 합으로 나눠 확률 벡터 생성

이 구현 덕분에 후속 `-log p(target)` 손실이 안전해집니다.

---

## rmsnorm(x)

```python
ms = sum(xi * xi for xi in x) / len(x)
scale = (ms + 1e-5) ** -0.5
```

RMSNorm은 LayerNorm보다 단순하며, 평균 제거 없이 magnitude만 정규화합니다.

microgpt에서는 GPT-2 스타일을 단순화하면서도 학습 안정성을 유지하기 위해 채택했습니다.

---

## 이 세 함수의 역할 정리

- `linear`: 표현 변환
- `softmax`: 확률화
- `rmsnorm`: 스케일 안정화

다음 장에서 이 함수들이 실제 attention 블록에서 어떻게 결합되는지 봅니다.
