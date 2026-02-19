---
layout: post
title: "microgpt.py 가이드 (09) - Adam 업데이트/스케줄: 수식과 코드 일치시키기"
date: 2026-02-19
permalink: /microgpt-guide-09-adam-update-and-schedule/
author: Andrej Karpathy
categories: ['LLM 학습', '최적화']
tags: [Adam, Optimizer, Learning Rate Decay, Bias Correction, microgpt]
original_url: "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py"
excerpt: "microgpt.py의 Adam 구현을 수식 관점에서 풀고, lr 선형 감쇠가 어떤 효과를 가지는지 정리합니다."
---

## 버퍼 정의 (147~149행)

- `m`: 1차 모멘트(평균)
- `v`: 2차 모멘트(분산 추정)

파라미터 개수와 동일 길이의 float 리스트로 관리합니다.

---

## 학습률 스케줄 (175행)

```python
lr_t = learning_rate * (1 - step / num_steps)
```

선형 decay로 step이 진행될수록 update 폭을 줄입니다.

- 초반: 빠른 탐색
- 후반: 안정적 수렴

---

## Adam 업데이트 (176~182행)

코드가 표준 Adam 수식을 거의 그대로 구현합니다.

1. `m = beta1*m + (1-beta1)*g`
2. `v = beta2*v + (1-beta2)*g^2`
3. bias correction: `m_hat`, `v_hat`
4. `p -= lr * m_hat / (sqrt(v_hat)+eps)`

---

## grad reset의 의미

```python
p.grad = 0
```

`Value` 기반 구현은 프레임워크가 자동 zero_grad를 해주지 않으므로,
다음 step에서 gradient 누적 오류를 막기 위해 반드시 필요합니다.

---

## 튜닝 포인트

- `learning_rate`, `beta1`, `beta2`
- `num_steps`
- `n_embd`, `n_layer`, `block_size`

작은 데이터셋(names)에서는 과적합이 매우 빠르게 나타날 수 있어,
loss 감소만 보고 성능을 판단하면 안 됩니다.

다음 장에서 최종 추론 루프와 확장 아이디어를 정리합니다.
