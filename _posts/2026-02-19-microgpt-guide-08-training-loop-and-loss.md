---
layout: post
title: "microgpt.py 가이드 (08) - 학습 루프/손실 계산: 문서 단위 next-token 학습"
date: 2026-02-19
permalink: /microgpt-guide-08-training-loop-and-loss/
author: Andrej Karpathy
categories: ['LLM 학습', '학습 루프']
tags: [Training Loop, NLL, Autoregressive, BOS, microgpt]
original_url: "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py"
excerpt: "step 루프에서 토큰/타깃 쌍을 만들고, 시퀀스 평균 손실을 계산하는 방식을 코드 그대로 설명합니다."
---

## step 루프 기본 구조 (152~169행)

각 step마다 문서 1개를 선택해 학습합니다.

```python
doc = docs[step % len(docs)]
tokens = [BOS] + ... + [BOS]
```

`step % len(docs)`로 문서를 순환하고, 전체 step 수는 `num_steps=1000`입니다.

---

## teacher forcing 방식

for pos_id in range(n):

- 입력 token = `tokens[pos_id]`
- 정답 token = `tokens[pos_id + 1]`

즉 항상 "다음 토큰"을 맞추는 autoregressive objective를 사용합니다.

---

## 손실 정의

```python
probs = softmax(logits)
loss_t = -probs[target_id].log()
loss = (1 / n) * sum(losses)
```

이것은 시퀀스 위치별 negative log likelihood 평균입니다.

- 위치별 손실을 합산
- 길이 `n`으로 정규화
- 문서 길이 차이를 완화

---

## backward 호출

```python
loss.backward()
```

`Value` 그래프 전체를 역전파해 모든 파라미터 grad를 누적합니다.

중요한 점: 매 step শেষে grad를 수동으로 0으로 초기화해야 합니다(Adam 루프에서 처리).

다음 장에서 Adam 업데이트 수식이 코드에 어떻게 들어있는지 봅니다.
