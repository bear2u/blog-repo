---
layout: post
title: "MLX LLM Tutorial 가이드 (06) - 모델/학습 루프: SimpleTransformer 학습 흐름"
date: 2026-02-17
permalink: /mlx-llm-tutorial-guide-06-simple-llm-model-and-training/
author: ddttom
categories: ['LLM 학습', '모델 학습']
tags: [SimpleTransformer, MultiHeadAttention, Adam, CrossEntropy, Training Loop]
original_url: "https://github.com/ddttom/mlx-llm-tutorial/blob/main/code/simple_llm.py"
excerpt: "simple_llm.py의 모델 클래스와 학습 루프를 기준으로 attention 블록부터 loss 계산까지 한 번에 정리합니다."
---

## 모델 구조

`SimpleTransformer`는 전형적인 decoder-style 교육용 구조입니다.

- token embedding
- position embedding
- transformer blocks(반복)
- final layer norm
- lm head

마스크는 `mx.tril(mx.ones((seq_len, seq_len)))`로 만들어 causal 제약을 줍니다.

---

## 블록 내부

핵심 모듈은 세 가지입니다.

1. `MultiHeadAttention`
2. `FeedForward`
3. `TransformerBlock`(residual + layer norm 포함)

학습자 입장에서는 "Attention -> FFN -> Residual" 패턴을 코드에서 직접 보는 것이 가장 큰 수확입니다.

---

## 손실 함수와 옵티마이저

`train_model(...)`의 손실 계산은 다음 흐름입니다.

```python
logits = model(x)
logits = logits.reshape(-1, logits.shape[-1])
y = y.reshape(-1)
loss = mx.mean(mx.losses.cross_entropy(logits, y))
```

옵티마이저는 `optim.Adam(learning_rate=lr)`를 사용하고,
`mx.value_and_grad(loss_fn)`로 그라디언트를 계산합니다.

---

## 실행 진입점

기본 실행은 학습 모드입니다.

```bash
python code/simple_llm.py
```

훈련 완료 후 모델/토크나이저를 저장하므로, 생성 모드는 다음 장처럼 분리 실행할 수 있습니다.

