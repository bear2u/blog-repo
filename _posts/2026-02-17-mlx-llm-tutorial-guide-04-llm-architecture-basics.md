---
layout: post
title: "MLX LLM Tutorial 가이드 (04) - LLM 아키텍처 기초: Transformer를 코드로 보기 전에 잡아야 할 것"
date: 2026-02-17
permalink: /mlx-llm-tutorial-guide-04-llm-architecture-basics/
author: ddttom
categories: ['LLM 학습', '모델 아키텍처']
tags: [Transformer, Attention, Tokenization, Decoder Only, Embedding]
original_url: "https://github.com/ddttom/mlx-llm-tutorial/blob/main/public/docs/llm-architecture.md"
excerpt: "토크나이징부터 Self-Attention, FFN, Residual까지 Transformer 핵심 블록을 코드 관점에서 빠르게 정리합니다."
---

## Transformer 핵심 블록

문서가 제시하는 기본 축은 다음과 같습니다.

1. Tokenization
2. Embedding + Positional Embedding
3. Self-Attention
4. Feed-Forward Network
5. LayerNorm + Residual
6. Output Projection

`simple_llm.py`와 `finetune_llm.py` 모두 이 구조를 기반으로 구현됩니다.

---

## 중요한 수식 2개

문서에서 가장 중요한 식은 사실 두 개입니다.

```python
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

```python
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

실습 코드에서 변수명만 달라질 뿐, 계산 흐름은 동일합니다.

---

## 디코더 전용 모델 관점

이 프로젝트의 간단한 LM은 사실상 디코더 중심입니다.

- 입력 시퀀스에서 다음 토큰 예측
- causal mask(하삼각 마스크) 사용
- 생성 시 이전 토큰을 조건으로 순차 샘플링

즉, "대화형 LLM의 최소 단위"를 직접 만드는 학습 구성입니다.

---

## 이 장에서 얻어야 하는 것

코드를 보기 전에 아래를 이해하면 이후 속도가 빨라집니다.

- attention은 "토큰 간 중요도 가중합"
- residual/layernorm은 학습 안정화 장치
- decoder LM은 "next token prediction" 문제

다음 장에서 `simple_llm.py`의 데이터와 토크나이저부터 실제로 뜯어봅니다.

