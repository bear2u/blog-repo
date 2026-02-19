---
layout: post
title: "microgpt.py 가이드 (01) - 소개/전체 구조 지도: 199줄로 GPT 전체를 담기"
date: 2026-02-19
permalink: /microgpt-guide-01-intro-and-code-map/
author: Andrej Karpathy
categories: ['LLM 학습', 'Python']
tags: [microgpt, GPT, Karpathy, Python, Code Reading]
original_url: "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py"
excerpt: "microgpt.py의 상위 구조를 먼저 잡고, 각 코드 블록이 GPT 학습 파이프라인에서 어떤 역할을 하는지 맵으로 정리합니다."
---

## 이 코드의 목적

파일 헤더(1~7행)에서 선언하는 메시지는 명확합니다.

- 순수 Python
- 의존성 최소화
- 효율보다 알고리즘 직관 우선

즉 "실전 모델 코드"보다 "GPT 알고리즘의 본질을 드러내는 학습용 레퍼런스"에 가깝습니다.

---

## 블록별 구성

1. 데이터/토크나이저 준비 (14~27행)
2. 오토그라드 엔진 `Value` (29~72행)
3. 파라미터 초기화 (74~90행)
4. 수학 유틸 + `gpt` forward (94~144행)
5. 학습 루프 + Adam (146~184행)
6. 추론 샘플링 (186~200행)

---

## 전체 실행 흐름

```text
docs -> char tokenizer -> token ids
-> gpt forward -> softmax -> NLL loss
-> backward(chain rule)
-> Adam update
-> BOS에서 시작해 autoregressive sampling
```

---

## 왜 중요한가

딥러닝 프레임워크(Pytorch/JAX)가 숨겨주는 과정을 전부 코드로 노출합니다.

- gradient accumulation
- attention scoring
- optimizer update

그래서 "동작은 되는데 원리를 모르는 상태"에서 빠져나오기에 매우 좋은 코드입니다.

다음 장에서 데이터와 토크나이저 부분부터 줄 단위로 들어갑니다.
