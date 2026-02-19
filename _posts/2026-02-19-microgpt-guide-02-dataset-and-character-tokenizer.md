---
layout: post
title: "microgpt.py 가이드 (02) - 데이터셋/문자 토크나이저: BOS 기반 최소 설계"
date: 2026-02-19
permalink: /microgpt-guide-02-dataset-and-character-tokenizer/
author: Andrej Karpathy
categories: ['LLM 학습', '데이터 전처리']
tags: [Tokenizer, Character-level, BOS, Dataset, microgpt]
original_url: "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py"
excerpt: "데이터 로딩과 문자 단위 토크나이저가 어떻게 구현되어 있고, 왜 BOS 하나로 시퀀스 시작/종료를 처리하는지 설명합니다."
---

## 입력 데이터 준비 (14~21행)

코드는 `input.txt`가 없으면 names.txt를 다운로드합니다.

```python
if not os.path.exists('input.txt'):
    ...
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
```

핵심 포인트:

- 빈 줄 제거
- 한 줄 = 한 문서(여기서는 이름)
- 문서 순서를 섞어 순차 편향 완화

---

## 문자 집합 기반 토크나이저 (23~27행)

```python
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
```

설계 의도:

- 문자 vocabulary를 직접 만들고 정렬해 id를 고정
- BOS를 vocabulary 마지막 id로 예약
- BOS 하나로 시작/종료 구분 역할까지 수행

---

## 문자 단위 접근의 장단점

장점:

- 구현 단순
- tokenizer 의존성 없음
- 데이터가 짧을 때 빠르게 학습 가능

단점:

- 긴 문장에서 시퀀스 길이 급증
- subword 대비 표현 효율 낮음

---

## 학습 시 실제 토큰 시퀀스

학습 루프(156~158행)에서 문서를 이렇게 변환합니다.

```python
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
n = min(block_size, len(tokens) - 1)
```

즉 `BOS name BOS` 구조의 next-token prediction을 학습합니다.

다음 장에서 이 토큰을 학습 가능하게 만드는 `Value` 오토그라드 엔진을 분석합니다.
