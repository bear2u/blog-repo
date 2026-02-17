---
layout: post
title: "MLX LLM Tutorial 가이드 (05) - 데이터/토크나이저: simple_llm.py의 입력 파이프라인"
date: 2026-02-17
permalink: /mlx-llm-tutorial-guide-05-simple-llm-data-and-tokenizer/
author: ddttom
categories: ['LLM 학습', '데이터 전처리']
tags: [simple_llm.py, Tokenizer, Shakespeare, Dataset, MLX]
original_url: "https://github.com/ddttom/mlx-llm-tutorial/blob/main/code/simple_llm.py"
excerpt: "tinyshakespeare 다운로드부터 문자 단위 vocab 구성, 입력/타깃 시퀀스 생성까지 학습 데이터 파이프라인을 정리합니다."
---

## 데이터 소스

`simple_llm.py`는 학습용 텍스트를 자동 다운로드합니다.

- URL: `karpathy/char-rnn`의 tinyshakespeare 데이터
- 함수: `download_dataset(url, filename)`

```python
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = download_dataset(url, "shakespeare.txt")
```

---

## 문자 단위 토크나이저

이 스크립트의 토크나이징은 문자(character) 단위입니다.

- `create_tokenizer(text, vocab_size=256)`
- 고유 문자 집합을 만들고 index 매핑 생성
- `char_to_idx`, `idx_to_char`를 저장

장점은 단순함이고, 단점은 시퀀스가 길어지기 쉽다는 점입니다.

---

## 입력/정답 쌍 생성

`prepare_data(...)`에서 학습 샘플을 만듭니다.

1. 텍스트 -> 인덱스 배열 변환
2. `seq_length` 단위로 reshape
3. `inputs = x[:, :-1]`, `targets = x[:, 1:]`

즉, 한 칸 오른쪽으로 민 next-character prediction 구성입니다.

---

## 저장되는 아티팩트

학습 후에는 아래 파일이 생성됩니다.

- `simple_llm_model.npz`: 모델 파라미터
- `tokenizer.json`: 문자 사전

이 둘이 다음 장의 학습/생성과 웹 인터페이스 연결에 핵심 입력이 됩니다.

