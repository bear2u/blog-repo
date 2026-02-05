---
layout: post
title: "MiniMind 완벽 가이드 (3) - Tokenizer 훈련"
date: 2025-02-04
permalink: /minimind-guide-03-tokenizer/
author: jingyaogong
categories: [AI]
tags: [MiniMind, Tokenizer, BPE, Vocabulary, Training]
original_url: "https://github.com/jingyaogong/minimind"
excerpt: "MiniMind의 BPE 토크나이저 훈련 과정을 분석합니다."
---

## Tokenizer 개요

토크나이저는 텍스트를 **토큰 ID**로 변환하는 핵심 컴포넌트입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tokenization Process                          │
│                                                                  │
│   "안녕하세요" ──▶ Tokenizer ──▶ [128, 456, 789, 234]           │
│                                                                  │
│   [128, 456, 789, 234] ──▶ Detokenizer ──▶ "안녕하세요"         │
└─────────────────────────────────────────────────────────────────┘
```

---

## MiniMind 토크나이저 특징

| 특성 | 값 |
|------|-----|
| **알고리즘** | BPE (Byte Pair Encoding) |
| **어휘 크기** | 6,400 |
| **특수 토큰** | `<|im_start|>`, `<|im_end|>` |
| **언어** | 중국어 + 영어 |

---

## BPE 알고리즘

### 원리

1. 문자 단위로 시작
2. 가장 빈번한 연속 쌍을 찾아 병합
3. 원하는 어휘 크기까지 반복

```python
# BPE 예시
"low lower lowest"

# 초기: ['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ...]

# 1단계: 'l' + 'o' -> 'lo' (가장 빈번한 쌍)
# 2단계: 'lo' + 'w' -> 'low'
# 3단계: 'low' + 'e' -> 'lowe'
# ...
```

---

## 토크나이저 훈련 코드

```python
# trainer/train_tokenizer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(data_path: str, vocab_size: int = 6400):
    # BPE 토크나이저 초기화
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # 특수 토큰 정의
    special_tokens = [
        "<unk>",           # 알 수 없는 토큰
        "<s>",             # 시작 (deprecated)
        "</s>",            # 종료 (deprecated)
        "<pad>",           # 패딩
        "<|im_start|>",    # 메시지 시작
        "<|im_end|>",      # 메시지 종료
    ]

    # 트레이너 설정
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,    # 최소 등장 빈도
        show_progress=True,
    )

    # 훈련 데이터 로드
    files = [data_path]

    # 훈련 실행
    tokenizer.train(files, trainer)

    # 저장
    tokenizer.save("model/tokenizer.json")

    return tokenizer
```

---

## 훈련 데이터 준비

### 데이터 형식

```text
# train_corpus.txt
안녕하세요, 저는 MiniMind입니다.
Hello, I am MiniMind.
这是一个测试句子。
...
```

### 데이터 수집

```python
def prepare_tokenizer_data(datasets: list, output_path: str):
    """여러 데이터셋에서 텍스트 추출"""
    texts = []

    for ds in datasets:
        for item in ds:
            if 'text' in item:
                texts.append(item['text'])
            elif 'content' in item:
                texts.append(item['content'])

    # 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
```

---

## 토크나이저 설정

### tokenizer_config.json

{% raw %}
```json
{
  "add_bos_token": false,
  "add_eos_token": false,
  "bos_token": "<|im_start|>",
  "eos_token": "<|im_end|>",
  "pad_token": "<pad>",
  "unk_token": "<unk>",
  "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
  "model_max_length": 8192,
  "tokenizer_class": "PreTrainedTokenizerFast"
}
```
{% endraw %}

---

## 토크나이저 사용

```python
from transformers import AutoTokenizer

# 로드
tokenizer = AutoTokenizer.from_pretrained("model/")

# 인코딩
text = "안녕하세요, MiniMind입니다."
tokens = tokenizer.encode(text)
print(tokens)  # [128, 456, 789, ...]

# 디코딩
decoded = tokenizer.decode(tokens)
print(decoded)  # "안녕하세요, MiniMind입니다."

# 채팅 템플릿 적용
messages = [
    {"role": "user", "content": "안녕하세요"},
    {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
]
chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
print(chat_text)
# <|im_start|>user
# 안녕하세요<|im_end|>
# <|im_start|>assistant
# 안녕하세요! 무엇을 도와드릴까요?<|im_end|>
```

---

## 어휘 크기 선택

| 어휘 크기 | 장점 | 단점 |
|-----------|------|------|
| **작음 (6K)** | 모델 크기 작음, 훈련 빠름 | 긴 토큰화, OOV 많음 |
| **큼 (32K)** | 효율적 토큰화 | 모델 크기 큼 |

MiniMind는 **6,400**을 사용하여 모델 크기를 최소화합니다.

---

## 특수 토큰

### 채팅 형식

```
<|im_start|>system
당신은 도움이 되는 AI 어시스턴트입니다.<|im_end|>
<|im_start|>user
안녕하세요<|im_end|>
<|im_start|>assistant
안녕하세요! 무엇을 도와드릴까요?<|im_end|>
```

### 추론 모델용 토큰

```
<|im_start|>user
1+1은?<|im_end|>
<|im_start|>assistant
<think>
이건 간단한 산술 문제입니다.
1 + 1 = 2
</think>
2입니다.<|im_end|>
```

---

## 토크나이저 평가

```python
def evaluate_tokenizer(tokenizer, test_texts: list):
    """토크나이저 성능 평가"""
    total_tokens = 0
    total_chars = 0

    for text in test_texts:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        total_chars += len(text)

    # 평균 문자/토큰 비율
    compression_ratio = total_chars / total_tokens
    print(f"Compression ratio: {compression_ratio:.2f} chars/token")

    # OOV 비율
    unk_token_id = tokenizer.unk_token_id
    unk_count = sum(1 for t in tokens if t == unk_token_id)
    oov_ratio = unk_count / total_tokens
    print(f"OOV ratio: {oov_ratio:.2%}")
```

---

*다음 글에서는 사전 훈련(Pretrain)을 살펴봅니다.*
