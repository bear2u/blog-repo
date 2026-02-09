---
layout: post
title: "Tiny Audio 완벽 가이드 (01) - 소개 및 개요"
date: 2026-02-09
permalink: /tiny-audio-guide-01-intro/
author: Alex Kroman
categories: [머신러닝, 음성인식]
tags: [ASR, Speech Recognition, GLM-ASR, Qwen3, PyTorch, HuggingFace, Audio ML]
original_url: "https://github.com/alexkroman/tiny-audio"
excerpt: "24시간에 $12로 ASR 모델을 훈련할 수 있는 미니멀한 음성 인식 코드베이스 Tiny Audio를 소개합니다."
---

## Tiny Audio란?

Tiny Audio는 **24시간에 단돈 $12로 ASR(Automatic Speech Recognition) 모델을 훈련**할 수 있는 미니멀한 음성 인식 코드베이스입니다. 복잡한 설정 없이 누구나 쉽게 음성 인식 모델을 커스터마이징하고 훈련할 수 있도록 설계되었습니다.

### 주요 수치

- **훈련 시간**: 24시간 (단일 GPU)
- **훈련 비용**: $12 (클라우드 GPU 기준)
- **모델 크기**: 총 ~1.2B 파라미터 (실제 훈련되는 파라미터는 12M)
- **성능**: 상용 ASR 시스템과 비교 가능한 수준

## 왜 Tiny Audio인가?

### 1. 미니멀한 설계

기존 ASR 훈련 프레임워크는 수천 줄의 복잡한 코드로 구성되어 있어 이해하고 수정하기 어렵습니다. Tiny Audio는:

- **간결한 코드베이스**: 핵심 로직만 포함
- **명확한 구조**: 각 컴포넌트의 역할이 명확하게 분리
- **최소한의 의존성**: 필수적인 라이브러리만 사용

### 2. 해킹 가능한 구조

연구자와 개발자가 쉽게 실험할 수 있도록:

- **모듈화된 아키텍처**: 각 컴포넌트를 독립적으로 교체 가능
- **다양한 Projector 옵션**: MLP, MOSA, MoE, QFormer 중 선택
- **확장 가능한 설계**: 새로운 모델이나 기능 추가가 용이

### 3. 저비용 훈련

전체 모델을 훈련하는 대신:

- **Frozen 컴포넌트 활용**: 사전 훈련된 모델 재사용
- **효율적인 파라미터 튜닝**: 12M 파라미터만 훈련
- **GPU 메모리 최적화**: 단일 GPU로 훈련 가능

## 핵심 아이디어

Tiny Audio의 혁신적인 접근 방식은 3단계 아키텍처에 있습니다:

```
Audio Input (16kHz)
    ↓
[Frozen Audio Encoder]  ← GLM-ASR (600M params, 사전 훈련됨)
    ↓
Audio Features
    ↓
[Trainable Projector]   ← MLP (12M params, 훈련 대상)
    ↓
Text Embeddings
    ↓
[Frozen LLM]            ← Qwen3 (600M params, 사전 훈련됨)
    ↓
Transcription
```

### Frozen Audio Encoder (GLM-ASR)

- **역할**: 오디오 신호를 의미 있는 특징 벡터로 변환
- **모델**: GLM-ASR (600M 파라미터)
- **상태**: Frozen (훈련되지 않음)
- **이점**: 강력한 오디오 표현 능력

### Trainable Projector

- **역할**: 오디오 특징을 LLM이 이해할 수 있는 텍스트 임베딩으로 변환
- **모델**: MLP, MOSA, MoE, 또는 QFormer
- **상태**: Trainable (훈련 대상)
- **크기**: 약 12M 파라미터
- **핵심**: 전체 시스템에서 유일하게 훈련되는 부분

### Frozen LLM (Qwen3)

- **역할**: 텍스트 임베딩을 실제 텍스트로 디코딩
- **모델**: Qwen3-0.6B (600M 파라미터)
- **상태**: Frozen (훈련되지 않음)
- **이점**: 언어 모델의 강력한 생성 능력 활용

## 주요 특징

### 1. Pipeline 추론

HuggingFace Transformers의 표준 pipeline API를 지원합니다:

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="alexkroman/tiny-audio",
    trust_remote_code=True
)

result = pipe("audio.wav")
print(result["text"])
```

### 2. 스트리밍 추론

실시간 음성 인식을 위한 스트리밍 API:

```python
from tiny_audio.inference import StreamingASRInference

inference = StreamingASRInference()

for audio_chunk in stream:
    partial_result = inference.process_chunk(audio_chunk)
    print(f"Partial: {partial_result}")

final_result = inference.finalize()
print(f"Final: {final_result}")
```

### 3. Word-level Timestamps

각 단어의 시작과 끝 시간을 제공합니다:

```python
result = pipe("audio.wav", return_timestamps="word")

for word_info in result["chunks"]:
    print(f"{word_info['text']}: {word_info['timestamp']}")
```

출력 예시:
```
Hello: (0.0, 0.5)
world: (0.6, 1.2)
this: (1.3, 1.6)
is: (1.7, 1.9)
Tiny: (2.0, 2.4)
Audio: (2.5, 3.0)
```

### 4. CLI 도구

명령줄에서 바로 사용할 수 있는 도구:

```bash
# 단일 파일 추론
tiny-audio transcribe audio.wav

# 디렉토리 일괄 처리
tiny-audio batch-transcribe ./audio_dir/

# 스트리밍 모드
tiny-audio stream --input-device microphone
```

## 아키텍처 개요

### 전체 구조

```
┌─────────────────────────────────────────────────┐
│                 Tiny Audio System               │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────────┐                             │
│  │  Audio Input  │  16kHz PCM                  │
│  └───────┬───────┘                             │
│          │                                      │
│          ▼                                      │
│  ┌───────────────┐                             │
│  │  GLM-ASR      │  Frozen Encoder             │
│  │  (600M)       │  Audio → Features           │
│  └───────┬───────┘                             │
│          │                                      │
│          ▼                                      │
│  ┌───────────────┐                             │
│  │  Projector    │  Trainable Layer            │
│  │  (12M)        │  Features → Embeddings      │
│  └───────┬───────┘                             │
│          │                                      │
│          ▼                                      │
│  ┌───────────────┐                             │
│  │  Qwen3        │  Frozen LLM                 │
│  │  (600M)       │  Embeddings → Text          │
│  └───────┬───────┘                             │
│          │                                      │
│          ▼                                      │
│  ┌───────────────┐                             │
│  │  Transcription│                             │
│  └───────────────┘                             │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 데이터 흐름

1. **오디오 입력**: 16kHz PCM 형식의 원시 오디오
2. **특징 추출**: GLM-ASR이 오디오를 고차원 특징 벡터로 변환
3. **모달리티 브리지**: Projector가 오디오 특징을 텍스트 임베딩으로 변환
4. **텍스트 생성**: Qwen3이 임베딩을 자연어로 디코딩
5. **출력**: 최종 텍스트 전사 결과

## 사용 사례

### 1. 커스텀 도메인 ASR

특정 도메인(의료, 법률, 기술 등)에 특화된 ASR 모델을 빠르게 구축:

- 도메인 특화 데이터셋으로 Projector만 재훈련
- 24시간 이내에 커스텀 모델 완성
- 기존 모델 대비 도메인 정확도 향상

### 2. 다국어 ASR 실험

다양한 언어에 대한 ASR 시스템 실험:

- 언어별 Projector 훈련
- 저자원 언어 지원
- 다국어 모델 앙상블

### 3. 실시간 자막 생성

스트리밍 기능을 활용한 실시간 응용:

- 라이브 방송 자막
- 회의 실시간 전사
- 음성 비서 백엔드

### 4. 연구 및 교육

ASR 시스템 학습 및 연구:

- ASR 아키텍처 이해
- 새로운 Projector 설계 실험
- 논문 재현 및 검증

### 5. Edge 디바이스 배포

경량화된 모델로 엣지 배포:

- 모바일 디바이스
- IoT 기기
- 오프라인 환경

## 라이선스

Tiny Audio는 **MIT 라이선스**로 배포됩니다:

- 상업적 사용 가능
- 수정 및 재배포 자유
- 소스 코드 공개 의무 없음
- 저작권 표시만 필요

```
MIT License

Copyright (c) 2026 Alex Kroman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 다음 단계

이제 Tiny Audio의 기본 개념을 이해했다면, 다음 챕터에서 실제로 설치하고 사용해보겠습니다:

- **[챕터 2: 설치 및 빠른 시작](/tiny-audio-guide-02-quick-start/)** - 설치 방법과 기본 사용법
- **[챕터 3: 아키텍처 상세](/tiny-audio-guide-03-architecture/)** - 각 컴포넌트의 상세 구조

## 참고 자료

- GitHub 저장소: [https://github.com/alexkroman/tiny-audio](https://github.com/alexkroman/tiny-audio)
- HuggingFace 모델: [https://huggingface.co/alexkroman/tiny-audio](https://huggingface.co/alexkroman/tiny-audio)
- GLM-ASR 논문: [링크 추가 필요]
- Qwen3 모델: [https://huggingface.co/Qwen](https://huggingface.co/Qwen)
