---
layout: post
title: "MiniMind 완벽 가이드 (1) - 소개 및 개요"
date: 2025-02-04
permalink: /minimind-guide-01-intro/
author: jingyaogong
categories: [AI]
tags: [MiniMind, LLM, PyTorch, Training, From Scratch]
original_url: "https://github.com/jingyaogong/minimind"
excerpt: "3원(약 600원), 2시간으로 25.8M 파라미터 LLM을 처음부터 훈련하는 MiniMind 프로젝트를 소개합니다."
---

## MiniMind란?

**MiniMind**는 **완전히 처음부터(from scratch)** 대형 언어 모델(LLM)을 훈련하는 오픈소스 프로젝트입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MiniMind 핵심 가치                            │
├─────────────────────────────────────────────────────────────────┤
│  • 3원 비용 - GPU 서버 렌탈 비용 (약 600원)                     │
│  • 2시간 훈련 - NVIDIA 3090 단일 GPU                            │
│  • 25.8M 파라미터 - GPT-3의 1/7000 크기                         │
│  • 완전 오픈소스 - 데이터셋 + 코드 + 모델                       │
│  • 교육 목적 - LLM 내부 동작 이해                               │
└─────────────────────────────────────────────────────────────────┘
```

> "대도지간(大道至簡)" - 위대한 길은 단순하다

---

## 왜 MiniMind인가?

### 기존 LLM의 문제점

| 문제 | 설명 |
|------|------|
| **블랙박스** | 내부 동작 원리를 이해하기 어려움 |
| **거대한 규모** | 수십억 파라미터로 개인 훈련 불가능 |
| **추상화된 프레임워크** | transformers, trl 등이 핵심 로직 숨김 |
| **높은 비용** | 훈련에 수천만 원 필요 |

### MiniMind의 해결책

- 모든 핵심 알고리즘을 **PyTorch 네이티브로 구현**
- 제3자 라이브러리의 추상화 인터페이스 **의존하지 않음**
- **레고 블록처럼** 직접 조립하는 경험 제공

---

## 주요 기능

### 1. 완전한 LLM 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                    MiniMind Training Pipeline                    │
│                                                                  │
│   Tokenizer ──▶ Pretrain ──▶ SFT ──▶ RLHF ──▶ Deploy           │
│                                                                  │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│   │ BPE     │  │ Causal  │  │ Instruct│  │ DPO/PPO │           │
│   │Training │  │   LM    │  │ Tuning  │  │  GRPO   │           │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 모델 시리즈

| 모델 | 파라미터 | VRAM | 출시 |
|------|----------|------|------|
| **MiniMind2-small** | 26M | 0.5 GB | 2025.04 |
| **MiniMind2** | 104M | 1.0 GB | 2025.04 |
| **MiniMind2-MoE** | 145M | 1.0 GB | 2025.04 |

### 3. 지원 훈련 방법

- **Pretrain** - 사전 훈련 (Causal LM)
- **SFT** - 지도 학습 미세조정
- **LoRA** - 저랭크 적응
- **DPO** - 직접 선호 최적화
- **PPO/GRPO/SPO** - 강화 학습
- **Distillation** - 지식 증류

---

## 프로젝트 구조

```
minimind/
├── model/
│   ├── model_minimind.py    # 핵심 모델 아키텍처
│   ├── model_lora.py        # LoRA 구현
│   └── tokenizer.json       # 토크나이저
├── trainer/
│   ├── train_pretrain.py    # 사전 훈련
│   ├── train_full_sft.py    # SFT 전체 미세조정
│   ├── train_lora.py        # LoRA 미세조정
│   ├── train_dpo.py         # DPO 훈련
│   ├── train_ppo.py         # PPO 훈련
│   ├── train_grpo.py        # GRPO 훈련
│   ├── train_spo.py         # SPO 훈련
│   ├── train_reason.py      # 추론 모델 훈련
│   ├── train_distillation.py # 지식 증류
│   └── train_tokenizer.py   # 토크나이저 훈련
├── scripts/
│   ├── web_demo.py          # Streamlit 데모
│   ├── serve_openai_api.py  # OpenAI API 서버
│   └── convert_model.py     # 모델 변환
├── dataset/                 # 데이터셋
└── eval_llm.py              # 평가
```

---

## 빠른 시작

### 요구사항

```bash
# Python 3.10+
pip install torch transformers datasets

# 선택적: 분산 훈련
pip install deepspeed
```

### 훈련 실행

```bash
# 1. 클론
git clone https://github.com/jingyaogong/minimind.git
cd minimind

# 2. 사전 훈련
python trainer/train_pretrain.py

# 3. SFT
python trainer/train_full_sft.py

# 4. 추론
python scripts/web_demo.py
```

---

## 데모 체험

| 플랫폼 | 링크 |
|--------|------|
| **Reasoning Model** | [ModelScope](https://www.modelscope.cn/studios/gongjy/MiniMind-Reasoning) |
| **Regular Model** | [ModelScope](https://www.modelscope.cn/studios/gongjy/MiniMind) |
| **영상 소개** | [Bilibili](https://www.bilibili.com/video/BV12dHPeqE72/) |

---

## 모델 다운로드

| 플랫폼 | 링크 |
|--------|------|
| **Hugging Face** | [MiniMind Collection](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5) |
| **ModelScope** | [gongjy Profile](https://www.modelscope.cn/profile/gongjy) |

---

## 이 가이드에서 다루는 내용

1. **소개 및 개요** (현재 글)
2. **모델 아키텍처** - Transformer, RoPE, RMSNorm, MoE
3. **Tokenizer 훈련** - BPE 토크나이저 구현
4. **Pretrain** - 사전 훈련 구현
5. **SFT** - 지도 학습 미세조정
6. **LoRA** - 효율적 미세조정
7. **RLHF** - DPO/PPO/GRPO 강화 학습
8. **추론 및 배포** - 추론, API 서버, 생태계 통합

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **언어** | Python 3.10+ |
| **프레임워크** | PyTorch |
| **분산 훈련** | DDP, DeepSpeed |
| **시각화** | WandB, SwanLab |
| **추론** | llama.cpp, vLLM, Ollama |

---

## 라이선스

Apache 2.0 License

---

*다음 글에서는 MiniMind의 모델 아키텍처를 살펴봅니다.*
