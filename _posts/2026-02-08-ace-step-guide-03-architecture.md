---
layout: post
title: "ACE-Step 1.5 완벽 가이드 (03) - 아키텍처 분석"
date: 2026-02-08
permalink: /ace-step-guide-03-architecture/
author: ACE Studio & StepFun
categories: [AI 음악, 오픈소스]
tags: [ACE-Step, Architecture, DiT, LLM, Diffusion Transformer, Language Model, Model Zoo]
original_url: "https://github.com/ace-step/ACE-Step-1.5"
excerpt: "DiT + LM 하이브리드 구조와 Intrinsic RL의 비밀"
---

## ACE-Step 1.5 아키텍처 개요

ACE-Step 1.5는 **Language Model (LM)**과 **Diffusion Transformer (DiT)**의 하이브리드 구조를 사용하여 상업급 음악을 생성합니다.

```
┌──────────────────────────────────────────────────────────────┐
│                      ACE-Step 1.5                             │
│                   Hybrid Architecture                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  User Query                                                   │
│    │                                                          │
│    │  "Upbeat pop song with guitar"                          │
│    ↓                                                          │
│  ┌───────────────────────────────────────────┐               │
│  │   Language Model (Omni-capable Planner)   │               │
│  │   ───────────────────────────────────     │               │
│  │   • Query Rewriting & Expansion           │               │
│  │   • Chain-of-Thought Reasoning            │               │
│  │   • Metadata Synthesis (BPM, Key, etc.)   │               │
│  │   • Lyrics & Caption Generation           │               │
│  │   • Audio Semantic Codes (5Hz)            │               │
│  └───────────────────────────────────────────┘               │
│                    ↓                                          │
│    Blueprint: {                                               │
│      caption: "Upbeat pop rock with electric guitars...",    │
│      lyrics: "[Verse 1]...",                                  │
│      bpm: 120,                                                │
│      keyscale: "C Major",                                     │
│      audio_codes: "<|audio_code_123|>...",                    │
│    }                                                          │
│                    ↓                                          │
│  ┌───────────────────────────────────────────┐               │
│  │      Diffusion Transformer (DiT)          │               │
│  │      ─────────────────────────────────    │               │
│  │   • VAE Latent Diffusion (5Hz)            │               │
│  │   • Adaptive Dual Guidance (ADG)          │               │
│  │   • Turbo Models (8-step)                 │               │
│  │   • Base Models (50-step)                 │               │
│  └───────────────────────────────────────────┘               │
│                    ↓                                          │
│    Generated Music (10s ~ 600s)                               │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Language Model as Omni-capable Planner

### LM의 역할

ACE-Step의 **Language Model**은 단순한 텍스트 처리기가 아닌 **전능한 계획자(Omni-capable Planner)**입니다.

```python
# LM의 핵심 기능
class LMHandler:
    def plan_music(self, user_query: str) -> Blueprint:
        """
        사용자 쿼리를 받아 포괄적인 음악 설계도(Blueprint) 생성
        """
        # 1. Query Rewriting & Expansion
        #    "pop song" → "Upbeat pop rock with electric guitars,
        #                  driving drums, and catchy synth hooks"

        # 2. Chain-of-Thought Reasoning
        #    Caption → BPM, Key, Time Signature, Duration 추론

        # 3. Lyrics Generation
        #    Verse, Chorus, Bridge 구조 생성

        # 4. Audio Semantic Codes
        #    5Hz audio tokens 생성 (LM-DiT 연결)

        return Blueprint(
            caption="...",
            lyrics="...",
            bpm=120,
            keyscale="C Major",
            audio_codes="<|audio_code_123|>...",
        )
```

### Chain-of-Thought (CoT) 메타데이터 추론

LM은 **Chain-of-Thought**를 사용하여 누락된 메타데이터를 추론합니다.

```
사용자 입력: "Energetic rock ballad"

LM CoT 추론 과정:
1. Genre: Rock ballad
   → BPM은 보통 60-80 사이

2. "Energetic" 키워드
   → BPM을 약간 높게 설정 (80-100)

3. Rock ballad의 일반적 특징
   → Key: Minor (감정적)
   → Time Signature: 4/4 (표준)

최종 출력:
{
  "bpm": 90,
  "keyscale": "A minor",
  "timesignature": "4",
  "duration": 180
}
```

### 5Hz Audio Semantic Codes

LM은 **5Hz 샘플링**으로 audio semantic tokens를 생성하여 DiT를 가이드합니다.

```
텍스트 → LM → Audio Semantic Codes (5Hz)
                     ↓
                   DiT → 48kHz Audio

Example Codes:
<|audio_code_10695|><|audio_code_54246|><|audio_code_89123|>...
        ↑                   ↑                   ↑
   0.2초 시점          0.4초 시점          0.6초 시점
```

**5Hz의 의미:**
- 1초에 5개의 semantic token
- 30초 곡 = 150 tokens
- 180초 곡 = 900 tokens

---

## Diffusion Transformer (DiT) 동작 원리

### VAE Latent Diffusion

```
Original Audio (48kHz)
       ↓
┌──────────────┐
│  VAE Encoder │  압축: 48000 samples/s → 5Hz latent
└──────────────┘
       ↓
Latent Space (5Hz)
       ↓
┌──────────────┐
│  DiT Model   │  Denoising process
└──────────────┘
       ↓
Latent Space (denoised)
       ↓
┌──────────────┐
│ VAE Decoder  │  복원: 5Hz latent → 48kHz audio
└──────────────┘
       ↓
Generated Audio (48kHz)
```

### Denoising Process

```python
# Turbo Model (8 steps)
timesteps = [0.97, 0.76, 0.615, 0.5, 0.395, 0.28, 0.18, 0.085, 0]
#            ↑                                                  ↑
#      완전한 노이즈                                        깨끗한 오디오

# Base Model (50 steps)
timesteps = linspace(1.0, 0, steps=50)

# Custom Timesteps (직접 지정 가능)
timesteps = [0.97, 0.85, 0.70, 0.50, 0.30, 0.15, 0.05, 0]
```

**Timestep Shift (shift 파라미터):**

```python
# shift = 1.0 (기본)
t_original = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

# shift = 3.0 (Turbo 권장)
# t_new = shift * t / (1 + (shift - 1) * t)
t_shifted = [1.0, 0.923, 0.818, 0.667, 0.429, 0.0]
#                    ↑ 더 많은 시간을 중간~끝 단계에 할애
```

### Classifier-Free Guidance (CFG)

```
┌─────────────────────────────────────────────────────────┐
│  Classifier-Free Guidance                               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  prediction = unconditional + scale * (conditional -     │
│                                        unconditional)    │
│                                                          │
│  - unconditional: 빈 프롬프트로 예측                        │
│  - conditional: 실제 프롬프트로 예측                         │
│  - scale: guidance_scale (1.0 ~ 15.0)                   │
│                                                          │
│  guidance_scale이 높을수록 프롬프트 충실도 증가               │
│  (단, Turbo 모델은 CFG 비활성화)                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Adaptive Dual Guidance (ADG)

**Base 모델 전용**

```
표준 CFG:
prediction = uncond + scale * (cond - uncond)

ADG (Adaptive Dual Guidance):
prediction = uncond + scale_1 * (cond - uncond) +
                      scale_2 * (reference - uncond)

장점:
- 참조 오디오 스타일 더 정확히 반영
- Cover/Repaint 작업 시 품질 향상
```

---

## Intrinsic Reinforcement Learning

ACE-Step의 독특한 특징은 **Intrinsic RL**을 사용한다는 점입니다.

### 기존 방식 (External Reward)

```
┌──────────────────────────────────────────────────────────┐
│  기존 RLHF (Reinforcement Learning from Human Feedback)   │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. 인간이 생성된 음악을 평가 (좋음/나쁨)                     │
│  2. Reward Model 훈련                                     │
│  3. Reward Model로 생성 모델 fine-tune                     │
│                                                           │
│  문제점:                                                   │
│  - 인간 선호도의 편향 (Bias)                                │
│  - Reward Model의 제한적 일반화                             │
│  - 인간 평가 데이터 수집 비용                                │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### ACE-Step의 Intrinsic RL

```
┌──────────────────────────────────────────────────────────┐
│  Intrinsic Reinforcement Learning                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. 모델 자체의 내부 메커니즘으로 품질 판단                     │
│  2. 외부 Reward Model 없음                                 │
│  3. 인간 선호도에 의존하지 않음                               │
│                                                           │
│  장점:                                                     │
│  ✓ 편향 제거 (No external bias)                           │
│  ✓ 일반화 능력 향상                                         │
│  ✓ 훈련 데이터 요구량 감소                                   │
│  ✓ 더 다양한 스타일 생성 가능                                │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## Model Zoo

ACE-Step은 다양한 DiT 모델과 LM 모델을 제공합니다.

### DiT Models (4종)

| DiT Model | Pre-Training | SFT | RL | CFG | Steps | Quality | Diversity | Fine-Tunability |
|-----------|:------------:|:---:|:--:|:---:|:-----:|:-------:|:---------:|:---------------:|
| `acestep-v15-base` | ✅ | ❌ | ❌ | ✅ | 50 | Medium | High | Easy |
| `acestep-v15-sft` | ✅ | ✅ | ❌ | ✅ | 50 | High | Medium | Easy |
| `acestep-v15-turbo` | ✅ | ✅ | ❌ | ❌ | 8 | Very High | Medium | Medium |
| `acestep-v15-turbo-rl` | ✅ | ✅ | ✅ | ❌ | 8 | Very High | Medium | Medium |

**특징 비교:**

```yaml
acestep-v15-base:
  장점: 높은 다양성, 쉬운 fine-tuning, Extract/Lego/Complete 지원
  단점: 느린 생성 속도 (50 steps)

acestep-v15-sft:
  장점: 향상된 품질, SFT로 프롬프트 충실도 개선
  단점: Extract/Lego/Complete 미지원

acestep-v15-turbo:
  장점: 초고속 생성 (8 steps), 상업급 품질
  단점: CFG 비활성화, 중간 정도의 fine-tunability
  권장: 대부분의 사용 사례에 최적

acestep-v15-turbo-rl:
  장점: Intrinsic RL로 더 높은 품질
  상태: 출시 예정
```

### LM Models (3종)

| LM Model | Base | Params | Audio Understanding | Composition | Copy Melody |
|----------|------|-------:|:-------------------:|:-----------:|:-----------:|
| `acestep-5Hz-lm-0.6B` | Qwen3-0.6B | 600M | Medium | Medium | Weak |
| `acestep-5Hz-lm-1.7B` | Qwen3-1.7B | 1.7B | Medium | Medium | Medium |
| `acestep-5Hz-lm-4B` | Qwen3-4B | 4B | Strong | Strong | Strong |

**모델 선택 가이드:**

```yaml
acestep-5Hz-lm-0.6B:
  VRAM: 6-12GB
  속도: 빠름
  장점: 낮은 메모리 사용, 기본 메타데이터 추론
  단점: 약한 오디오 이해, 제한적인 작곡 능력

acestep-5Hz-lm-1.7B:
  VRAM: 12-16GB
  속도: 중간
  장점: 균형잡힌 성능, 대부분의 작업에 충분
  권장: 기본 모델

acestep-5Hz-lm-4B:
  VRAM: 16GB+
  속도: 느림
  장점: 최고 품질, 강력한 오디오 이해, 복잡한 작곡
  추천: 전문가용, 고품질 작업
```

---

## 모델 선택 전략

### 사용 사례별 권장 조합

```python
# 빠른 실험 & 프로토타입 (VRAM 6-12GB)
config = {
    "dit_model": "acestep-v15-turbo",
    "lm_model": "acestep-5Hz-lm-0.6B",
    "inference_steps": 8,
}

# 균형잡힌 품질 & 속도 (VRAM 12-16GB)
config = {
    "dit_model": "acestep-v15-turbo",
    "lm_model": "acestep-5Hz-lm-1.7B",
    "inference_steps": 8,
}

# 최고 품질 (VRAM 16GB+)
config = {
    "dit_model": "acestep-v15-turbo",
    "lm_model": "acestep-5Hz-lm-4B",
    "inference_steps": 8,
}

# Fine-tuning & 연구 (VRAM 16GB+)
config = {
    "dit_model": "acestep-v15-base",
    "lm_model": "acestep-5Hz-lm-1.7B",
    "inference_steps": 50,
    "use_adg": True,
    "guidance_scale": 7.0,
}

# DiT 전용 (VRAM ≤6GB, LLM 없이)
config = {
    "dit_model": "acestep-v15-turbo",
    "lm_model": None,
    "init_llm": False,
    "inference_steps": 8,
}
```

### 작업 유형별 모델 선택

```yaml
Text2Music (텍스트 → 음악):
  권장: acestep-v15-turbo + acestep-5Hz-lm-1.7B

Cover Generation (스타일 변환):
  권장: acestep-v15-base + ADG
  이유: CFG 지원, 참조 오디오 활용

Repaint (부분 수정):
  권장: acestep-v15-turbo (빠른 반복)

Extract/Lego/Complete (트랙 분리/생성):
  필수: acestep-v15-base
  이유: Turbo 모델은 미지원

LoRA Training (커스텀 스타일):
  권장: acestep-v15-turbo
  이유: 빠른 검증 반복
```

---

## 아키텍처 다이어그램

### 전체 파이프라인

```
User Input
    ↓
┌─────────────────────────────────────────────┐
│  LLM Handler                                 │
│  ─────────────────────────────────────       │
│                                              │
│  IF thinking=True:                           │
│    1. CoT Metadata Inference                 │
│       caption → BPM, Key, Duration...        │
│                                              │
│    2. Query Rewriting (optional)             │
│       "pop" → "Upbeat pop rock with..."      │
│                                              │
│    3. Audio Semantic Codes Generation        │
│       text → 5Hz audio tokens                │
│                                              │
└─────────────────────────────────────────────┘
    ↓
Blueprint {
  caption: str,
  lyrics: str,
  bpm: int,
  keyscale: str,
  audio_codes: str,
  ...
}
    ↓
┌─────────────────────────────────────────────┐
│  DiT Handler                                 │
│  ─────────────────────────────────────────  │
│                                              │
│  1. Text Embedding (Qwen3-Embedding)         │
│     caption + lyrics → embeddings            │
│                                              │
│  2. Audio Codes Embedding                    │
│     audio_codes → condition vector           │
│                                              │
│  3. VAE Latent Initialization                │
│     Random noise → latent space (5Hz)        │
│                                              │
│  4. Denoising Loop (8~50 steps)              │
│     FOR each timestep:                       │
│       latent = DiT(latent, condition, t)     │
│                                              │
│  5. VAE Decode                               │
│     latent (5Hz) → audio (48kHz)             │
│                                              │
└─────────────────────────────────────────────┘
    ↓
Generated Audio (48kHz, 10s~600s)
```

### DiT 내부 구조

```
Input Latent (B, C, L)  # B=batch, C=channels, L=length
    ↓
┌──────────────────┐
│ Time Embedding   │  t → time_emb
└──────────────────┘
    ↓
┌──────────────────┐
│ Condition Concat │  [latent, text_emb, audio_codes_emb]
└──────────────────┘
    ↓
┌──────────────────┐
│ Transformer      │  N layers of:
│  Blocks (N)      │  - Self-Attention
│                  │  - Cross-Attention (condition)
│                  │  - FFN
└──────────────────┘
    ↓
┌──────────────────┐
│ Output Projection│  → predicted noise
└──────────────────┘
    ↓
noise_pred → latent_next
```

---

## 핵심 개념 정리

### 1. Hybrid Architecture

```
LM (Planner) + DiT (Generator) = 상업급 품질
```

- LM: 사용자 쿼리 → 포괄적 블루프린트
- DiT: 블루프린트 → 고품질 음악

### 2. 5Hz Semantic Codes

```
48kHz Audio → VAE → 5Hz Latent → Semantic Tokens
```

- 압축률: 9600:1
- 30초 곡 = 150 tokens

### 3. Intrinsic RL

```
No External Reward Model
→ Less Bias
→ Better Generalization
```

### 4. Turbo vs Base

```
Turbo: 8 steps, No CFG, 초고속
Base: 50 steps, CFG, Extract/Lego/Complete
```

### 5. Model Zoo 전략

```
VRAM ≤6GB:   DiT only
6-12GB:      DiT + LM-0.6B
12-16GB:     DiT + LM-1.7B (권장)
16GB+:       DiT + LM-4B (최고 품질)
```

---

## 다음 챕터

다음 챕터에서는 **Gradio UI 사용법**을 상세히 다룹니다:

- Generation 탭 사용법
- Simple Mode vs Advanced Mode
- 참조 오디오 및 메타데이터 제어
- Dataset 및 Training 탭

**[04장: Gradio UI 사용법](/ace-step-guide-04-gradio-ui/)**
