---
layout: post
title: "effGen 완벽 가이드 (04) - 모델 및 백엔드"
date: 2026-02-09
permalink: /effgen-guide-04-models/
author: Gaurav Srivastava
categories: [AI 에이전트, Python]
tags: [SLM, AI Agent, Small Language Models, Tool Use, Multi-Agent, Python, Qwen, vLLM]
original_url: "https://github.com/ctrl-gaurav/effGen"
excerpt: "effGen에서 지원하는 다양한 모델 백엔드와 최적화 전략, 추천 SLM 모델 가이드"
---

# effGen 완벽 가이드 (04) - 모델 및 백엔드

## 목차
1. [지원 모델 백엔드](#지원-모델-백엔드)
2. [모델 로딩 옵션](#모델-로딩-옵션)
3. [백엔드별 성능 비교](#백엔드별-성능-비교)
4. [GPU 메모리 관리](#gpu-메모리-관리)
5. [추천 SLM 모델](#추천-slm-모델)
6. [모델 선택 가이드](#모델-선택-가이드)
7. [고급 설정](#고급-설정)

---

## 지원 모델 백엔드

effGen은 5가지 모델 백엔드를 지원하여 로컬 실행부터 클라우드 API까지 유연하게 선택할 수 있습니다.

### 백엔드 개요

| 백엔드 | 용도 | 장점 | 단점 |
|--------|------|------|------|
| **Transformers** | 로컬 추론 (기본) | 간편한 설정, 광범위한 모델 지원 | 상대적으로 느린 속도 |
| **vLLM** | 로컬 고속 추론 | 5-10x 빠른 속도, PagedAttention | 추가 설치 필요 |
| **OpenAI** | 클라우드 API | 강력한 성능 (GPT-4 등) | API 비용, 네트워크 의존 |
| **Anthropic** | 클라우드 API | Claude 시리즈 | API 비용, 네트워크 의존 |
| **Gemini** | 클라우드 API | Google의 멀티모달 모델 | API 비용, 네트워크 의존 |

### 1. Transformers (Hugging Face)

가장 범용적인 백엔드로, 수천 개의 오픈소스 모델을 지원합니다.

```python
from effgen import load_model

# 기본 사용
model = load_model(
    "Qwen/Qwen2.5-1.5B-Instruct",
    backend="transformers"  # 기본값이므로 생략 가능
)

# 상세 설정
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    backend="transformers",
    quantization="4bit",           # 메모리 절약
    device_map="auto",             # GPU 자동 할당
    torch_dtype="auto",            # 데이터 타입 자동 선택
    trust_remote_code=True         # 커스텀 코드 신뢰
)
```

**적합한 경우**:
- 처음 시작하거나 프로토타이핑 단계
- 다양한 모델을 빠르게 테스트하고 싶을 때
- GPU 메모리가 충분하지 않아 양자화가 필요한 경우

### 2. vLLM (고속 추론)

vLLM은 PagedAttention과 연속 배칭을 사용하여 처리량을 대폭 향상시킵니다.

```python
from effgen import load_model

# vLLM 백엔드
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    backend="vllm",
    tensor_parallel_size=1,        # GPU 병렬화 수
    dtype="half",                   # float16 사용
    max_model_len=8192,            # 최대 시퀀스 길이
    gpu_memory_utilization=0.9     # GPU 메모리 사용률
)

# 멀티 GPU 사용
model = load_model(
    "Qwen/Qwen2.5-14B-Instruct",
    backend="vllm",
    tensor_parallel_size=2,        # 2개 GPU에 분산
    pipeline_parallel_size=1
)
```

**설치**:
```bash
pip install vllm
```

**적합한 경우**:
- 프로덕션 환경에서 높은 처리량이 필요한 경우
- 여러 요청을 동시에 처리해야 하는 API 서버
- 충분한 GPU 메모리가 있는 경우 (최소 16GB+)

### 3. OpenAI API

GPT 시리즈 모델을 사용합니다.

```python
import os
from effgen import load_model

# API 키 설정
os.environ["OPENAI_API_KEY"] = "sk-..."

# GPT-4o Mini (비용 효율적)
model = load_model(
    "gpt-4o-mini",
    backend="openai",
    temperature=0.7,
    max_tokens=4096
)

# GPT-4 Turbo (강력한 성능)
model = load_model(
    "gpt-4-turbo-preview",
    backend="openai",
    temperature=0.5
)

# 커스텀 설정
model = load_model(
    "gpt-4o-mini",
    backend="openai",
    api_base="https://your-proxy.com/v1",  # 프록시 사용
    timeout=60,                            # 타임아웃
    max_retries=3                          # 재시도 횟수
)
```

**비용 참고** (2026년 2월 기준):
- GPT-4o Mini: $0.15/1M input tokens, $0.60/1M output tokens
- GPT-4 Turbo: $10/1M input tokens, $30/1M output tokens

### 4. Anthropic API

Claude 시리즈 모델을 사용합니다.

```python
import os
from effgen import load_model

# API 키 설정
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

# Claude 3.5 Sonnet (균형잡힌 성능)
model = load_model(
    "claude-3-5-sonnet-20240620",
    backend="anthropic",
    max_tokens=8192,
    temperature=0.7
)

# Claude 3 Haiku (빠르고 저렴)
model = load_model(
    "claude-3-haiku-20240307",
    backend="anthropic",
    max_tokens=4096
)

# Claude Opus (최고 성능)
model = load_model(
    "claude-3-opus-20240229",
    backend="anthropic",
    max_tokens=4096,
    temperature=0.5
)
```

**비용 참고**:
- Claude 3 Haiku: $0.25/1M input tokens, $1.25/1M output tokens
- Claude 3.5 Sonnet: $3/1M input tokens, $15/1M output tokens

### 5. Google Gemini

Google의 멀티모달 모델을 사용합니다.

```python
import os
from effgen import load_model

# API 키 설정
os.environ["GOOGLE_API_KEY"] = "AIza..."

# Gemini 1.5 Flash (빠르고 저렴)
model = load_model(
    "gemini-1.5-flash",
    backend="gemini",
    temperature=0.7,
    max_tokens=8192
)

# Gemini 1.5 Pro (강력한 성능)
model = load_model(
    "gemini-1.5-pro",
    backend="gemini",
    temperature=0.5,
    max_tokens=32768  # 긴 컨텍스트 지원
)
```

---

## 모델 로딩 옵션

effGen은 다양한 최적화 옵션을 제공합니다.

### 1. 양자화 (Quantization)

모델 가중치를 낮은 정밀도로 변환하여 메모리를 절약합니다.

```python
from effgen import load_model

# 4bit 양자화 (메모리 75% 절약)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization="4bit",
    bnb_4bit_compute_dtype="float16",      # 계산 시 사용할 타입
    bnb_4bit_quant_type="nf4",             # 정규화된 4bit
    bnb_4bit_use_double_quant=True         # 이중 양자화로 추가 절약
)

# 8bit 양자화 (메모리 50% 절약)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization="8bit",
    llm_int8_threshold=6.0                 # 양자화 임계값
)

# 양자화 없음 (최고 품질)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization=None,
    torch_dtype="float16"  # 또는 "bfloat16"
)
```

**메모리 사용량 비교** (Qwen2.5-7B 기준):

| 양자화 | 메모리 사용량 | 상대 품질 | 속도 |
|--------|--------------|----------|------|
| None (FP16) | ~14GB | 100% | 기준 |
| 8bit | ~7GB | ~98% | 1.1x |
| 4bit | ~3.5GB | ~95% | 1.3x |

### 2. Device Map (GPU 할당)

여러 GPU에 모델을 분산하거나 CPU 오프로딩을 설정합니다.

```python
# 자동 할당 (권장)
model = load_model(
    "Qwen/Qwen2.5-14B-Instruct",
    device_map="auto"  # 가용한 GPU에 자동 분산
)

# 수동 할당
model = load_model(
    "Qwen/Qwen2.5-14B-Instruct",
    device_map={
        "model.embed_tokens": 0,           # GPU 0에 임베딩
        "model.layers.0-15": 0,            # GPU 0에 레이어 0-15
        "model.layers.16-31": 1,           # GPU 1에 레이어 16-31
        "model.norm": 1,                   # GPU 1에 정규화
        "lm_head": 1                       # GPU 1에 출력 레이어
    }
)

# CPU 오프로딩
model = load_model(
    "Qwen/Qwen2.5-14B-Instruct",
    device_map="auto",
    offload_folder="./offload",            # CPU로 오프로드할 경로
    offload_state_dict=True
)
```

### 3. Max Memory (메모리 제한)

각 장치의 최대 메모리 사용량을 제한합니다.

```python
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    max_memory={
        0: "10GB",      # GPU 0에 최대 10GB
        1: "10GB",      # GPU 1에 최대 10GB
        "cpu": "30GB"   # CPU에 최대 30GB
    }
)
```

### 4. Attention Implementation

다양한 어텐션 구현을 선택할 수 있습니다.

```python
# Flash Attention 2 (가장 빠름)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    attn_implementation="flash_attention_2"  # A100, H100 등에서 지원
)

# SDPA (Scaled Dot Product Attention)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    attn_implementation="sdpa"  # PyTorch 2.0+ 기본
)

# Eager (표준 구현)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    attn_implementation="eager"  # 가장 호환성 높음
)
```

**설치** (Flash Attention 2):
```bash
pip install flash-attn --no-build-isolation
```

---

## 백엔드별 성능 비교

실제 벤치마크 결과를 통해 백엔드 간 성능 차이를 확인합니다.

### 처리량 벤치마크

**테스트 환경**:
- 모델: Qwen2.5-7B-Instruct
- GPU: NVIDIA A100 40GB
- 입력: 512 tokens, 출력: 128 tokens
- 동시 요청: 10개

| 백엔드 | 처리량 (req/sec) | 지연시간 (ms) | 상대 속도 |
|--------|------------------|---------------|-----------|
| Transformers (FP16) | 2.1 | 4,760 | 1.0x |
| Transformers (4bit) | 2.8 | 3,570 | 1.3x |
| vLLM (FP16) | 18.5 | 540 | 8.8x |
| vLLM (AWQ 4bit) | 24.3 | 410 | 11.6x |

### 메모리 효율성

**Qwen2.5-7B 모델 기준**:

| 설정 | GPU 메모리 | 배치 크기 | Tokens/sec |
|------|-----------|-----------|------------|
| Transformers FP16 | 14.2 GB | 1 | 42 |
| Transformers 4bit | 3.8 GB | 4 | 108 |
| vLLM FP16 | 15.6 GB | 32 | 856 |
| vLLM AWQ 4bit | 5.2 GB | 16 | 612 |

### 실전 예제: 백엔드 비교

```python
import time
from effgen import load_model, Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# 동일한 작업을 다른 백엔드로 실행
query = "Calculate (123.45 * 67.89) + sqrt(12345) and explain the steps"

# 1. Transformers (기본)
model_tf = load_model("Qwen/Qwen2.5-3B-Instruct", backend="transformers")
agent_tf = Agent(config=AgentConfig(model=model_tf, tools=[Calculator()]))

start = time.time()
result_tf = agent_tf.run(query)
time_tf = time.time() - start
print(f"Transformers: {time_tf:.2f}s")

# 2. vLLM (고속)
model_vllm = load_model("Qwen/Qwen2.5-3B-Instruct", backend="vllm")
agent_vllm = Agent(config=AgentConfig(model=model_vllm, tools=[Calculator()]))

start = time.time()
result_vllm = agent_vllm.run(query)
time_vllm = time.time() - start
print(f"vLLM: {time_vllm:.2f}s (speedup: {time_tf/time_vllm:.1f}x)")

# 출력 예시:
# Transformers: 3.45s
# vLLM: 0.42s (speedup: 8.2x)
```

---

## GPU 메모리 관리

효율적인 GPU 메모리 사용을 위한 전략입니다.

### 1. 메모리 요구사항 계산

모델의 메모리 사용량을 추정하는 공식:

```
메모리(GB) = (파라미터 수 × bytes_per_param) / (1024^3) × overhead_factor

- FP32: bytes_per_param = 4
- FP16/BF16: bytes_per_param = 2
- INT8: bytes_per_param = 1
- INT4: bytes_per_param = 0.5
- overhead_factor: 1.2-1.5 (KV 캐시, 활성화 등)
```

**예시 계산** (Qwen2.5-7B):
```python
# FP16
memory_fp16 = (7 * 10**9 * 2) / (1024**3) * 1.3
# = 17.0 GB

# 4bit
memory_4bit = (7 * 10**9 * 0.5) / (1024**3) * 1.2
# = 3.9 GB
```

### 2. GPU 선택 가이드

| GPU | VRAM | 추천 모델 (4bit) | 추천 모델 (FP16) |
|-----|------|------------------|------------------|
| RTX 3060 | 12GB | 최대 7B | 최대 3B |
| RTX 3090/4090 | 24GB | 최대 14B | 최대 7B |
| A100 40GB | 40GB | 최대 30B | 최대 14B |
| A100 80GB | 80GB | 최대 70B | 최대 30B |

### 3. 메모리 절약 기법

```python
from effgen import load_model
import torch

# 기법 1: 4bit 양자화 + 이중 양자화
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization="4bit",
    bnb_4bit_use_double_quant=True,  # 추가 ~0.5GB 절약
    bnb_4bit_compute_dtype="float16"
)

# 기법 2: Gradient Checkpointing (파인튜닝 시)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    gradient_checkpointing=True  # 활성화 메모리 절약
)

# 기법 3: KV 캐시 최적화
from effgen.core.agent import AgentConfig

config = AgentConfig(
    model=model,
    max_new_tokens=512,              # 출력 길이 제한
    cache_implementation="static"    # 정적 캐시 사용
)

# 기법 4: 메모리 정리
torch.cuda.empty_cache()  # 사용하지 않는 메모리 해제
torch.cuda.synchronize()  # GPU 동기화
```

### 4. 멀티 GPU 전략

```python
# 전략 1: Tensor Parallelism (vLLM)
model = load_model(
    "Qwen/Qwen2.5-14B-Instruct",
    backend="vllm",
    tensor_parallel_size=2  # 2개 GPU에 텐서 분할
)

# 전략 2: Pipeline Parallelism (Transformers)
model = load_model(
    "Qwen/Qwen2.5-14B-Instruct",
    device_map="auto",
    max_memory={0: "20GB", 1: "20GB"}  # 각 GPU에 균등 분배
)

# 전략 3: Sequential Loading (큰 모델)
model = load_model(
    "Qwen/Qwen2.5-32B-Instruct",
    device_map="auto",
    offload_folder="./offload",
    max_memory={
        0: "20GB",   # GPU 0
        1: "20GB",   # GPU 1
        "cpu": "50GB"  # 나머지는 CPU로
    }
)
```

---

## 추천 SLM 모델

effGen에 최적화된 Small Language Models를 소개합니다.

### 1. Qwen2.5 시리즈 (강력 추천)

Alibaba의 Qwen 시리즈는 effGen에 가장 적합합니다.

#### Qwen2.5-1.5B-Instruct

**특징**:
- 파라미터: 1.5B
- 컨텍스트 길이: 32K tokens
- 메모리: 4bit 양자화 시 ~1GB
- 속도: RTX 3060에서 ~45 tokens/sec

```python
from effgen import load_model

model = load_model(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization="4bit"
)
```

**장점**:
- 매우 빠른 추론 속도
- 낮은 메모리 요구사항
- 기본적인 도구 사용 능력 우수

**적합한 용도**:
- 실시간 대화형 애플리케이션
- 리소스 제약 환경
- 간단한 태스크 자동화

#### Qwen2.5-3B-Instruct

**특징**:
- 파라미터: 3B
- 컨텍스트 길이: 32K tokens
- 메모리: 4bit 양자화 시 ~2GB

```python
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization="4bit",
    attn_implementation="flash_attention_2"
)
```

**장점**:
- 1.5B 대비 추론 품질 향상
- 복잡한 도구 조합 가능
- 여전히 빠른 속도 유지

**적합한 용도**:
- 개인 비서 에이전트
- 데이터 분석 자동화
- 중간 복잡도의 멀티스텝 태스크

#### Qwen2.5-7B-Instruct

**특징**:
- 파라미터: 7B
- 컨텍스트 길이: 32K tokens
- 메모리: 4bit 양자화 시 ~4GB

```python
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization="4bit",
    device_map="auto"
)
```

**장점**:
- 높은 추론 품질
- 복잡한 멀티에이전트 조율 가능
- 긴 컨텍스트 이해 능력

**적합한 용도**:
- 연구 보조 에이전트
- 코드 생성 및 리뷰
- 복잡한 문제 해결

### 2. Phi-3 시리즈

Microsoft의 소형 모델로, 매우 높은 품질을 자랑합니다.

```python
# Phi-3-Mini (3.8B)
model = load_model(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization="4bit",
    trust_remote_code=True
)

# Phi-3-Small (7B)
model = load_model(
    "microsoft/Phi-3-small-8k-instruct",
    quantization="4bit",
    trust_remote_code=True
)

# Phi-3-Medium (14B)
model = load_model(
    "microsoft/Phi-3-medium-4k-instruct",
    quantization="4bit",
    trust_remote_code=True
)
```

**특징**:
- 높은 instruction following 능력
- 수학 및 코딩 태스크에 강함
- 상대적으로 짧은 컨텍스트 (4K-8K)

### 3. Gemma-2 시리즈

Google의 오픈소스 모델입니다.

```python
# Gemma-2-2B
model = load_model(
    "google/gemma-2-2b-it",
    quantization="4bit"
)

# Gemma-2-9B
model = load_model(
    "google/gemma-2-9b-it",
    quantization="4bit"
)
```

**특징**:
- 안전성 필터링 내장
- 다국어 지원
- 긴 컨텍스트 처리 가능

### 4. Llama-3.2 시리즈

Meta의 최신 소형 모델입니다.

```python
# Llama-3.2-1B
model = load_model(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantization="4bit"
)

# Llama-3.2-3B
model = load_model(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization="4bit"
)
```

**특징**:
- 빠른 추론 속도
- 효율적인 토큰 사용
- 강력한 커뮤니티 지원

### 모델 비교 벤치마크

**MMLU (Massive Multitask Language Understanding) 점수**:

| 모델 | 크기 | MMLU | 도구 사용 | 속도 | 메모리 (4bit) |
|------|------|------|-----------|------|---------------|
| Qwen2.5-1.5B | 1.5B | 61.2 | ⭐⭐⭐⭐ | ⚡⚡⚡⚡⚡ | 1GB |
| Qwen2.5-3B | 3B | 67.8 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡ | 2GB |
| Qwen2.5-7B | 7B | 74.5 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | 4GB |
| Phi-3-Mini | 3.8B | 68.8 | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | 2.5GB |
| Gemma-2-2B | 2B | 56.0 | ⭐⭐⭐ | ⚡⚡⚡⚡⚡ | 1.5GB |
| Llama-3.2-3B | 3B | 63.4 | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | 2GB |

---

## 모델 선택 가이드

사용 사례에 따른 최적의 모델을 선택하는 가이드입니다.

### 1. 개발 단계별 선택

```python
# 프로토타이핑 (빠른 실험)
model = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")
# - 빠른 반복 개발
# - 낮은 리소스 요구
# - 기본 기능 검증

# 개발 및 테스트
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
# - 실제 품질 평가
# - 엣지 케이스 테스트
# - 도구 통합 검증

# 프로덕션
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    backend="vllm",
    dtype="half",
    gpu_memory_utilization=0.9
)
# - 최고 품질
# - 높은 처리량
# - 안정적인 성능
```

### 2. 용도별 추천

```python
# 실시간 대화형 앱
model = load_model(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization="4bit",
    attn_implementation="flash_attention_2"
)
# 우선순위: 속도 > 품질

# 데이터 분석 자동화
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization="4bit"
)
# 우선순위: 균형 (속도 + 품질)

# 연구 및 복잡한 추론
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization="8bit"  # 더 높은 품질
)
# 우선순위: 품질 > 속도

# 코드 생성 및 리뷰
model = load_model(
    "microsoft/Phi-3-small-8k-instruct",
    quantization="4bit",
    trust_remote_code=True
)
# Phi-3는 코딩 태스크에 강함

# 고객 지원 봇
model = load_model(
    "google/gemma-2-2b-it",
    quantization="4bit"
)
# 안전성 필터링 내장
```

### 3. 하드웨어별 추천

```python
# RTX 3060 (12GB)
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization="4bit",
    max_memory={0: "10GB"}
)

# RTX 4090 (24GB)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization="4bit",
    # 또는 3B FP16
    # quantization=None, torch_dtype="float16"
)

# A100 (40GB)
model = load_model(
    "Qwen/Qwen2.5-14B-Instruct",
    backend="vllm",
    dtype="half"
)

# CPU Only (피하는 것이 좋음)
model = load_model(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization="4bit",
    device_map="cpu"
)
# 매우 느림 (~5 tokens/sec)
```

---

## 고급 설정

### 1. 생성 파라미터 튜닝

```python
from effgen import Agent
from effgen.core.agent import AgentConfig

agent = Agent(config=AgentConfig(
    model=model,
    # 생성 파라미터
    temperature=0.7,        # 창의성 (0.0-2.0)
    top_p=0.9,              # nucleus sampling
    top_k=50,               # top-k sampling
    repetition_penalty=1.1, # 반복 방지
    max_new_tokens=2048,    # 최대 출력 길이
    do_sample=True,         # 샘플링 활성화

    # 중단 조건
    stop_sequences=["</tool>", "Human:"],

    # 빔 서치 (더 나은 품질, 느림)
    num_beams=1,            # 1=greedy, >1=beam search
    early_stopping=False
))

# 태스크별 최적 설정
# 창의적 작업 (글쓰기, 브레인스토밍)
creative_config = AgentConfig(
    model=model,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.2
)

# 정확한 작업 (계산, 코딩)
precise_config = AgentConfig(
    model=model,
    temperature=0.1,
    top_p=0.9,
    repetition_penalty=1.0
)

# 균형잡힌 설정 (일반 대화)
balanced_config = AgentConfig(
    model=model,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)
```

### 2. 커스텀 토크나이저

```python
from transformers import AutoTokenizer

# 커스텀 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    padding_side="left"  # 배치 처리 시 왼쪽 패딩
)

# 특수 토큰 추가
special_tokens = {
    "additional_special_tokens": ["<|tool|>", "</|tool|>"]
}
tokenizer.add_special_tokens(special_tokens)

# 모델 로딩 시 전달
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    tokenizer=tokenizer
)
```

### 3. 로컬 모델 캐시

```python
import os

# 캐시 디렉토리 설정
os.environ["HF_HOME"] = "/data/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/transformers_cache"

# 오프라인 모드 (사전 다운로드 필요)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    local_files_only=True,  # 로컬 캐시만 사용
    cache_dir="/data/models"
)
```

### 4. 배치 추론

```python
from effgen import Agent
from effgen.core.agent import AgentConfig

# vLLM 백엔드로 배치 처리
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    backend="vllm",
    max_num_seqs=32  # 동시 시퀀스 수
)

agent = Agent(config=AgentConfig(model=model))

# 여러 쿼리를 배치로 처리
queries = [
    "What is the capital of France?",
    "Calculate 25 * 4",
    "Summarize the concept of machine learning"
]

# 병렬 실행
results = agent.batch_run(queries, batch_size=8)
for query, result in zip(queries, results):
    print(f"Q: {query}\nA: {result}\n")
```

---

## 다음 단계

이제 effGen의 다양한 모델 백엔드와 최적화 전략을 이해했습니다. 다음 챕터에서는 강력한 도구 시스템과 프로토콜 어댑터를 살펴봅니다.

**[다음: 챕터 05 - 도구 시스템 및 프로토콜 →](/effgen-guide-05-tools/)**

---

## 참고 자료

1. vLLM Documentation. https://docs.vllm.ai/
2. Hugging Face Transformers. https://huggingface.co/docs/transformers
3. Qwen2.5 Technical Report. https://arxiv.org/abs/2412.15115
4. BitsAndBytes Quantization. https://github.com/TimDettmers/bitsandbytes

---

**전체 가이드 목차**:
- [01장: 소개 및 개요](/effgen-guide-01-intro/)
- [02장: 설치 및 빠른 시작](/effgen-guide-02-quick-start/)
- [03장: 핵심 아키텍처](/effgen-guide-03-architecture/)
- [04장: 모델 및 백엔드](/effgen-guide-04-models/) ← 현재 문서
- [05장: 도구 시스템 및 프로토콜](/effgen-guide-05-tools/)
- [06장: 멀티에이전트 및 태스크 분해](/effgen-guide-06-multi-agent/)
- [07장: 고급 활용 및 프로덕션](/effgen-guide-07-advanced/)
