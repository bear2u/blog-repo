---
layout: post
title: "ACE-Step 1.5 완벽 가이드 (09) - GPU 최적화"
date: 2026-02-08
permalink: /ace-step-guide-09-gpu-optimization/
author: ACE Studio & StepFun
categories: [AI 음악, 오픈소스]
tags: [ACE-Step, GPU, VRAM, Optimization, CUDA, ROCm, Intel GPU, MPS, Performance]
original_url: "https://github.com/ace-step/ACE-Step-1.5"
excerpt: "VRAM별 최적화 전략과 GPU 호환성 완벽 가이드"
---

## GPU 최적화 개요

ACE-Step 1.5는 **자동 GPU 적응 시스템**을 제공합니다. 시작 시 GPU의 사용 가능한 VRAM을 감지하고, 최적의 설정을 자동으로 구성합니다.

### 최적화 목표

```
┌──────────────────────────────────────────────────────────┐
│  GPU 자동 감지                                             │
│    ↓                                                      │
│  VRAM 기반 Tier 분류 (Tier 1~7)                           │
│    ↓                                                      │
│  최적화 전략 적용:                                          │
│    • LM 모델 선택 (0.6B/1.7B/4B/비활성화)                   │
│    • Duration 제한 (3~10분)                                │
│    • Batch 크기 조정 (1~8)                                 │
│    • CPU Offload 자동 활성화                                │
│    • Quantization & Compile 기본 활성화                     │
└──────────────────────────────────────────────────────────┘
```

---

## VRAM별 최적화 전략

### GPU Tier 구성표

ACE-Step 1.5는 VRAM에 따라 7개의 Tier로 분류됩니다.

| VRAM | Tier | LM 모드 | 최대 Duration | 최대 Batch | LM 메모리 할당 |
|------|------|---------|--------------|-----------|--------------|
| **≤4GB** | Tier 1 | 사용 불가 | 3분 | 1 | - |
| **4-6GB** | Tier 2 | 사용 불가 | 6분 | 1 | - |
| **6-8GB** | Tier 3 | 0.6B (선택) | LM: 4분 / 없음: 6분 | LM: 1 / 없음: 2 | 3GB |
| **8-12GB** | Tier 4 | 0.6B (선택) | LM: 4분 / 없음: 6분 | LM: 2 / 없음: 4 | 3GB |
| **12-16GB** | Tier 5 | 0.6B / 1.7B | LM: 4분 / 없음: 6분 | LM: 2 / 없음: 4 | 0.6B: 3GB, 1.7B: 8GB |
| **16-24GB** | Tier 6 | 0.6B / 1.7B / 4B | 8분 | LM: 4 / 없음: 8 | 0.6B: 3GB, 1.7B: 8GB, 4B: 12GB |
| **≥24GB** | Unlimited | 모든 모델 | 10분 | 8 | 제한 없음 |

### Tier별 권장 설정

#### Tier 1-2: ≤6GB VRAM (RTX 3060, GTX 1660 Ti)

**DiT 전용 모드 - LLM 비활성화로 VRAM 확보**

```bash
# 환경변수 설정
export ACESTEP_INIT_LLM=false

# 실행
uv run acestep --config_path acestep-v15-turbo
```

**특징:**
- ✅ **빠른 생성 속도** - LM 추론 과정 생략
- ✅ **메모리 절약** - DiT에 전체 VRAM 할당
- ⚠️ **제한된 기능** - CoT, Query Rewrite, Audio Understanding 비활성화

**권장 워크플로우:**

```python
# Simple 프롬프트로 직접 생성
prompt = "upbeat electronic dance music with strong bass"
# → DiT가 직접 해석하여 생성 (LM 없음)
```

#### Tier 3-4: 6-12GB VRAM (RTX 3060 Ti, RTX 4060)

**경량 LM 모델 + CPU Offload**

```bash
# 실행 예시
uv run acestep \
  --lm_model_path acestep-5Hz-lm-0.6B \
  --offload_to_cpu true \
  --config_path acestep-v15-turbo
```

**최적화 설정:**

```python
# .env 설정
ACESTEP_INIT_LLM=auto
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
ACESTEP_OFFLOAD_TO_CPU=true
```

**권장 사용 패턴:**
- ✅ **짧은 음악** (1-2분) - 안정적 생성
- ✅ **배치 크기 1-2** - 메모리 안전
- ⚠️ **CoT 활용** - 0.6B 모델은 중간 수준 품질

#### Tier 5: 12-16GB VRAM (RTX 3080, RTX 4070 Ti)

**표준 LM 모델 - 균형잡힌 품질**

```bash
# 권장 설정
uv run acestep \
  --lm_model_path acestep-5Hz-lm-1.7B \
  --config_path acestep-v15-turbo
```

**최적화 포인트:**

```yaml
LM Model: acestep-5Hz-lm-1.7B
Max Duration: 4분 (LM 사용), 6분 (DiT only)
Batch Size: 2 (LM 사용), 4 (DiT only)
Offload: Auto (필요시 활성화)
Quantization: Enabled
Compile: Enabled
```

**권장 워크플로우:**
- ✅ **풀송 생성** (3-4분) - 안정적
- ✅ **CoT + Query Rewrite** - 좋은 품질
- ✅ **Cover/Repaint** - 원활한 편집

#### Tier 6: 16-24GB VRAM (RTX 3090, RTX 4080)

**대형 LM 모델 - 고품질 생성**

```bash
# 권장 설정
uv run acestep \
  --lm_model_path acestep-5Hz-lm-4B \
  --config_path acestep-v15-turbo
```

**최적화 전략:**

```python
# 배치 생성 활용
settings = {
    "lm_model": "acestep-5Hz-lm-4B",
    "max_duration": 480,  # 8분
    "batch_size": 4,      # LM 사용 시
    "offload_to_cpu": False,
    "quantization": True,
    "compile": True
}
```

**권장 사용 패턴:**
- ✅ **긴 곡 생성** (6-8분) - 안정적
- ✅ **배치 생성** (4개 동시) - 빠른 탐색
- ✅ **Audio Understanding** - 강력한 오디오 분석
- ✅ **고품질 CoT** - 4B 모델의 뛰어난 메타데이터 생성

#### Tier 7: ≥24GB VRAM (RTX 4090, A100)

**무제한 모드 - 최대 성능**

```bash
# 최대 성능 설정
uv run acestep \
  --lm_model_path acestep-5Hz-lm-4B \
  --config_path acestep-v15-turbo \
  --init_service true
```

**최적화 없이 최대 활용:**

```python
# 10분 음악 생성
settings = {
    "duration": 600,      # 10분 (600초)
    "batch_size": 8,      # 최대 배치
    "lm_model": "4B",
    "offload": False,
    "quantization": True,
    "compile": True
}
```

**고급 워크플로우:**
- ✅ **장편 생성** (10분+) - 완전 지원
- ✅ **대량 배치** (8개 동시) - 빠른 변형 탐색
- ✅ **LoRA 훈련** - 여유로운 VRAM으로 빠른 훈련
- ✅ **Multi-Track** - 복잡한 다중 트랙 작업

---

## DiT 전용 모드 vs LLM 모드

### DiT 전용 모드 (LLM 비활성화)

**활성화 방법:**

```bash
# 방법 1: 환경변수
export ACESTEP_INIT_LLM=false
uv run acestep

# 방법 2: 커맨드 라인
uv run acestep --init_llm false

# 방법 3: .env 파일
ACESTEP_INIT_LLM=false
```

**장점:**
- ⚡ **빠른 생성** - LM 추론 과정 생략 (약 30-50% 빠름)
- 💾 **낮은 VRAM** - LM 메모리 없이 DiT에 집중
- 🚀 **높은 Duration** - Tier 3에서도 6분 생성 가능

**단점:**
- ❌ **CoT 비활성화** - 메타데이터 자동 생성 없음
- ❌ **Query Rewrite 없음** - 간단한 프롬프트 직접 사용
- ❌ **Audio Understanding 없음** - 오디오 분석 기능 비활성화

**사용 시나리오:**
```python
# DiT only - 직접적인 프롬프트
prompt = "epic orchestral soundtrack with strings and brass"
# → DiT가 직접 해석, 빠르게 생성

# LM 모드에서는:
# User Query → LM (CoT) → Metadata + Caption → DiT
# DiT only에서는:
# User Query → DiT (직접)
```

### LLM 모드 (Chain-of-Thought)

**활성화 방법:**

```bash
# 자동 감지 (권장)
uv run acestep  # VRAM > 6GB이면 자동 활성화

# 강제 활성화 (위험: OOM 가능)
uv run acestep --init_llm true --lm_model_path acestep-5Hz-lm-0.6B
```

**LM 모델별 특징:**

| 모델 | 크기 | VRAM | CoT 품질 | Audio Understanding | Query Rewrite |
|------|------|------|---------|---------------------|---------------|
| **0.6B** | ~600MB | 6-12GB | 중간 | 중간 | ✅ |
| **1.7B** | ~1.7GB | 12-16GB | 좋음 | 좋음 | ✅ |
| **4B** | ~4GB | 16GB+ | 뛰어남 | 강력함 | ✅ |

**LM 모드 워크플로우:**

```python
# 1. User Query (간단한 입력)
query = "chill lofi hip hop for studying"

# 2. LM Chain-of-Thought 생성
cot_output = {
    "duration": 180,
    "bpm": 85,
    "key": "C major",
    "time_signature": "4/4",
    "caption": "Relaxing lofi hip hop beat with mellow piano, soft drums, and vinyl crackle",
    "structure": "intro, verse, chorus, verse, outro"
}

# 3. DiT 생성 (LM의 블루프린트 기반)
generated_audio = dit_generate(cot_output)
```

**장점:**
- 🎯 **정교한 메타데이터** - 자동으로 BPM, Key, Structure 생성
- 🎨 **창의적 확장** - 간단한 프롬프트를 풍부한 캡션으로 확장
- 🔍 **Audio Understanding** - 참조 오디오 분석 (4B 모델 강력)

---

## CPU 오프로드 (offload_to_cpu)

### CPU Offload란?

**VRAM이 부족할 때, 일부 모델 레이어를 CPU RAM으로 이동하여 GPU 메모리를 절약합니다.**

```
┌─────────────────────────────────────────┐
│  GPU VRAM (Limited)                     │
│  ─────────────────────                  │
│  • DiT Core Layers (필수)               │
│  • LM Active Layers (추론 중)            │
└─────────────────────────────────────────┘
          ↕ (Offload)
┌─────────────────────────────────────────┐
│  CPU RAM (Larger)                       │
│  ─────────────────────                  │
│  • LM Inactive Layers                   │
│  • DiT Non-Critical Layers              │
└─────────────────────────────────────────┘
```

### 자동 활성화 조건

```python
# ACE-Step 내부 로직
if vram < 16_000:  # 16GB 미만
    offload_to_cpu = True
else:
    offload_to_cpu = False
```

### 수동 제어

```bash
# 강제 활성화 (VRAM 절약)
uv run acestep --offload_to_cpu true

# 강제 비활성화 (빠른 생성)
uv run acestep --offload_to_cpu false

# 환경변수
export ACESTEP_OFFLOAD_TO_CPU=true
```

### 성능 비교

| 설정 | VRAM 사용량 | 생성 속도 | 권장 시나리오 |
|------|------------|----------|--------------|
| **Offload: True** | 낮음 (-30%) | 중간 (-10~20% 느림) | VRAM < 16GB |
| **Offload: False** | 높음 | 빠름 | VRAM ≥ 16GB |

**실제 벤치마크 (RTX 3080 12GB):**

```yaml
# Offload: True
- VRAM: 8.5GB
- 생성 시간 (3분 음악): 12초

# Offload: False
- VRAM: 11.2GB
- 생성 시간 (3분 음악): 10초
```

---

## Quantization & Compile

### Quantization (양자화)

**모델 가중치를 낮은 정밀도로 변환하여 메모리와 연산량 감소**

```python
# 기본 활성화 (자동)
quantization = True  # FP16 또는 INT8

# 효과:
# - VRAM 사용량: -20~30%
# - 생성 속도: +10~20% 빠름
# - 품질 손실: 거의 없음 (FP16)
```

**ACE-Step의 Quantization:**

```
FP32 (원본) → FP16 (기본) → INT8 (선택적)
  4 bytes      2 bytes       1 byte
```

### Compile (PyTorch 2.0+)

**PyTorch 모델을 최적화된 기계 코드로 컴파일**

```python
# 기본 활성화
compile = True  # torch.compile() 사용

# 효과:
# - 첫 실행: 느림 (컴파일 시간)
# - 이후 실행: +20~30% 빠름
# - VRAM: 약간 증가
```

**Compile 동작:**

```
┌──────────────────────────────────────────┐
│  첫 실행 (Warm-up)                        │
│  ────────────────────────                │
│  1. 모델 그래프 분석                       │
│  2. 최적화된 커널 생성                     │
│  3. 캐시 저장                             │
│  → 시간: +5~10초 (한 번만)                │
└──────────────────────────────────────────┘
          ↓
┌──────────────────────────────────────────┐
│  이후 실행 (Fast Path)                    │
│  ────────────────────────                │
│  • 캐시된 커널 재사용                      │
│  • 빠른 생성 (+20~30%)                    │
└──────────────────────────────────────────┘
```

### GPU별 Quantization & Compile 지원

| GPU | Quantization | Compile | 비고 |
|-----|--------------|---------|------|
| **NVIDIA CUDA** | ✅ FP16, INT8 | ✅ | 완전 지원 |
| **AMD ROCm** | ✅ FP16 | ⚠️ 제한적 | TORCH_COMPILE_BACKEND=eager |
| **Intel GPU** | ✅ FP16 | ✅ | 기본 활성화 |
| **MPS (Apple)** | ✅ FP16 | ⚠️ 제한적 | macOS 최적화 |

---

## 배치 크기 조정

### Batch Size란?

**한 번에 생성할 음악 트랙 개수**

```python
# Batch Size 1 (기본)
generate(prompt, batch_size=1)
# → 1개 트랙 생성

# Batch Size 4
generate(prompt, batch_size=4)
# → 4개 트랙 동시 생성 (다른 랜덤 시드)
```

### VRAM별 권장 Batch Size

| VRAM | LM 사용 | LM 미사용 | 비고 |
|------|---------|-----------|------|
| **≤4GB** | - | 1 | 최소 VRAM |
| **4-6GB** | - | 1 | DiT only |
| **6-8GB** | 1 | 2 | 0.6B LM |
| **8-12GB** | 2 | 4 | 0.6B LM |
| **12-16GB** | 2 | 4 | 1.7B LM |
| **16-24GB** | 4 | 8 | 4B LM |
| **≥24GB** | 8 | 8 | 무제한 |

### Batch 생성 전략

**빠른 변형 탐색:**

```python
# 1. 높은 Batch로 여러 변형 생성
results = generate(
    prompt="epic cinematic trailer music",
    batch_size=4,
    seed=None  # 랜덤 시드
)

# 2. 가장 좋은 결과 선택
best_result = results[2]

# 3. Cover로 세밀 조정
refined = cover(
    reference_audio=best_result,
    prompt="add more brass and percussion"
)
```

---

## Duration 제한 (VRAM별)

### Constrained Decoding

**ACE-Step은 GPU Tier에 따라 Duration을 자동 제한합니다.**

```python
# 내부 로직 예시
if vram <= 4000:
    max_duration = 180  # 3분
elif vram <= 6000:
    max_duration = 360  # 6분
elif vram <= 8000:
    max_duration = 240 if lm_enabled else 360  # 4분/6분
# ...
```

### Duration 초과 시 동작

```python
# 사용자 요청: 10분 (600초)
# GPU Tier 3 (6-8GB): 최대 4분

# ACE-Step 동작:
# 1. 경고 메시지 출력
print("Warning: Requested duration 600s exceeds GPU limit 240s")

# 2. 자동으로 제한
actual_duration = min(requested_duration, max_duration)  # 240초

# 3. 생성
generate(duration=actual_duration)
```

### Duration vs VRAM 사용량

**예상 VRAM 사용량 (DiT 생성 시):**

```yaml
30초: ~2GB
1분: ~3GB
2분: ~4GB
3분: ~5GB
4분: ~6GB
6분: ~8GB
8분: ~10GB
10분: ~12GB
```

---

## CUDA, ROCm, Intel GPU, MPS 설정

### NVIDIA CUDA

**기본 설정 (자동 최적화):**

```bash
# 기본 실행
uv run acestep

# GPU 선택 (여러 GPU 있을 때)
CUDA_VISIBLE_DEVICES=0 uv run acestep  # 첫 번째 GPU
CUDA_VISIBLE_DEVICES=1 uv run acestep  # 두 번째 GPU
```

**고급 CUDA 설정:**

```python
# 환경변수
export CUDA_LAUNCH_BLOCKING=1  # 디버깅 시
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 30xx 시리즈

# PyTorch 설정
import torch
torch.backends.cudnn.benchmark = True  # 성능 향상
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 활성화 (Ampere+)
```

### AMD ROCm

**ROCm 설치 워크플로우:**

```bash
# 1. 가상환경 생성
python -m venv .venv
source .venv/bin/activate

# 2. ROCm PyTorch 설치
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# 3. ACE-Step 설치
pip install -e .

# 4. 환경변수 설정 (RDNA3)
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # RX 7900 XT/XTX, RX 9070 XT
export MIOPEN_FIND_MODE=FAST
export TORCH_COMPILE_BACKEND=eager
export ACESTEP_LM_BACKEND=pt

# 5. 실행
python -m acestep.acestep_v15_pipeline --port 7680
```

**GPU별 환경변수:**

| GPU | HSA_OVERRIDE_GFX_VERSION |
|-----|--------------------------|
| **RX 7900 XT/XTX** | 11.0.0 |
| **RX 7800 XT** | 11.0.1 |
| **RX 7600** | 11.0.2 |
| **RX 6900 XT** | 10.3.0 |
| **RX 9070 XT** | 11.0.0 |

**Windows ROCm:**

```batch
REM start_gradio_ui_rocm.bat 사용 (자동 환경변수 설정)
start_gradio_ui_rocm.bat
```

### Intel GPU

**지원 현황:**

```yaml
플랫폼: Windows (테스트됨)
테스트 기기: Ultra 9 285H 내장 그래픽
PyTorch: 2.8.0 (Intel Extension for PyTorch)
기능:
  - LLM 추론: ✅ (0.6B 모델 테스트)
  - DiT 생성: ✅
  - Offload: ❌ (기본 비활성화)
  - Compile: ✅
  - Quantization: ✅
제한사항:
  - 2분 이상 음악 생성 시 LLM 추론 속도 저하
  - nanovllm 가속 미지원
```

**Intel GPU 설정:**

```bash
# 1. Intel Extension for PyTorch 설치
pip install torch torchvision --index-url https://pytorch-extension.intel.com/

# 2. ACE-Step 실행
uv run acestep --lm_model_path acestep-5Hz-lm-0.6B
```

### Apple MPS (Metal Performance Shaders)

**macOS Apple Silicon 최적화:**

```bash
# 기본 실행 (자동 MPS 감지)
uv run acestep

# MLX 백엔드 사용 (Apple Silicon 네이티브)
uv run acestep --backend mlx
```

**MPS vs CPU 성능:**

| 기기 | CPU | MPS | 배속 |
|------|-----|-----|------|
| **M1 Pro** | 45초 | 12초 | 3.75x |
| **M2 Max** | 38초 | 9초 | 4.22x |
| **M3 Max** | 32초 | 7초 | 4.57x |

*3분 음악 생성 기준

---

## GPU 호환성 가이드

### GPU 진단 도구

```bash
# GPU 감지 테스트
python scripts/check_gpu.py
```

**출력 예시:**

```
=== GPU Detection Report ===
GPU Type: NVIDIA
GPU Name: NVIDIA GeForce RTX 3080
VRAM: 12288 MB
CUDA Version: 12.1
PyTorch Version: 2.8.0
Build Type: CUDA

Tier: Tier 5
Recommended LM Model: acestep-5Hz-lm-1.7B
Max Duration: 240s (with LM) / 360s (without LM)
Max Batch Size: 2 (with LM) / 4 (without LM)
```

### 디버그 모드: GPU 시뮬레이션

**다른 VRAM 환경 테스트:**

```bash
# 4GB GPU 시뮬레이션 (Tier 1)
MAX_CUDA_VRAM=4 uv run acestep

# 8GB GPU 시뮬레이션 (Tier 4)
MAX_CUDA_VRAM=8 uv run acestep

# 12GB GPU 시뮬레이션 (Tier 5)
MAX_CUDA_VRAM=12 uv run acestep

# 16GB GPU 시뮬레이션 (Tier 6)
MAX_CUDA_VRAM=16 uv run acestep
```

**사용 시나리오:**
- ✅ **테스트** - 고급 GPU에서 저급 Tier 동작 확인
- ✅ **개발** - GPU Tier 설정 검증
- ✅ **PR 제출 전** - 다양한 VRAM 환경 테스트

---

## 트러블슈팅

### GPU 감지 안 됨 (AMD ROCm)

**증상:**
```
No GPU detected, running on CPU
```

**해결 방법:**

```bash
# 1. GPU 진단 실행
python scripts/check_gpu.py

# 2. ROCm 설치 확인
rocm-smi  # GPU 목록 표시되어야 함

# 3. PyTorch ROCm 빌드 확인
python -c "import torch; print(f'ROCm: {torch.version.hip}')"

# 4. RDNA3 GPU 환경변수 설정
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# 5. 재실행
python -m acestep.acestep_v15_pipeline --port 7680
```

### CPU 전용 PyTorch 설치됨

**증상:**
```
Build type: CPU-only
torch.cuda.is_available() = False
```

**해결 방법:**

```bash
# NVIDIA GPU
pip uninstall torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu121

# AMD GPU (ROCm)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# 확인
python -c "import torch; print(torch.cuda.is_available())"  # True
```

### NVIDIA GPU 감지 안 됨 (CUDA)

**진단 순서:**

```bash
# 1. NVIDIA 드라이버 확인
nvidia-smi

# 실패 시: https://www.nvidia.com/download/index.aspx 에서 드라이버 설치

# 2. CUDA 버전 확인
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
nvidia-smi  # "CUDA Version: X.X" 확인

# 3. PyTorch CUDA 재설치
pip uninstall torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### WSL2 GPU 접근 문제

**NVIDIA GPU (WSL2):**

```bash
# 1. Windows에 NVIDIA 드라이버 설치 (WSL2 내부 아님)
# 2. WSL2에 CUDA 툴킷 설치
# 3. 가이드: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
```

**AMD GPU (WSL2):**
- ROCm WSL2 지원 제한적
- 권장: 네이티브 Linux 사용 또는 Windows에서 `start_gradio_ui_rocm.bat` 사용

### Out of Memory (OOM) 오류

**증상:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**해결 전략:**

```bash
# 1. DiT 전용 모드로 전환
export ACESTEP_INIT_LLM=false
uv run acestep

# 2. 작은 LM 모델 사용
uv run acestep --lm_model_path acestep-5Hz-lm-0.6B

# 3. CPU Offload 활성화
uv run acestep --offload_to_cpu true

# 4. Duration 줄이기
# Gradio UI에서: Duration 3분 이하로 설정

# 5. Batch Size 줄이기
# Gradio UI에서: Batch Size 1로 설정
```

---

## 성능 벤치마크

### 생성 속도 (3분 음악 기준)

| GPU | VRAM | LM 모델 | Offload | 시간 | 실시간 배속 |
|-----|------|---------|---------|------|-----------|
| **A100 80GB** | 80GB | 4B | No | 1.8s | 100x |
| **RTX 4090** | 24GB | 4B | No | 4.2s | 42x |
| **RTX 3090** | 24GB | 4B | No | 6.5s | 27x |
| **RTX 4080** | 16GB | 1.7B | No | 8.1s | 22x |
| **RTX 3080** | 12GB | 1.7B | Yes | 12.3s | 14x |
| **RTX 4060 Ti** | 8GB | 0.6B | Yes | 18.7s | 9x |
| **RTX 3060** | 6GB | DiT only | Yes | 15.2s | 11x |

*실시간 배속 = 180s (3분) / 생성시간

### VRAM 사용량 (3분 음악 생성)

| 설정 | LM 모델 | Offload | VRAM | 설명 |
|------|---------|---------|------|------|
| **최소** | None | - | 4.2GB | DiT only |
| **경량** | 0.6B | Yes | 6.8GB | 저VRAM 권장 |
| **표준** | 1.7B | Yes | 9.5GB | 균형잡힌 설정 |
| **고급** | 1.7B | No | 11.2GB | 빠른 생성 |
| **최대** | 4B | No | 15.7GB | 최고 품질 |

### Duration별 생성 시간 (RTX 3090, 1.7B LM)

| Duration | VRAM | 생성 시간 | 비고 |
|----------|------|----------|------|
| **30초** | 3.2GB | 2.1s | 빠른 테스트 |
| **1분** | 4.1GB | 3.5s | 짧은 루프 |
| **2분** | 5.8GB | 5.2s | 표준 길이 |
| **3분** | 7.2GB | 6.5s | 풀송 (권장) |
| **4분** | 8.9GB | 8.3s | 긴 곡 |
| **6분** | 12.1GB | 11.7s | 확장 버전 |
| **8분** | 15.6GB | 15.2s | Tier 6 전용 |
| **10분** | 18.9GB | 19.1s | Tier 7 전용 |

---

## 커뮤니티 기여

### GPU 설정 개선 PR

**ACE-Step은 커뮤니티 피드백을 환영합니다!**

```python
# acestep/gpu_config.py
# 현재 설정이 여러분의 GPU에서 차선책이라면,
# 더 나은 파라미터를 테스트하고 PR을 제출해주세요!

# 예: RTX 4060 Ti 8GB에서 4분 생성 가능 확인
if vram == 8000:
    max_duration = 240  # 현재: 4분
    # → 테스트 결과 5분도 안정적이라면
    max_duration = 300  # PR 제출!
```

**기여 가이드라인:**

1. **테스트 환경 명시**
   - GPU 모델
   - VRAM 크기
   - OS 및 드라이버 버전

2. **반복 테스트**
   - 최소 10회 이상 생성 테스트
   - OOM 없이 안정적으로 동작 확인

3. **PR 제출**
   - `acestep/gpu_config.py` 수정
   - 테스트 결과 포함
   - 커뮤니티 개선에 기여!

---

## 다음 단계

**이 가이드에서는:**
- ✅ VRAM별 최적화 전략 이해
- ✅ DiT 전용 vs LLM 모드 비교
- ✅ CPU Offload, Quantization, Compile 활용
- ✅ 배치 크기 및 Duration 제한 파악
- ✅ CUDA, ROCm, Intel GPU, MPS 설정
- ✅ GPU 호환성 및 트러블슈팅

**다음 글에서는:**
- 🎉 **ACE-Step 1.5 결론 및 활용**
- 🎯 실제 활용 시나리오 (콘텐츠 제작, 음악 프로듀싱)
- 🌐 커뮤니티 리소스 및 다음 단계
- 🎵 "음악을 Play(연주/놀이)하세요" 최종 메시지

---

*GPU 최적화를 통해 ACE-Step 1.5의 성능을 최대한 끌어내세요. 여러분의 하드웨어에 맞는 최적 설정을 찾는 것이 창의적 워크플로우의 시작입니다!*
