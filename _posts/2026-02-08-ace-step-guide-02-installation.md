---
layout: post
title: "ACE-Step 1.5 완벽 가이드 (02) - 설치 및 시작"
date: 2026-02-08
permalink: /ace-step-guide-02-installation/
author: ACE Studio & StepFun
categories: [AI 음악, 오픈소스]
tags: [ACE-Step, Installation, Setup, Windows, Linux, macOS, GPU]
original_url: "https://github.com/ace-step/ACE-Step-1.5"
excerpt: "플랫폼별 설치 방법과 GPU 환경 설정"
---

## 요구사항

### 시스템 요구사항

```yaml
OS: Windows 10+, Linux, macOS
Python: 3.11 (정확히)
GPU: CUDA GPU 권장 (AMD ROCm, Intel GPU, MPS 지원)
Disk: 15-20GB (모델 포함)
```

### GPU & VRAM 권장 사양

| VRAM | 설정 | 예상 성능 |
|------|------|-----------|
| **≤6GB** | DiT only (LLM 비활성화) | 빠름, 기본 품질 |
| **6-12GB** | LM-0.6B + offload | 중간 품질 |
| **12-16GB** | LM-1.7B | 좋은 품질 |
| **16GB+** | LM-4B + 배치 생성 | 최고 품질 |

---

## Windows 설치 (권장: Portable 패키지)

### 방법 1: Windows Portable 패키지 (가장 쉬움)

```batch
# 1. 다운로드
https://files.acemusic.ai/acemusic/win/ACE-Step-1.5.7z

# 2. 압축 해제
ACE-Step-1.5.7z를 원하는 폴더에 압축 해제

# 3. 실행
start_gradio_ui.bat    # Gradio UI 시작
start_api_server.bat   # REST API 서버 시작
```

**패키지 구성:**

```
ACE-Step-1.5/
├── python_embeded/          # 내장 Python 3.11
│   ├── python.exe
│   └── Lib/ (모든 의존성 사전 설치)
├── start_gradio_ui.bat      # UI 실행 스크립트
├── start_api_server.bat     # API 실행 스크립트
├── check_update.bat         # Git 업데이트
├── merge_config.bat         # 설정 병합
└── PortableGit/ (선택)      # 업데이트 기능용
```

### Portable 패키지 스크립트 설정

#### start_gradio_ui.bat 커스터마이징

```batch
REM UI 언어 (en, zh, he, ja, ko)
set LANGUAGE=ko

REM 다운로드 소스 (auto, huggingface, modelscope)
set DOWNLOAD_SOURCE=--download-source auto

REM Git 업데이트 체크 (true/false)
set CHECK_UPDATE=true

REM 모델 설정
set CONFIG_PATH=--config_path acestep-v15-turbo
set LM_MODEL_PATH=--lm_model_path acestep-5Hz-lm-1.7B

REM LLM 초기화 (auto/true/false)
REM Auto: VRAM > 6GB이면 활성화, 아니면 비활성화
REM set INIT_LLM=--init_llm true   # 강제 활성화 (OOM 위험)
REM set INIT_LLM=--init_llm false  # 강제 비활성화 (DiT 전용)
```

#### start_api_server.bat 커스터마이징

```batch
REM LLM 초기화 (환경변수)
REM set ACESTEP_INIT_LLM=true   # LLM 강제 활성화
REM set ACESTEP_INIT_LLM=false  # DiT 전용 모드

REM LM 모델 경로 (선택)
REM set LM_MODEL_PATH=--lm-model-path acestep-5Hz-lm-0.6B
```

### 업데이트 & 유지보수

```batch
# 업데이트 확인 (PortableGit 필요)
check_update.bat

# 설정 충돌 시 병합
merge_config.bat

# 환경 테스트
quick_test.bat

# uv 재설치 (필요시)
install_uv.bat
```

---

## Linux 설치

### 표준 설치 (CUDA)

```bash
# 1. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 레포지토리 클론
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5

# 3. 의존성 설치
uv sync

# 4. 실행
uv run acestep  # Gradio UI
```

### AMD ROCm GPU 설치

**중요:** `uv run acestep`는 CUDA PyTorch를 설치하므로 ROCm 설정을 덮어씁니다.

```bash
# 1. 가상환경 생성
python -m venv .venv
source .venv/bin/activate

# 2. ROCm PyTorch 설치
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# 3. ACE-Step 설치 (uv 없이)
pip install -e .

# 4. 직접 실행
python -m acestep.acestep_v15_pipeline --port 7680
```

#### RDNA3 GPU (RX 7000/9000 시리즈) 설정

```bash
# GPU 감지 문제 시 환경변수 설정
# RX 7900 XT/XTX, RX 9070 XT
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# RX 7800 XT, RX 7700 XT
export HSA_OVERRIDE_GFX_VERSION=11.0.1

# RX 7600
export HSA_OVERRIDE_GFX_VERSION=11.0.2

# GPU 진단 도구 실행
python scripts/check_gpu.py

# ROCm 설치 확인
rocm-smi
```

### Python 3.11 주의사항 (Ubuntu)

Ubuntu는 Python 3.11.0rc1 (프리릴리스)를 제공하는데, vLLM 백엔드에서 세그멘테이션 폴트를 일으킬 수 있습니다.

```bash
# 안정 버전 설치 (≥ 3.11.12 권장)
# deadsnakes PPA 사용
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11

# 또는 PyTorch 백엔드 사용
uv run acestep --backend pt
```

---

## macOS 설치

### Apple Silicon (M1/M2/M3)

```bash
# 1. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 클론 & 설치
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync

# 3. 실행 (MPS 가속)
uv run acestep
```

**macOS 전용 기능:**

- **MLX 지원** - Apple Silicon 네이티브 가속
- **MPS (Metal Performance Shaders)** - GPU 가속

```python
# MLX 의존성 (자동 설치됨)
mlx>=0.25.2
mlx-lm>=0.20.0
```

---

## 모델 다운로드

### 자동 다운로드 (권장)

첫 실행 시 자동으로 다운로드됩니다.

```bash
uv run acestep
# → 모델 없으면 자동 다운로드 시작
```

### 수동 다운로드 (CLI)

```bash
# 메인 모델 다운로드
uv run acestep-download

# 모든 모델 다운로드
uv run acestep-download --all

# ModelScope에서 다운로드
uv run acestep-download --download-source modelscope

# HuggingFace에서 다운로드
uv run acestep-download --download-source huggingface

# 특정 모델만 다운로드
uv run acestep-download --model acestep-v15-sft

# 사용 가능한 모델 목록
uv run acestep-download --list

# 커스텀 디렉토리에 다운로드
uv run acestep-download --dir /path/to/checkpoints
```

### huggingface-cli 사용

```bash
# 메인 모델 (vae, Qwen3-Embedding-0.6B, acestep-v15-turbo, acestep-5Hz-lm-1.7B)
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ./checkpoints

# 선택적 LM 모델
huggingface-cli download ACE-Step/acestep-5Hz-lm-0.6B --local-dir ./checkpoints/acestep-5Hz-lm-0.6B
huggingface-cli download ACE-Step/acestep-5Hz-lm-4B --local-dir ./checkpoints/acestep-5Hz-lm-4B

# 선택적 DiT 모델
huggingface-cli download ACE-Step/acestep-v15-base --local-dir ./checkpoints/acestep-v15-base
huggingface-cli download ACE-Step/acestep-v15-sft --local-dir ./checkpoints/acestep-v15-sft
```

### 사용 가능한 모델

| 모델 | 설명 | 크기 | 용도 |
|------|------|------|------|
| **acestep-v15-turbo** | 기본 DiT 모델 | ~2GB | 일반 생성 (권장) |
| **acestep-v15-base** | Base DiT | ~2GB | Fine-tuning 용이 |
| **acestep-v15-sft** | SFT DiT | ~2GB | 높은 품질 |
| **acestep-5Hz-lm-0.6B** | 경량 LM | ~600MB | 6-12GB VRAM |
| **acestep-5Hz-lm-1.7B** | 표준 LM | ~1.7GB | 12-16GB VRAM |
| **acestep-5Hz-lm-4B** | 대형 LM | ~4GB | 16GB+ VRAM |

---

## 첫 실행

### Gradio UI 시작

```bash
# 기본 실행
uv run acestep

# 옵션 포함 실행
uv run acestep \
  --port 7860 \
  --server-name 0.0.0.0 \
  --language ko \
  --init_service true \
  --config_path acestep-v15-turbo \
  --lm_model_path acestep-5Hz-lm-1.7B
```

**주요 옵션:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--port` | 7860 | 서버 포트 |
| `--server-name` | 127.0.0.1 | 서버 주소 (0.0.0.0 = 네트워크 접근) |
| `--language` | en | UI 언어 (en, zh, he, ja, ko) |
| `--init_service` | false | 시작 시 모델 자동 초기화 |
| `--init_llm` | auto | LLM 초기화 (true/false/auto) |
| `--offload_to_cpu` | auto | CPU 오프로드 (VRAM < 16GB 시 자동) |
| `--download-source` | auto | 다운로드 소스 |

### REST API 서버 시작

```bash
# 기본 실행
uv run acestep-api

# 인증 포함 실행
uv run acestep-api --api-key sk-your-secret-key
```

브라우저에서 `http://localhost:7860` (Gradio) 또는 `http://localhost:8001` (API) 접속.

---

## 환경변수 설정 (.env)

```bash
# .env.example 복사
cp .env.example .env

# .env 편집
nano .env
```

**.env 예시:**

```bash
# LLM 초기화 모드
ACESTEP_INIT_LLM=auto  # auto, true, false

# 모델 경로
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B

# 다운로드 소스
ACESTEP_DOWNLOAD_SOURCE=auto  # auto, huggingface, modelscope

# API 인증
ACESTEP_API_KEY=sk-your-secret-key
```

**ACESTEP_INIT_LLM 처리 흐름:**

```
GPU 감지 (전체) → ACESTEP_INIT_LLM 오버라이드 → 모델 로딩
```

| 값 | 동작 |
|----|------|
| `auto` (또는 빈 값) | GPU 자동 감지 결과 사용 (권장) |
| `true` / `1` / `yes` | GPU 감지 후 LLM 강제 활성화 (OOM 위험) |
| `false` / `0` / `no` | 순수 DiT 모드 강제, 더 빠른 생성 |

---

## GPU별 최적화 설정

### ≤6GB VRAM (RTX 3060, GTX 1660 Ti)

```bash
# DiT 전용 모드 (LLM 비활성화)
export ACESTEP_INIT_LLM=false
uv run acestep --config_path acestep-v15-turbo
```

### 6-12GB VRAM (RTX 3060 Ti, RTX 4060)

```bash
# 경량 LM 모델 + CPU 오프로드
uv run acestep \
  --lm_model_path acestep-5Hz-lm-0.6B \
  --offload_to_cpu true
```

### 12-16GB VRAM (RTX 3080, RTX 4070)

```bash
# 표준 LM 모델
uv run acestep \
  --lm_model_path acestep-5Hz-lm-1.7B \
  --config_path acestep-v15-turbo
```

### 16GB+ VRAM (RTX 3090, RTX 4090, A100)

```bash
# 대형 LM 모델 + 배치 생성
uv run acestep \
  --lm_model_path acestep-5Hz-lm-4B \
  --config_path acestep-v15-turbo
```

---

## 인증 설정

### Gradio 인증

```bash
uv run acestep \
  --auth-username admin \
  --auth-password password123
```

### API 인증

```bash
uv run acestep \
  --enable-api \
  --api-key sk-your-secret-key
```

### 동시 인증 (Gradio + API)

```bash
uv run acestep \
  --enable-api \
  --api-key sk-api-123456 \
  --auth-username admin \
  --auth-password gradio-pass
```

---

## 문제 해결

### GPU 감지 안 됨 (AMD ROCm)

```bash
# GPU 진단 도구 실행
python scripts/check_gpu.py

# ROCm 설치 확인
rocm-smi

# RDNA3 GPU 환경변수 설정
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### Python 3.11.0rc1 세그멘테이션 폴트 (Ubuntu)

```bash
# PyTorch 백엔드 사용
uv run acestep --backend pt

# 또는 안정 버전 설치
sudo apt install python3.11
```

### Windows에서 uv 설치 실패

```powershell
# PowerShell로 수동 설치
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 또는 Portable 패키지 사용 (python_embeded 내장)
start_gradio_ui.bat
```

---

## 다음 단계

**이 가이드에서는:**
- ✅ 플랫폼별 설치 완료
- ✅ GPU 환경 설정
- ✅ 모델 다운로드
- ✅ 첫 실행 성공

**다음 글에서는:**
- 🏗️ ACE-Step 1.5 아키텍처 심층 분석
- 🧠 DiT + LM 하이브리드 구조 이해
- 📊 Model Zoo 및 모델 선택 가이드

---

*이제 ACE-Step 1.5가 준비되었습니다. 다음 글에서 아키텍처를 이해하고 효과적으로 활용하는 방법을 배워봅시다!*
