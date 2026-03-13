---
layout: post
title: "BitNet 완벽 가이드 (02) - 설치 및 빌드"
date: 2026-03-13
permalink: /bitnet-guide-02-build-and-install/
author: microsoft
categories: [AI, bitnet]
tags: [Trending, GitHub, bitnet, LLM, Inference, Build, GitHub Trending]
original_url: "https://github.com/microsoft/BitNet"
excerpt: "README와 gpu/README를 근거로 bitnet.cpp의 CPU/GPU 빌드 경로와 필수 요구사항을 정리합니다."
---

## 이 문서의 목적

- BitNet(bitnet.cpp)을 **로컬에서 빌드/실행하기 위한 최소 요구사항**을 정리합니다.
- CPU 경로(루트 README)와 GPU 경로(`gpu/README.md`)를 **혼동하지 않도록** 분리해서 안내합니다.

---

## 빠른 요약 (무엇을 먼저 결정할까?)

- **CPU 경로(루트)**: `README.md`의 “Build from source” + `setup_env.py` + `run_inference.py` 흐름이 기본입니다.
- **GPU 커널 경로(gpu/)**: `gpu/README.md`의 “bitnet_kernels 빌드 + 테스트/생성(generate.py)” 흐름이 별도로 존재합니다.
- 서브모듈(예: `3rdparty/llama.cpp`)을 사용하므로, 클론 시 `--recursive`가 필요합니다. (`README.md`, `.gitmodules`)

---

## 요구사항(루트 CPU 경로)

`README.md` 기준으로 요구사항이 명시됩니다.

- Python: `python>=3.9`
- CMake: `cmake>=3.22`
- clang: `clang>=18`
- conda: “highly recommend”
- Windows: Visual Studio 2022 + C++/CMake/Clang 도구 설치를 권장(Developer Command Prompt/PowerShell 사용)

## 설치/빌드(루트 CPU 경로)

### 1) 클론

```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
```

### 2) Python 의존성

```bash
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp

pip install -r requirements.txt
```

### 3) 모델 다운로드 + 환경 구성(Quant 포함)

루트 README에는 Hugging Face에서 GGUF를 내려받고 `setup_env.py`로 환경을 맞추는 예시가 있습니다.

```bash
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

`setup_env.py --help`에 따르면, 주요 파라미터는 다음과 같습니다.

- `--model-dir/-md`: 모델 디렉토리
- `--quant-type/-q`: `i2_s` 또는 `tl1`
- `--use-pretuned/-p`: pretuned 커널 파라미터 사용

---

## GPU 경로(gpu/ 디렉토리)

GPU 커널은 루트와 별도의 README(`gpu/README.md`)에 설치/빌드/성능 테스트 흐름이 정리되어 있습니다.

### 1) (권장) 별도 conda env

```bash
conda create --name bitnet-gpu "python<3.13"
conda activate bitnet-gpu
pip install -r requirements.txt
```

### 2) 커널 빌드 + 테스트

```bash
cd bitnet_kernels
bash compile.sh
cd ..
python test.py
```

### 3) End-to-end 추론(변환 포함)

`gpu/README.md` 예시는 `convert_safetensors.py` / `convert_checkpoint.py`를 통해 체크포인트를 변환한 뒤 `generate.py`를 실행합니다.

---

## 빌드/설치 파이프라인(개략)

```mermaid
flowchart TD
  A[git clone --recursive] --> B[pip install -r requirements.txt]
  B --> C[huggingface-cli download ...]
  C --> D[python setup_env.py ...]
  D --> E[python run_inference.py ...]
  D --> F[python run_inference_server.py ...]
  A --> G[gpu/ (optional)]
  G --> H[bitnet_kernels compile.sh]
  H --> I[python test.py]
  G --> J[convert_* + generate.py]
```

---

## 주의사항/함정

- **서브모듈 누락**: `.gitmodules`에 `3rdparty/llama.cpp`가 명시되어 있으므로 `--recursive` 없이 클론하면 빌드/실행 단계에서 문제가 날 수 있습니다.
- **Windows 빌드**: 루트 README에서 VS2022 “Developer Command Prompt/PowerShell” 사용을 중요하게 강조합니다.
- **CPU/GPU 문서 혼동**: 루트 README와 `gpu/README.md`는 목적/흐름이 다릅니다(커널/체크포인트 변환/성능 테스트 포함 여부).

---

## TODO / 확인 필요

- 루트 경로에서 실제 바이너리(예: `build/bin/llama-server`)가 생성되는 과정은 `setup_env.py` 및 `CMakeLists.txt`의 내부 로직을 추가로 읽고 확인이 필요합니다.
- GPU 경로의 실제 요구 CUDA/toolchain 범위는 `gpu/README.md`와 `bitnet_kernels/` 빌드 스크립트 기준으로 재확인이 필요합니다.

---

## 위키 링크

- `[[BitNet Guide - Index]]` → [가이드 목차](/blog-repo/bitnet-guide/)
- `[[BitNet Guide - Quick Inference]]` → [03. 실행(추론) 빠른 시작](/blog-repo/bitnet-guide-03-quick-inference/)

---

*다음 글에서는 `run_inference.py` / `run_inference_server.py`를 기준으로 최소 실행 루트를 정리합니다.*

