---
layout: post
title: "BitNet 완벽 가이드 (04) - 구성요소/아키텍처"
date: 2026-03-13
permalink: /bitnet-guide-04-architecture/
author: microsoft
categories: [AI, bitnet]
tags: [Trending, GitHub, bitnet, LLM, Inference, Architecture, GitHub Trending]
original_url: "https://github.com/microsoft/BitNet"
excerpt: "루트 디렉토리 맵과 .gitmodules(llama.cpp), gpu/README를 근거로 BitNet 레포의 구성요소를 정리합니다."
---

## 이 문서의 목적

- “BitNet 레포가 무엇을 포함하고, 어떤 식으로 실행에 연결되는지”를 **파일/디렉토리 기준으로** 정리합니다.
- 루트(bitnet.cpp)와 GPU 커널(`gpu/`)을 한 그림에서 보이게 합니다.

---

## 빠른 요약 (디렉토리 중심)

루트 기준 상위 디렉토리(예: `src/`, `gpu/`, `utils/`)가 기능 단위로 나뉘어 있습니다. (`README.md`, 레포 트리)

추론 실행 스크립트가 루트에 존재합니다.

- `setup_env.py`: 모델/환경 준비(README 예시)
- `run_inference.py`: 단일 실행(README 예시 + argparse)
- `run_inference_server.py`: 서버 실행(argparse + build/bin/llama-server 실행)

그리고 `3rdparty/llama.cpp`는 서브모듈로 포함됩니다. (`.gitmodules`)

---

## 구성요소 지도

### 1) 루트(bitnet.cpp)

- `src/`, `include/`: CPU 측 구현/헤더(구체 클래스/엔트리포인트는 추가 확인 필요)
- `preset_kernels/`: pretuned 커널 파라미터/프리셋(이름 그대로 “미리 튜닝된” 값일 가능성)
- `utils/`: 벤치마크/변환 스크립트(예: `utils/e2e_benchmark.py`, `utils/convert-helper-bitnet.py`가 README에 언급)
- `docs/`: 추가 문서

### 2) 서브모듈: `3rdparty/llama.cpp`

`.gitmodules`에 `3rdparty/llama.cpp`가 정의되어 있으며, `run_inference_server.py`는 `build/bin/llama-server` 산출물 실행을 가정합니다.

### 3) GPU 커널: `gpu/`

`gpu/README.md`는 **W2A8 GEMV 커널**(커스텀 CUDA 커널)과 변환/생성 파이프라인을 별도 흐름으로 제공합니다.

---

## 런타임 관점 아키텍처(개략)

```mermaid
flowchart LR
  subgraph Root[Repo Root]
    A[setup_env.py] --> B[models/... gguf]
    C[run_inference.py]
    D[run_inference_server.py]
    E[utils/* benchmark/convert]
  end

  subgraph Llama[3rdparty/llama.cpp (submodule)]
    L1[llama-cli/llama-server]
  end

  subgraph GPU[gpu/]
    G1[bitnet_kernels/compile.sh]
    G2[test.py]
    G3[convert_* + generate.py]
  end

  B --> C --> L1
  B --> D --> L1
  G1 --> G2
  G3 --> GPU
```

---

## “어디를 먼저 보면 좋은가?”

- 설치/빌드/실행: `README.md`, `requirements.txt`, `CMakeLists.txt`, `setup_env.py`, `run_inference*.py`
- GPU 커널: `gpu/README.md`, `gpu/bitnet_kernels/`, `gpu/convert_checkpoint.py`
- 벤치마크/도구: `utils/e2e_benchmark.py`, `utils/generate-dummy-bitnet-model.py`

---

## 주의사항/함정

- `run_inference_server.py`는 빌드 산출물 경로(`build/bin/llama-server`)에 강하게 의존합니다.
- 루트(gguf 기반)와 `gpu/`(체크포인트 변환 + 커널) 흐름은 **서로 다른 실행 루트**이므로, 섞어서 따라가면 “파일이 없다/커맨드가 다르다” 류의 혼란이 생깁니다.

---

## TODO / 확인 필요

- `src/` 내부의 “추론 엔진/핵심 커널 호출 지점”은 본문에서 디렉토리 단위로만 정리했습니다. 실제 데이터 플로우(토크나이즈→디코딩→커널 호출)는 `src/`와 `3rdparty/llama.cpp`의 엔트리포인트를 추가로 읽고 Mermaid 시퀀스로 확정하는 것이 좋습니다.

---

## 위키 링크

- `[[BitNet Guide - Index]]` → [가이드 목차](/blog-repo/bitnet-guide/)
- `[[BitNet Guide - Build & Install]]` → [02. 설치 및 빌드](/blog-repo/bitnet-guide-02-build-and-install/)
- `[[BitNet Guide - Ops & Troubleshooting]]` → [05. 운영/최적화/트러블슈팅](/blog-repo/bitnet-guide-05-ops-and-troubleshooting/)

---

*다음 글에서는 README의 FAQ/벤치마크 스크립트/서버 실행을 기반으로 운영 체크리스트를 만듭니다.*

