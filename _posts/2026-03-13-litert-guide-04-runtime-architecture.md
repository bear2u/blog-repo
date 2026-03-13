---
layout: post
title: "LiteRT 완벽 가이드 (04) - 런타임 구성요소"
date: 2026-03-13
permalink: /litert-guide-04-runtime-architecture/
author: google-ai-edge
categories: [개발 도구, litert]
tags: [Trending, GitHub, litert, Runtime, Bindings, Architecture, GitHub Trending]
original_url: "https://github.com/google-ai-edge/LiteRT"
excerpt: "litert/README.md의 Directory Map과 루트 디렉토리 구성을 근거로 런타임/컴파일러/바인딩/툴 체계를 정리합니다."
---

## 이 문서의 목적

- LiteRT 소스 트리에서 “어디에 무엇이 있는지”를 개발자 관점으로 빠르게 지도화합니다.
- C/C++ API와 각 언어 바인딩, runtime/compiler 경계를 분리합니다.

---

## 빠른 요약(litert/README.md 기준)

`litert/README.md`는 Directory Map을 통해 주요 구성요소를 설명합니다.

- `c/`: ABI stable C API(헤더/구현)
- `cc/`: 공개 C++ API(ABI stable은 아님)
- `core/`: compiler/runtime 공유 내부 코드(예: schema.fbs)
- `compiler/`: 컴파일러 내부 코드(`lrt::internal`)
- `runtime/`: 런타임 내부 코드(`lrt::internal`)
- `python/`, `kotlin/`, `js/`: 언어 바인딩
- `tools/`: 벤치마크/정확도 평가/플러그인 CLI 등
- `vendors/`: SoC 벤더별 코드

근거:
- `litert/README.md`

---

## 1) “공개 API”와 “내부 구현” 경계

문서 기준으로:

- App 개발자가 직접 쓰는 경계: `c/`, `cc/`
- 내부 구현/공유: `core/`, `runtime/`, `compiler/` (namespace `lrt::internal` 언급)

---

## 2) 루트 레벨에서 함께 보이는 큰 덩어리

레포 루트에는 다음이 함께 존재합니다.

- `litert/`: LiteRT 핵심
- `tflite/`: TensorFlow Lite 관련(역사적/호환성/구성요소가 함께 있는 구조로 보임)
- `third_party/`: 의존성(서브모듈 포함)

근거:
- 레포 디렉토리 구성
- `.gitmodules`(third_party/tensorflow)

---

## 런타임/컴파일러/바인딩 관계(개략)

```mermaid
flowchart TB
  subgraph Public[Public APIs]
    C[c/ (C API)]
    CPP[cc/ (C++ API)]
  end

  subgraph Internal[Internal]
    CORE[core/]
    RT[runtime/]
    COMP[compiler/]
  end

  subgraph Bindings[Language bindings]
    PY[python/]
    KT[kotlin/]
    JS[js/]
  end

  C --> CORE
  CPP --> CORE
  CORE --> RT
  CORE --> COMP
  PY --> C
  KT --> C
  JS --> C
```

---

## 주의사항/함정

- “compiled model”/“runtime libraries” 같은 빌드 타깃은 Bazel/CMake 문서와 함께 읽어야 정확합니다. (근거: 루트 `README.md`가 두 문서 링크를 제공)
- `tflite/`와 `litert/`의 역할 분리는 문서/코드 기준으로 확인이 필요합니다(이 챕터는 디렉토리 맵 중심).

---

## TODO / 확인 필요

- `tools/benchmark_model` 같은 실행 파일(문서에 예시)은 빌드 시스템별 산출 경로가 다르므로, 실제 산출 경로/사용법을 “플랫폼별”로 정리하면 좋습니다.

---

## 위키 링크

- `[[LiteRT Guide - Index]]` → [가이드 목차](/blog-repo/litert-guide/)
- `[[LiteRT Guide - Build Systems]]` → [03. 빌드 시스템 개요](/blog-repo/litert-guide-03-build-systems/)
- `[[LiteRT Guide - Ops]]` → [05. 운영/최적화/트러블슈팅](/blog-repo/litert-guide-05-ops-and-troubleshooting/)

---

*다음 글에서는 docker_build/README와 build instructions의 Troubleshooting 힌트를 바탕으로 운영/장애 대응 체크리스트를 만듭니다.*

