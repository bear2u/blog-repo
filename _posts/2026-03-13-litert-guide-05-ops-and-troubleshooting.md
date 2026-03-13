---
layout: post
title: "LiteRT 완벽 가이드 (05) - 운영/최적화/트러블슈팅"
date: 2026-03-13
permalink: /litert-guide-05-ops-and-troubleshooting/
author: google-ai-edge
categories: [개발 도구, litert]
tags: [Trending, GitHub, litert, Troubleshooting, Bazel, Docker, GitHub Trending]
original_url: "https://github.com/google-ai-edge/LiteRT"
excerpt: "docker_build/README.md와 g3doc 빌드 문서를 근거로 빌드 실패/환경 이슈 대응 체크리스트를 정리합니다."
---

## 이 문서의 목적

- LiteRT 소스 빌드에서 가장 흔한 문제(환경 구성, 리소스 부족, 권한/마운트, 플랫폼별 설정)를 체크리스트로 정리합니다.
- docker_build(hermetic)와 로컬 Bazel/CMake 경로 각각의 트러블슈팅 포인트를 구분합니다.

---

## 빠른 요약(문서 근거)

- docker_build는 “Docker daemon 리소스/마운트 권한/로그 확인”을 Troubleshooting로 명시합니다. (`docker_build/README.md`)
- Bazel 빌드는 플랫폼별 의존성 설치/환경 변수 설정(특히 Windows)이 문서로 제공됩니다. (`g3doc/instructions/BUILD_INSTRUCTIONS.md`)
- 서브모듈(`third_party/tensorflow`)이 존재합니다. (`.gitmodules`)

---

## 1) docker_build 경로 Troubleshooting

`docker_build/README.md`가 제시하는 체크리스트:

1. Docker daemon에 충분한 RAM/CPU 할당
2. 현재 디렉토리 마운트 권한 확인
3. Docker 로그에서 구체 오류 확인

디버깅 쉘(문서 예시):

```bash
docker run --rm -it --user $(id -u):$(id -g) -e HOME=/litert_build -e USER=$(id -un) \
  -v $(pwd):/litert_build litert_build_env bash
```

---

## 2) 로컬 Bazel 빌드 이슈(플랫폼별)

### Linux

문서는 OpenJDK, clang/libc++ 등 의존성 설치 후 `./configure` → `bazel build ...` 흐름을 제시합니다. (`BUILD_INSTRUCTIONS.md`)

### Windows

문서는 `BAZEL_VC`, `BAZEL_SH` 환경 변수와 `.bazelrc.user` 설정 예시를 제공합니다. (`BUILD_INSTRUCTIONS.md`)

---

## 3) 서브모듈/의존성 이슈

`.gitmodules`에 `third_party/tensorflow` 서브모듈이 정의되어 있으므로, 소스 체크아웃 시 submodule 초기화가 필요합니다.

운영 체크:

- `git submodule status`로 초기화 여부 확인
- hermetic docker 빌드는 submodule 업데이트를 자동 처리한다고 문서에 설명이 있으나, 로컬 빌드 경로에서는 직접 확인 권장

근거:
- `.gitmodules`
- `docker_build/README.md`

---

## 4) 성능/검증(문서 예시 기준)

`BUILD_INSTRUCTIONS.md`는 `benchmark_model` 실행 예시를 제공합니다.

```bash
./bazel-bin/litert/tools/benchmark_model --model=path/to/model.tflite --num_threads=4
```

근거:
- `g3doc/instructions/BUILD_INSTRUCTIONS.md`

---

## TODO / 확인 필요

- “NPU 가속/벤더별 라이브러리 푸시” 같은 고급 운영 항목은 `BUILD_INSTRUCTIONS.md`의 Android 섹션(예: Qualcomm 예시)과 `litert/vendors/`를 함께 읽고, 디바이스별 체크리스트로 정리하는 것이 좋습니다.

---

## 위키 링크

- `[[LiteRT Guide - Index]]` → [가이드 목차](/blog-repo/litert-guide/)
- `[[LiteRT Guide - Docker Build]]` → [02. 소스 빌드(도커)](/blog-repo/litert-guide-02-build-with-docker/)
- `[[LiteRT Guide - Build Systems]]` → [03. 빌드 시스템 개요](/blog-repo/litert-guide-03-build-systems/)
- `[[LiteRT Guide - Runtime Architecture]]` → [04. 런타임 구성요소](/blog-repo/litert-guide-04-runtime-architecture/)

