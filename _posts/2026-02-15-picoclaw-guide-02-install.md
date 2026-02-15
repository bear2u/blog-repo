---
layout: post
title: "PicoClaw 가이드 (02) - 설치(바이너리/소스): 릴리즈, make build/install, 멀티 아키텍처"
date: 2026-02-15
permalink: /picoclaw-guide-02-install/
author: Sipeed
categories: [AI 에이전트, 개발 도구]
tags: ["PicoClaw", "Install", "Go", "Make", "Release"]
original_url: "https://github.com/sipeed/picoclaw#-install"
excerpt: "릴리즈 바이너리로 빠르게 설치하거나, 소스에서 make로 빌드/설치하는 방법을 정리합니다. 멀티 플랫폼 빌드와 기본 산출물 경로도 함께 봅니다."
---

## 방법 1) 릴리즈(프리컴파일) 바이너리

README는 가장 간단한 설치로 “릴리즈 페이지의 프리컴파일 바이너리”를 안내합니다.

- 릴리즈: `https://github.com/sipeed/picoclaw/releases`

운영 환경에서는 보통 이 경로가 가장 예측 가능하고, “소스 빌드에 따른 차이”가 줄어듭니다.

---

## 방법 2) 소스에서 빌드(개발자 권장)

README의 소스 빌드 흐름:

```bash
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw

make deps
make build
```

`Makefile`을 보면 `build`는 내부적으로 `go generate ./...`를 먼저 수행하고, 현재 OS/아키텍처에 맞는 바이너리를 `build/` 아래에 생성합니다.

---

## 설치: `make install`

`make install`은 기본값 기준으로:

- 바이너리를 `~/.local/bin/picoclaw`에 설치하고
- 실행 권한을 부여합니다.

설치 경로는 `INSTALL_PREFIX`로 조정할 수 있습니다.

---

## 멀티 플랫폼 빌드: `make build-all`

README와 `Makefile`에는 멀티 플랫폼 빌드가 포함돼 있습니다.

- linux/amd64, linux/arm64, linux/riscv64
- darwin/arm64
- windows/amd64

배포용 빌드 아티팩트를 한 번에 만들고 싶으면 이 타깃이 유용합니다.

다음 장에서는 설치 후 `picoclaw onboard`로 설정/워크스페이스를 초기화하고, 첫 `agent` 실행까지 진행합니다.

