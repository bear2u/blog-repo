---
layout: post
title: "NautilusTrader 가이드 (02) - 설치/배포 채널: PyPI, Nautech 인덱스, stable/nightly/develop"
date: 2026-02-17
permalink: /nautilus-trader-guide-02-installation/
author: Nautech Systems
categories: [개발 도구, 퀀트 트레이딩]
tags: [NautilusTrader, Installation, PyPI, uv, Wheels, Nightly]
original_url: "https://nautilustrader.io/docs/latest/getting_started/installation/"
excerpt: "공식 설치 가이드는 Python 3.12-3.14 + 64비트 플랫폼을 기준으로 합니다. PyPI와 Nautech 패키지 인덱스의 차이, nightly/develop 휠 사용 전략을 정리합니다."
---

## 지원 플랫폼/런타임

설치 문서 기준 공식 지원 축:

- Python 3.12-3.14
- 64비트 환경(Linux/macOS/Windows x86_64/ARM64 일부)

Linux는 glibc 버전 요구사항(2.35+)도 확인이 필요합니다.

---

## 기본 설치: `uv pip install`

문서에서 권장하는 기본 흐름:

```bash
uv pip install nautilus_trader
```

특정 통합 의존성이 필요하면 extras를 사용합니다.

```bash
uv pip install "nautilus_trader[docker,ib]"
```

---

## Nautech package index 사용

NautilusTrader는 자체 인덱스(`packages.nautechsystems.io`)도 제공합니다.

- stable 휠
- `nightly` / `develop` 기반 pre-release 휠

예:

```bash
uv pip install nautilus_trader --pre --index-url=https://packages.nautechsystems.io/simple
```

운영 관점 권장:

- 실거래: stable 우선
- 사전 검증/테스트 베드: nightly/develop

---

## 브랜치/버전 감각

README/설치 문서 기준:

- `master`: 최신 릴리즈 소스
- `nightly`: `develop` 기반 일일 스냅샷
- `develop`: 활성 개발 브랜치

즉, 기능 실험 속도와 운영 안정성 사이에서 채널을 분리해서 쓰는 것이 핵심입니다.

다음 장에서는 소스 빌드와 개발 환경을 정리합니다.

