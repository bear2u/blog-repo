---
layout: post
title: "NautilusTrader 가이드 (09) - 정밀도/릴리즈/검증: 64/128-bit와 빌드 provenance"
date: 2026-02-17
permalink: /nautilus-trader-guide-09-precision-release-provenance/
author: Nautech Systems
categories: [개발 도구, 퀀트 트레이딩]
tags: [NautilusTrader, Precision, Release, Supply Chain, Attestation]
original_url: "https://github.com/nautechsystems/nautilus_trader"
excerpt: "NautilusTrader는 standard(64-bit)와 high(128-bit) precision 모드를 제공합니다. 브랜치별 릴리즈 채널과 SLSA 기반 attestation 검증 흐름까지 함께 정리합니다."
---

## 정밀도 모드(Price/Quantity/Money)

README 기준 두 가지 모드:

- **standard precision**: 64-bit
- **high precision**: 128-bit

공식 Python wheel은 플랫폼별 제약이 있으며(예: Windows wheel의 128-bit 제약 언급), Rust 크레이트는 feature flag로 고정밀 모드를 켤 수 있습니다.

---

## 브랜치/릴리즈 운영 감각

브랜치 성격:

- `master`: 릴리즈 기준
- `nightly`: 일일 스냅샷
- `develop`: 개발 중심

실무 권장:

- 연구/사전검증 환경에서 nightly/develop로 기능 선검증
- 실거래 환경은 stable 중심으로 단계적 업데이트

---

## 공급망 보안: attestation

설치 문서는 배포 아티팩트에 대한 provenance 검증 흐름을 제공합니다.

- GitHub CLI `gh attestation verify ...`
- 공식 워크플로우에서 빌드된 아티팩트인지 검증

금융 시스템 특성상 “패키지 신뢰성”은 기능 못지않게 중요하므로, CI 파이프라인에 검증 단계를 넣는 편이 좋습니다.

다음 장에서는 운영 중 자주 부딪히는 문제와 안전 체크리스트를 정리합니다.

