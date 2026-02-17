---
layout: page
title: NautilusTrader 가이드
permalink: /nautilus-trader-guide/
icon: fas fa-chart-line
---

# NautilusTrader 완벽 가이드

> **Rust 코어 + Python API로 백테스트와 라이브 트레이딩 코드를 일치시키는 이벤트 기반 알고리즘 트레이딩 플랫폼**

**NautilusTrader**는 고성능 알고리즘 트레이딩 플랫폼으로, 백테스트와 라이브 운영에서 **동일한 전략 코드**를 사용하는 패리티(parity)를 핵심 가치로 둡니다. Python 네이티브 개발 경험을 제공하면서도, 성능/안전성이 중요한 코어는 Rust(Cython/PyO3 바인딩 포함)로 구현되어 있습니다.

- 원문 저장소: https://github.com/nautechsystems/nautilus_trader
- 공식 문서: https://nautilustrader.io/docs/
- 패키지: https://pypi.org/project/nautilus_trader/

> 주의: 라이브 트레이딩은 실제 자본 리스크가 있습니다. 이 시리즈는 프레임워크 사용 관점의 기술 가이드이며, 투자 자문이 아닙니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개/핵심 개념](/blog-repo/nautilus-trader-guide-01-intro/) | 왜 NautilusTrader인가, 아키텍처 철학 |
| 02 | [설치/배포 채널](/blog-repo/nautilus-trader-guide-02-installation/) | PyPI/Nautech 인덱스, stable/nightly/develop |
| 03 | [소스 빌드/개발 환경](/blog-repo/nautilus-trader-guide-03-build-and-dev-setup/) | rustup/clang/uv, Makefile 워크플로우 |
| 04 | [백테스트 API](/blog-repo/nautilus-trader-guide-04-backtesting-apis/) | BacktestNode(고수준) vs BacktestEngine(저수준) |
| 05 | [전략 개발](/blog-repo/nautilus-trader-guide-05-strategy-development/) | Strategy 핸들러, 주문/포지션 이벤트, 타이머 |
| 06 | [어댑터/통합](/blog-repo/nautilus-trader-guide-06-adapters-and-integrations/) | 데이터/실행 클라이언트, 통합 상태(planned/beta/stable) |
| 07 | [라이브 트레이딩 노드](/blog-repo/nautilus-trader-guide-07-live-trading-node/) | TradingNodeConfig, 멀티 베뉴, 운영 시 주의 |
| 08 | [캐시/메시지 버스](/blog-repo/nautilus-trader-guide-08-cache-and-message-bus/) | 상태 저장, Pub/Sub, Redis 연동 포인트 |
| 09 | [정밀도/릴리즈/검증](/blog-repo/nautilus-trader-guide-09-precision-release-provenance/) | 64/128-bit precision, 릴리즈 브랜치, attestation |
| 10 | [운영/트러블슈팅](/blog-repo/nautilus-trader-guide-10-ops-troubleshooting/) | 실전 체크리스트, Jupyter 주의, 재현성/안전성 |

