---
layout: post
title: "NautilusTrader 가이드 (07) - 라이브 트레이딩 노드: TradingNodeConfig와 reconciliation"
date: 2026-02-17
permalink: /nautilus-trader-guide-07-live-trading-node/
author: Nautech Systems
categories: [개발 도구, 퀀트 트레이딩]
tags: [NautilusTrader, Live Trading, TradingNode, Reconciliation, Risk]
original_url: "https://nautilustrader.io/docs/latest/concepts/live/"
excerpt: "라이브 트레이딩은 Backtest와 같은 코드라도 운영 설정이 핵심입니다. TradingNodeConfig, 실행 엔진 재조정(reconciliation), 멀티 노드 운영 주의사항을 정리합니다."
---

## 라이브 모드의 기본 원칙

라이브 문서는 첫 문장부터 금융 리스크를 강조합니다.

- 백테스트와 코드 동일성은 강점이지만
- 운영 설정/복구/검증을 모르면 위험이 커집니다.

---

## `TradingNodeConfig` 핵심

라이브에서 제일 중요한 설정 축:

- `trader_id`, `instance_id`
- timeout 계열(연결/재조정/종료)
- `data_clients`, `exec_clients`
- `LiveExecEngineConfig`

이 값들이 실거래 안정성(재연결/복구/정리)에 직결됩니다.

---

## Reconciliation(실행 상태 재조정)

문서에서 매우 상세히 다루는 부분입니다.

- 시작 시 venue 상태와 내부 상태 정합 맞추기
- in-flight 주문 지연 모니터링
- open order 정합 검사
- 누락/불일치 시 상태 수렴 로직

실전에서는 API rate limit, 타임스탬프 지연, 히스토리 윈도우 제한 때문에 false positive를 줄이도록 임계값을 보수적으로 튜닝해야 합니다.

---

## 운영상 중요한 경고

라이브 문서의 대표 경고:

- **프로세스당 TradingNode 1개 권장**
- **Jupyter 노트북 라이브 운용 비권장**(이벤트 루프/상태 안정성 이슈)

즉, 라이브는 서비스 프로세스(스크립트/프로세스 매니저)로 운영하는 것이 정석입니다.

다음 장에서는 캐시와 메시지 버스 관점에서 데이터/상태 흐름을 정리합니다.

