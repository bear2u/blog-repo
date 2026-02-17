---
layout: post
title: "NautilusTrader 가이드 (04) - 백테스트 API: BacktestNode(고수준) vs BacktestEngine(저수준)"
date: 2026-02-17
permalink: /nautilus-trader-guide-04-backtesting-apis/
author: Nautech Systems
categories: [개발 도구, 퀀트 트레이딩]
tags: [NautilusTrader, Backtesting, BacktestNode, BacktestEngine, Data Pipeline]
original_url: "https://nautilustrader.io/docs/latest/concepts/backtesting/"
excerpt: "NautilusTrader 백테스트는 두 API 레벨을 제공합니다. BacktestNode는 운영형 멀티런, BacktestEngine은 세밀한 제어에 유리합니다."
---

## 두 API 레벨의 성격

백테스트 문서는 선택 기준을 명확히 분리합니다.

- **고수준 (`BacktestNode`)**
  - 여러 `BacktestRunConfig`를 묶어 실행
  - 데이터 카탈로그/설정 중심 운영에 유리
- **저수준 (`BacktestEngine`)**
  - 데이터/컴포넌트를 수동으로 조립
  - 실험/미세 제어/반복 실험에 유리

---

## 저수준 API에서 성능 포인트

문서에서 특히 강조하는 부분은 데이터 정렬 비용입니다.

- `add_data(..., sort=True)`를 반복하면 매번 큰 정렬이 누적
- 다중 인스트루먼트 대용량일수록 병목이 커짐

권장 패턴:

1. `sort=False`로 다 넣고
2. 마지막에 `sort_data()` 한 번 수행

---

## 반복 실행 전략

반복 백테스트에서 흔히 놓치는 점:

- `BacktestEngine.reset()`은 상태를 초기화하지만 데이터/인스트루먼트는 보존 정책이 적용됨
- 런마다 완전 분리된 환경이 필요하면 `BacktestNode` 기반 멀티런이 더 명시적

즉, “연구 실험”과 “운영형 배치 백테스트”를 API 레벨로 나눠 쓰는 게 합리적입니다.

다음 장에서는 전략 코드 구조(핸들러, 주문 이벤트, 포지션 이벤트)를 정리합니다.

