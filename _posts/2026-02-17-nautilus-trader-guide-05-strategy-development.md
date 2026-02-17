---
layout: post
title: "NautilusTrader 가이드 (05) - 전략 개발: Strategy 핸들러, 주문/포지션 이벤트, 타이머"
date: 2026-02-17
permalink: /nautilus-trader-guide-05-strategy-development/
author: Nautech Systems
categories: [개발 도구, 퀀트 트레이딩]
tags: [NautilusTrader, Strategy, Actor, Order Events, Position Events]
original_url: "https://nautilustrader.io/docs/latest/concepts/strategies/"
excerpt: "Strategy는 Actor를 상속하며 데이터·주문·포지션·라이프사이클 이벤트를 처리합니다. 핸들러 구조와 실전 구현 포인트를 정리합니다."
---

## Strategy는 Actor의 확장

전략 클래스는 `Strategy`를 상속하며:

- Actor 기능(데이터 요청/구독, 메시징, 타이머)
- 주문 관리 기능

을 함께 가집니다.

---

## 구현 시 핵심 패턴

전략은 보통 아래 순서로 구성합니다.

1. `on_start`: 인스트루먼트 조회, 히스토리 요청, 라이브 구독
2. `on_bar`/`on_quote_tick` 등 데이터 핸들러에서 시그널 계산
3. 주문 제출/수정/취소
4. `on_order_*`, `on_position_*` 이벤트로 상태 추적

문서의 중요한 주의점:

- `__init__` 단계에서 clock/logger 등 런타임 컴포넌트 사용 금지

---

## 핸들러 체인 이해

이벤트는 “구체 핸들러 → 일반 핸들러” 순서로 흐릅니다.

- 예: `on_order_filled` → `on_order_event` → `on_event`

이 구조를 이해하면:

- 세밀한 이벤트 로깅
- 공통 후처리(리스크 계산/알림)

를 깔끔하게 분리할 수 있습니다.

---

## 전략 코드 재사용성

문서가 지속적으로 강조하는 점:

- 같은 전략 소스를 백테스트와 라이브에 재사용 가능

다만 라이브에서는 지연/슬리피지/부분체결/거래소 규칙 차이로 인해 결과가 달라질 수 있으므로, 다음 장의 어댑터/통합 레이어 이해가 필수입니다.

