---
layout: post
title: "NautilusTrader 가이드 (01) - 소개/핵심 개념: 백테스트-라이브 패리티를 목표로 한 트레이딩 엔진"
date: 2026-02-17
permalink: /nautilus-trader-guide-01-intro/
author: Nautech Systems
categories: [개발 도구, 퀀트 트레이딩]
tags: [NautilusTrader, Algorithmic Trading, Event Driven, Rust, Python]
original_url: "https://github.com/nautechsystems/nautilus_trader"
excerpt: "NautilusTrader는 이벤트 기반 아키텍처로 백테스트와 라이브 트레이딩의 코드 패리티를 지향합니다. 핵심 철학과 전체 구조를 먼저 정리합니다."
---

## NautilusTrader가 푸는 문제

많은 트레이딩 팀이 겪는 문제는 다음입니다.

- 연구/백테스트 코드는 Python
- 실거래 코드는 다른 언어/엔진으로 재구현
- 그 결과 모델 로직과 운영 로직이 어긋남

NautilusTrader는 이 갭을 줄이기 위해 **동일 전략 코드로 백테스트와 라이브를 잇는 것**을 핵심 목표로 둡니다.

---

## 아키텍처 철학

`docs/concepts/architecture.md`에서 강조하는 큰 축:

- DDD(도메인 중심 설계)
- 이벤트 기반 구조
- 메시징 패턴(Pub/Sub, Req/Rep)
- Ports and Adapters(헥사고날 스타일)
- Crash-only에 가까운 복구 관점

그리고 품질 우선순위는 대체로:

1. Reliability
2. Performance
3. Modularity/Testability/Maintainability

순으로 제시됩니다.

---

## 코어 컴포넌트

아키텍처 문서 기준 핵심 블록:

- `NautilusKernel`: 전체 수명주기/컴포넌트 오케스트레이션
- `MessageBus`: 시스템 간 메시지 전달 백본
- `Cache`: 인메모리 상태 저장소(주문/포지션/마켓데이터)
- `DataEngine`: 시장 데이터 처리/라우팅
- `ExecutionEngine`: 주문 라이프사이클/체결 동기화
- `RiskEngine`: 사전 리스크 체크와 노출 관리

---

## 환경 컨텍스트

NautilusTrader는 동일 코어를 아래 환경으로 분기합니다.

- `Backtest`: 과거 데이터 + 시뮬레이트 베뉴
- `Sandbox`: 실시간 데이터 + 시뮬레이트 베뉴
- `Live`: 실시간 데이터 + 실제 베뉴(페이퍼/실계좌)

이 “공통 코어 + 환경 컨텍스트” 모델이 전략 코드 재사용성을 높이는 핵심입니다.

다음 장에서는 설치 채널(PyPI/사설 인덱스/개발휠)과 운영 시 버전 선택 기준을 정리합니다.

