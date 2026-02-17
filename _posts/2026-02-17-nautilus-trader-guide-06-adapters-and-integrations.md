---
layout: post
title: "NautilusTrader 가이드 (06) - 어댑터/통합: DataClient·ExecutionClient와 베뉴 연결"
date: 2026-02-17
permalink: /nautilus-trader-guide-06-adapters-and-integrations/
author: Nautech Systems
categories: [개발 도구, 퀀트 트레이딩]
tags: [NautilusTrader, Adapters, DataClient, ExecutionClient, Multi Venue]
original_url: "https://nautilustrader.io/docs/latest/integrations/"
excerpt: "NautilusTrader는 ports-and-adapters 구조로 다양한 거래소/브로커/데이터 제공자를 통합합니다. 각 통합의 상태(planned/building/beta/stable)와 사용 포인트를 정리합니다."
---

## 어댑터의 역할

`docs/concepts/adapters.md` 기준 어댑터 구성요소:

- `HttpClient` (REST)
- `WebSocketClient` (실시간)
- `InstrumentProvider`
- `DataClient`
- `ExecutionClient`

핵심은 각 베뉴의 원시 API를 Nautilus 도메인 모델로 정규화하는 것입니다.

---

## InstrumentProvider 전략

라이브 노드에서 인스트루먼트 로딩 정책은 크게 두 가지입니다.

- 전량 로딩(`load_all=True`)
- 명시 ID만 로딩(`load_ids=[...]`)

초기 운영에서는 필요한 종목만 제한해서 시작하는 편이 안전하고 디버깅도 쉽습니다.

---

## 통합 상태 읽는 법

`docs/integrations/index.md`는 통합별 상태를 표시합니다.

- `planned`
- `building`
- `beta`
- `stable`

실거래/실계좌 운영 전에는 해당 통합이 최소 `beta` 이상인지, 그리고 본인 전략이 쓰는 기능(주문 타입/체결 피드/히스토리 요청)이 구현됐는지 반드시 확인해야 합니다.

---

## 멀티 베뉴 운용

NautilusTrader는 멀티 베뉴 구성을 전제로 설계되어:

- 한 노드에서 여러 데이터/실행 클라이언트 연결
- 마켓메이킹/차익거래 등 멀티마켓 전략 구현

이 가능합니다. 다음 장에서 라이브 노드 설정과 reconciliation을 구체적으로 봅니다.

