---
layout: post
title: "NautilusTrader 가이드 (08) - 캐시/메시지 버스: 상태 일관성과 컴포넌트 결합도 낮추기"
date: 2026-02-17
permalink: /nautilus-trader-guide-08-cache-and-message-bus/
author: Nautech Systems
categories: [개발 도구, 퀀트 트레이딩]
tags: [NautilusTrader, Cache, MessageBus, Redis, Event Driven]
original_url: "https://nautilustrader.io/docs/latest/concepts/cache/"
excerpt: "Cache는 시스템의 인메모리 상태 저장소이고, MessageBus는 컴포넌트 간 통신 백본입니다. 라이브/백테스트에서 상태 일관성을 확보하는 핵심 포인트를 정리합니다."
---

## Cache: 시스템의 기억장치

캐시는 다음을 저장/조회합니다.

- 마켓데이터(틱/바/오더북)
- 주문/포지션/계좌 상태
- 인스트루먼트 메타

전략에서 `self.cache`를 통해 최근 히스토리를 즉시 조회할 수 있습니다(최신 인덱스 0 방식).

---

## 용량/지속성 설정

`CacheConfig`에서 자주 만지는 값:

- `tick_capacity`
- `bar_capacity`
- `database`(Redis 등)
- `flush_on_start`

바 용량은 bar type마다 별도로 적용되므로, 다중 타임프레임 전략일수록 메모리 예산을 미리 계산해야 합니다.

---

## MessageBus: 느슨한 결합의 핵심

메시지 버스는:

- point-to-point
- pub/sub
- req/rep

패턴을 지원합니다. Strategy/Actor는 `publish_data`, `publish_signal`, 그리고 필요하면 `self.msgbus` 직접 접근으로 메시징을 확장할 수 있습니다.

---

## Redis는 언제 필요한가

설치 문서는 Redis를 **선택 사항**으로 명시합니다.

- 캐시/메시지 버스 백엔드를 Redis로 쓰는 경우만 필요

즉, 단일 노드 실험/백테스트에서는 필수가 아니고, 복원성/지속성이 중요해질 때 도입하는 방식이 현실적입니다.

다음 장에서는 정밀도 모드와 릴리즈/검증(공급망 attestation) 관점을 다룹니다.

