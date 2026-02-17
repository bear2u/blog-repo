---
layout: post
title: "NautilusTrader 가이드 (10) - 운영/트러블슈팅: 라이브 배포 체크리스트와 흔한 함정"
date: 2026-02-17
permalink: /nautilus-trader-guide-10-ops-troubleshooting/
author: Nautech Systems
categories: [개발 도구, 퀀트 트레이딩]
tags: [NautilusTrader, Operations, Troubleshooting, Live Trading, Reliability]
original_url: "https://nautilustrader.io/docs/latest/concepts/live/"
excerpt: "라이브 트레이딩 운영에서 중요한 것은 기능보다 안정성입니다. Jupyter 비권장, 멀티 노드 구성, 재조정 임계값, 로그/모니터링 관점의 실전 체크리스트를 정리합니다."
---

## 1) Jupyter는 연구용으로만

라이브 문서가 명시적으로 경고하는 부분:

- Jupyter 이벤트 루프 충돌 가능성
- 셀 재실행 순서/커널 상태 비결정성

즉, 라이브는 독립 프로세스(서비스)로 실행하는 것이 맞습니다.

---

## 2) 프로세스 모델

문서 권장:

- 프로세스당 TradingNode 하나
- 병렬 운용이 필요하면 프로세스를 분리

단일 프로세스 내 다중 노드 강행은 상태/글로벌 리소스 충돌 리스크를 키웁니다.

---

## 3) 재조정(Reconciliation) 튜닝

라이브 운영에서 흔한 문제는 다음입니다.

- venue 응답 지연
- bulk 조회 누락
- API rate limit

그래서:

- interval/threshold/retry를 보수적으로 설정
- single-order query 제한/지연도 고려

해야 false positive와 과도한 API 사용을 줄일 수 있습니다.

---

## 4) 백테스트-라이브 괴리 줄이기

코드 패리티가 있어도 괴리는 발생합니다.

- 슬리피지/체결 우선순위/부분체결
- 네트워크 지연
- 베뉴별 주문 제약

운영에서 중요한 건 “괴리가 난다”가 아니라, **괴리를 측정/설명 가능한 상태를 유지**하는 것입니다.

---

## 5) 최소 운영 체크리스트

1. stable 버전 고정 및 릴리즈 노트 확인
2. 전략/리스크/실행 설정 버전 관리
3. 핵심 메트릭(주문 지연, reject율, reconciliation 경고) 모니터링
4. 장애 시 재기동/복구 절차 문서화
5. 소액/페이퍼 단계 검증 후 실계좌 확장

이 시리즈는 입문-구조-운영의 뼈대를 잡는 데 초점을 맞췄습니다. 실제 배포 전에는 공식 문서의 최신 릴리즈 가이드를 반드시 함께 확인하세요.

