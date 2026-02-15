---
layout: post
title: "ClawRouter 완벽 가이드 (05) - 아키텍처: 프록시/라우팅/스트리밍"
date: 2026-02-15
permalink: /clawrouter-guide-05-architecture/
author: BlockRun
categories: [AI 에이전트, ClawRouter]
tags: [Architecture, Proxy, Dedup, Routing Engine, SSE, Heartbeat]
original_url: "https://github.com/BlockRunAI/ClawRouter"
excerpt: "아키텍처 문서의 흐름(로컬 프록시, dedup, 14-dim 라우팅, SSE 하트비트, 폴백)을 운영자 관점에서 요약합니다."
---

## 전체 그림

`docs/architecture.md`는 ClawRouter를 "로컬 프록시"로 설명합니다.

```text
OpenClaw(또는 클라이언트)
  -> ClawRouter proxy(localhost)
     - dedup cache
     - router(가중치 스코어러)
     - x402 payment
     - fallback chain
     - SSE heartbeat
  -> BlockRun API
  -> upstream providers
```

---

## 요청 플로우(핵심 단계)

아키텍처 문서의 순서를 운영자 관점으로 정리하면:

1. 요청 수신(OpenAI 호환)
2. deduplication: 동일 요청 재생/동시요청 합치기
3. 라우팅(`blockrun/auto`일 때): 프롬프트 기반으로 티어/모델 결정
4. 잔액 체크(충분/low/empty)
5. 스트리밍이면 SSE 하트비트로 타임아웃 방지
6. 402 응답을 받으면 x402 서명 후 재시도
7. provider 오류 시 fallback chain으로 재시도

---

## 라우팅 엔진: 가중치 기반 스코어러

문서에는 "14-dimension" 가중치 스코어러가 등장합니다.
핵심 포인트는:

- 라우팅 결정을 위해 외부 API를 호출하지 않음
- 로컬에서 신호(signals)를 모아 점수/확신도를 계산

---

## 다음 글

다음 글에서는 x402 결제 플로우와 지갑 키 보관/백업, /wallet 같은 운영 커맨드를 정리합니다.

- 다음: [ClawRouter (06) - 지갑과 결제(x402)](/blog-repo/clawrouter-guide-06-wallet-x402/)
