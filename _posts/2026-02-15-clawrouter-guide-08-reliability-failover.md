---
layout: post
title: "ClawRouter 완벽 가이드 (08) - 신뢰성/폴백: 오류 재시도와 subscription failover"
date: 2026-02-15
permalink: /clawrouter-guide-08-reliability-failover/
author: BlockRun
categories: [AI 에이전트, ClawRouter]
tags: [Reliability, Fallback, Failover, Subscription, OpenClaw]
original_url: "https://github.com/BlockRunAI/ClawRouter"
excerpt: "provider 오류 시 fallback chain으로 재시도하는 구조와, OpenClaw의 fallback 메커니즘으로 subscription 모델을 1순위로 두고 ClawRouter를 최종 폴백으로 두는 패턴을 정리합니다."
---

## ClawRouter 내부 폴백(문서 요지)

`docs/architecture.md`는 특정 상태코드(400/401/402/429/5xx 등)에서 다음 모델로 넘어가는 fallback chain을 설명합니다.

운영 포인트:
- 특정 모델/프로바이더가 불안정하면 "요청을 버리지" 않고 다음 후보로 넘기는 것이 목표

---

## 빈 지갑 폴백(무료 티어)

`docs/features.md`는 지갑이 비었을 때 무료 모델로 자동 폴백하는 동작을 설명합니다.

- 지갑 잔액이 0이어도 "아예 멈추지 않게" 하는 설계

---

## subscription + ClawRouter failover 패턴

`docs/subscription-failover.md`는 "구독(Claude Pro/ChatGPT Plus 등)"을 ClawRouter에 직접 넣기보다,
OpenClaw의 provider/fallback 체계를 이용하는 방식을 권장합니다.

요지:

1. primary 모델: subscription/provider를 OpenClaw에서 설정
2. fallback: `blockrun/auto`를 fallback chain에 추가

예시:

```bash
openclaw models set anthropic/claude-sonnet-4-5
openclaw models fallbacks add blockrun/auto
```

---

## 다음 글

다음 글에서는 키/스캐너 경고/위협 모델을 포함한 보안 관점을 정리합니다.

- 다음: [ClawRouter (09) - 보안 모델](/blog-repo/clawrouter-guide-09-security/)
