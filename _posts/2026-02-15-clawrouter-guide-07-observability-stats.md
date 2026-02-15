---
layout: post
title: "ClawRouter 완벽 가이드 (07) - 관측과 비용 절감: /stats와 로그 읽기"
date: 2026-02-15
permalink: /clawrouter-guide-07-observability-stats/
author: BlockRun
categories: [AI 에이전트, ClawRouter]
tags: [Observability, Stats, Logs, Savings, Dedup]
original_url: "https://github.com/BlockRunAI/ClawRouter"
excerpt: "/stats로 절감액을 확인하고, OpenClaw 로그에서 라우팅 티어/모델 선택을 검증하는 방법을 정리합니다."
---

## /stats

`docs/features.md`는 비용 절감 추적을 위해 `/stats` 커맨드를 제공합니다.

```text
/stats
```

출력에는 대략 다음이 포함됩니다.

- 기간(예: 7일)
- 총 요청 수
- 총 비용
- 기준 비용(예: Opus 가정)
- 절감액/절감률
- 티어별 분포

---

## 로그로 라우팅 확인

`docs/troubleshooting.md`는 라우팅이 동작하는지 확인하는 방법으로 로그 팔로우를 제시합니다.

```bash
openclaw logs --follow
```

여기서 SIMPLE/MEDIUM/REASONING 등의 티어와 선택된 모델, 추정 비용/절감률을 확인합니다.

---

## 로컬 저장 경로(문서에 등장하는 것들)

문서에는 다음 경로들이 반복 등장합니다.

- 지갑 키: `~/.openclaw/blockrun/wallet.key`
- (통계/로그) `~/.openclaw/blockrun/logs/` (문서 예시)

---

## 다음 글

다음 글에서는 fallback chain, 무료 티어 폴백, subscription 기반 provider와의 failover 패턴을 정리합니다.

- 다음: [ClawRouter (08) - 신뢰성/폴백](/blog-repo/clawrouter-guide-08-reliability-failover/)
