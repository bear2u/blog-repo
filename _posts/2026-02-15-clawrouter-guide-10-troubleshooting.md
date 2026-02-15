---
layout: post
title: "ClawRouter 완벽 가이드 (10) - 트러블슈팅/운영: 흔한 오류와 업데이트"
date: 2026-02-15
permalink: /clawrouter-guide-10-troubleshooting/
author: BlockRun
categories: [AI 에이전트, ClawRouter]
tags: [Troubleshooting, Updates, blockrun/auto, Port 8402, OpenClaw]
original_url: "https://github.com/BlockRunAI/ClawRouter"
excerpt: "문서에 정리된 흔한 오류(Unknown model, No API key, config validation, 잔액 부족, 포트 충돌)와 재설치/업데이트, 라우팅 검증 방법을 정리합니다."
---

## Quick checklist

`docs/troubleshooting.md`의 체크리스트를 운영자 관점으로 요약하면:

1. 설치된 버전 확인
2. 프록시가 떠 있는지 확인(`:8402` health)
3. 로그에서 라우팅이 찍히는지 확인
4. `/stats`로 절감 확인

---

## 자주 나오는 오류들(문서 요지)

- "Unknown model: blockrun/auto" / "Unknown model: auto"
  - 플러그인이 로드되지 않았거나 버전 이슈

- "No API key found for provider blockrun"
  - auth profile이 제대로 주입되지 않았을 수 있음

- "Config validation failed: plugin not found"
  - 플러그인 디렉토리는 삭제됐는데 config에 남아있는 케이스

- "Insufficient funds" / "No USDC balance"
  - 지갑 펀딩 필요

---

## 포트 8402 충돌

문서는 v0.4.1+에서 기존 프록시를 재사용하도록 개선됐다고 설명합니다.
필요하면 포트를 바꿀 수 있습니다.

```bash
export BLOCKRUN_PROXY_PORT=8403
openclaw gateway restart
```

---

## 업데이트/재설치

문서가 제시하는 업데이트 경로:

```bash
curl -fsSL https://raw.githubusercontent.com/BlockRunAI/ClawRouter/main/scripts/reinstall.sh | bash
openclaw gateway restart
```

---

## 라우팅 검증

```bash
openclaw logs --follow
```

로그에서 티어/모델/절감률이 정상적으로 찍히는지 확인합니다.
