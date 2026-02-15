---
layout: post
title: "ClawRouter 완벽 가이드 (04) - 설정 레퍼런스(환경변수, 오버라이드, 가중치)"
date: 2026-02-15
permalink: /clawrouter-guide-04-configuration/
author: BlockRun
categories: [AI 에이전트, ClawRouter]
tags: [Configuration, Env, Wallet, Proxy, Overrides, Scoring]
original_url: "https://github.com/BlockRunAI/ClawRouter"
excerpt: "BLOCKRUN_WALLET_KEY/BLOCKRUN_PROXY_PORT, wallet.key 우선순위, routing overrides와 scoring weights 등 핵심 설정을 정리합니다."
---

## 환경변수

`docs/configuration.md` 기준 핵심 환경변수는 다음입니다.

- `BLOCKRUN_WALLET_KEY`: 지갑 개인키(없으면 파일/자동생성)
- `BLOCKRUN_PROXY_PORT`: 로컬 프록시 포트(기본 8402)

예시:

```bash
export BLOCKRUN_WALLET_KEY=0x...
export BLOCKRUN_PROXY_PORT=8403
openclaw gateway restart
```

---

## 지갑 키 해석 우선순위

문서는 우선순위를 명시합니다.

1. `~/.openclaw/blockrun/wallet.key` (저장된 키)
2. `BLOCKRUN_WALLET_KEY` (환경변수)
3. 자동 생성 후 파일 저장

운영 팁:
- 이미 파일이 존재하면 환경변수를 바꿔도 "파일"이 우선일 수 있습니다.

---

## 라우팅 오버라이드(개념)

`docs/features.md`는 라우팅 동작을 강제하는 오버라이드 예시를 보여줍니다.

- agenticMode를 항상 켜기
- (필요 시) 특정 티어/모델 매핑 조정

정확한 키/스키마는 버전에 따라 달라질 수 있으니, 최종 근거는 `docs/configuration.md`와 `openclaw.plugin.json`의 configSchema를 기준으로 확인하는 편이 안전합니다.

---

## Scoring weights

ClawRouter의 라우팅은 "가중치 기반 스코어링"입니다.
정확한 차원/가중치/키워드 목록은 문서의 Scoring Weights 섹션을 기준으로 확인합니다.

---

## 다음 글

다음 글에서는 프록시 내부에서 요청이 어떻게 처리되는지( dedup -> route -> balance -> 402 payment -> streaming )를 아키텍처 문서 기준으로 정리합니다.

- 다음: [ClawRouter (05) - 아키텍처](/blog-repo/clawrouter-guide-05-architecture/)
