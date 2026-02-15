---
layout: post
title: "ClawRouter 완벽 가이드 (02) - 설치와 빠른 시작"
date: 2026-02-15
permalink: /clawrouter-guide-02-installation/
author: BlockRun
categories: [AI 에이전트, ClawRouter]
tags: [ClawRouter, OpenClaw, Plugin, blockrun/auto, x402, USDC, Base]
original_url: "https://github.com/BlockRunAI/ClawRouter"
excerpt: "OpenClaw 플러그인 설치, 기본 모델을 blockrun/auto로 설정, 로컬 프록시 헬스체크와 지갑 펀딩 흐름을 정리합니다."
---

## 1) 설치

문서에 나오는 설치 경로는 2가지입니다.

### A. OpenClaw 플러그인 설치

```bash
openclaw plugins install @blockrun/clawrouter
```

### B. 스크립트 설치(레포 README)

```bash
curl -fsSL https://blockrun.ai/ClawRouter-update | bash
openclaw gateway restart
```

---

## 2) 기본 모델을 `blockrun/auto`로 설정

```bash
openclaw models set blockrun/auto
```

또는 채팅에서 프로필/모델을 바꾸는 단축 명령을 제공합니다.

```text
/model auto
/model eco
/model premium
/model free
```

---

## 3) 지갑 생성/저장 위치 이해하기

README 기준으로 지갑 키는 보통 다음 위치에 저장됩니다.

- `~/.openclaw/blockrun/wallet.key`

해당 파일이 있으면(저장된 지갑) 그 지갑이 우선 사용됩니다.

---

## 4) 로컬 프록시 헬스체크

기본 프록시 포트는 `8402`입니다.

```bash
curl http://localhost:8402/health
# 또는
curl "http://localhost:8402/health?full=true"
```

---

## 5) 지갑 펀딩

문서에서는 "Base 네트워크의 USDC"로 펀딩하는 흐름을 안내합니다.

- 설치/로그에 표시된 주소로 USDC 전송
- 소액으로도 많은 요청을 처리할 수 있다는 가정(요청당 소액 과금)

---

## 다음 글

다음 글에서는 라우팅이 실제로 어떻게 동작하는지(프로필/티어/세션 핀ning/agentic auto-detect)를 정리합니다.

- 다음: [ClawRouter (03) - 라우팅 프로필과 티어](/blog-repo/clawrouter-guide-03-routing-profiles/)
