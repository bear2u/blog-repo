---
layout: post
title: "Explain OpenClaw 완벽 가이드 (04) - 아키텍처와 레포 맵"
date: 2026-02-14
permalink: /explain-openclaw-guide-04-architecture-repo-map/
author: centminmod
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, Architecture, Repo Map, Gateway, WebSocket]
original_url: "https://github.com/centminmod/explain-openclaw"
excerpt: "Gateway 중심 데이터 흐름(채널->정책->세션->에이전트 턴->도구->응답)과, 코드에서 어디를 보면 되는지(레포 맵)를 요약합니다."
---
## High-level data flow

Explain OpenClaw의 기술 문서(architecture.md)는 메시지 흐름을 이렇게 잡습니다.

```text
Messaging apps -> Gateway -> (policy/routing) -> agent turn -> (tools) -> response
```

Gateway는 control plane 성격이 강해서, 대부분의 CLI/상태/대시보드 호출도 결국 Gateway의 RPC로 수렴합니다.

---

## 메시지 라이프사이클(개념)

1. Ingress: 채널 어댑터가 이벤트 수신
2. Authorization: DM/그룹 정책과 allowlist 적용
3. Routing: 세션 키/에이전트 선택
4. Context: 디스크에서 히스토리/세션 로드, 컨텍스트 구성
5. Agent turn: 모델 호출(스트리밍), 필요 시 도구 호출
6. Delivery: 채널 포맷으로 응답 전송
7. Persistence: 트랜스크립트/메타데이터 기록

---

## Repo map: 어디를 보면 되는가?

Explain OpenClaw의 repo-map.md 요지를 "읽는 순서"로 바꾸면 이렇습니다.

- Gateway: `src/gateway/` + `docs/gateway/`
- Channels: `src/channels/` + `src/telegram/` 등 채널별 폴더
- Agent turns: `src/auto-reply/` 와 `src/auto-reply/reply/agent-runner.ts`
- Config: `src/config/` (타입/검증/마이그레이션)
- Security: `src/security/` (audit, 정책, 외부 컨텐츠 처리)

팁: "모든 걸 다 읽기" 대신, 하나의 채널(예: Telegram)을 잡고 메시지가 응답으로 가는 경로를 따라가면 구조가 빨리 잡힙니다.

---

## 다음 글

다음 글에서는 Mac mini와 Isolated VPS 배포를 묶어서, loopback + 터널/Tailscale 중심의 안전한 운영 패턴을 정리합니다.

- 다음: [Explain OpenClaw (05) - 배포 1: Mac mini/VPS](/blog-repo/explain-openclaw-guide-05-deploy-mac-vps/)
