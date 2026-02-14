---
layout: post
title: "Explain OpenClaw 완벽 가이드 (02) - 핵심 개념과 용어"
date: 2026-02-14
permalink: /explain-openclaw-guide-02-concepts-glossary/
author: centminmod
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, Gateway, Channel, Session, Tool, Pairing, Allowlist, Threat Model]
original_url: "https://github.com/centminmod/explain-openclaw"
excerpt: "Gateway 중심 구조와 trust boundary, pairing/allowlist, bind(loopback/lan/tailnet) 같은 필수 용어를 한 번에 정리합니다."
---
## 핵심 전제: Gateway 호스트가 보안 경계다

Explain OpenClaw의 위협 모델 문서가 강조하는 메시지:

- OpenClaw는 단순 챗봇이 아니라 도구 호출과 자동화를 연결할 수 있다.
- 따라서 **Gateway가 실행되는 호스트**가 사실상의 신뢰 경계다.

---

## Glossary(운영자 관점 압축)

| 용어 | 의미 | 운영/보안 포인트 |
|------|------|------------------|
| Gateway | 항상 켜져 있는 제어면 프로세스 | 포트/프록시 노출과 인증이 핵심 |
| Channel | Telegram/WhatsApp/Discord/iMessage 커넥터 | dmPolicy/groupPolicy가 1차 방어선 |
| Session | 대화 히스토리와 메타데이터(기본은 디스크 JSONL) | 길어질수록 토큰/비용/드리프트 증가 |
| Agent turn | 컨텍스트 구성 -> 모델 호출 -> 도구 호출(옵션) -> 응답 | 10+ 단계 작업은 실패 확률이 올라감 |
| Tool | 웹/브라우저/파일/exec 등 모델이 호출 가능한 기능 | 켜는 순간 공격 성공 시 피해가 커짐 |
| Pairing | 승인 기반 접근 제어(DM/디바이스) | "모르는 사람이 트리거"하는 것을 현실적으로 차단 |
| Allowlist | 허용된 사용자/그룹/계정 목록 | 실질적인 보안 경계 역할 |
| gateway.bind | Gateway가 바인딩하는 네트워크 범위 | 기본은 loopback, 원격은 터널/테일넷 우선 |
| trustedProxies | 프록시 뒤에서 클라이언트 IP 판별 체인 | X-Forwarded-For 스푸핑 방지 |
| mDNS discovery | LAN에서 Gateway 자동 발견 브로드캐스트 | 공유 네트워크에서는 끄는 게 안전 |

---

## 운영자가 먼저 고정해야 하는 3가지

### 1) DM 정책: `pairing` 또는 `allowlist`

```bash
openclaw pairing list telegram
openclaw pairing approve telegram <CODE>
```

### 2) 네트워크 노출: loopback + 터널

- SSH 터널
```bash
ssh -N -L 18789:127.0.0.1:18789 user@gateway-host
```

- Tailscale Serve: UX가 좋고 tailnet-only로 운영하기 쉬움

### 3) 도구 표면: 최소에서 시작

프롬프트 인젝션은 "모델이 텍스트를 잘못 출력"에서 끝나지 않습니다.
도구가 켜져 있으면 **파일/네트워크/메시징** 같은 실제 행동으로 번질 수 있습니다.

---

## 다음 글

다음 글에서는 목적별로 CLI를 묶고, `openclaw security audit`를 운영 루틴으로 만드는 방법을 정리합니다.

- 다음: [Explain OpenClaw (03) - CLI 빠른 참조와 보안 감사](/blog-repo/explain-openclaw-guide-03-cli-and-audit/)
