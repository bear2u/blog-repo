---
layout: post
title: "Explain OpenClaw 완벽 가이드 (03) - CLI 빠른 참조와 보안 감사"
date: 2026-02-14
permalink: /explain-openclaw-guide-03-cli-and-audit/
author: centminmod
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, CLI, Security Audit, Hardening, Troubleshooting]
original_url: "https://github.com/centminmod/explain-openclaw"
excerpt: "온보딩부터 상태/로그/채널/원격 접근까지 자주 쓰는 CLI를 묶고, 보안 감사(--deep/--fix)로 풋건을 제거하는 방법을 정리합니다."
---
## 자주 쓰는 명령(목적별)

### 설치/온보딩

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
openclaw onboard --install-daemon
```

### 상태/헬스/대시보드

```bash
openclaw gateway status
openclaw status
openclaw health

# 토큰이 필요한 경우 토큰 포함 URL을 출력
openclaw dashboard
```

### 채널/로그/페어링

```bash
openclaw channels list
openclaw channels status
openclaw channels logs

openclaw pairing list telegram
openclaw pairing approve telegram <CODE>
```

### 원격 접근(포트 공개 대신)

```bash
ssh -N -L 18789:127.0.0.1:18789 user@gateway-host
```

---

## `openclaw security audit`를 운영 루틴으로 만들기

Explain OpenClaw의 보안 감사 레퍼런스는 이 명령을 "구성/권한/정책 스캐너"로 설명합니다.

| 명령 | 요약 |
|------|------|
| `openclaw security audit` | 로컬 config, 파일 권한, 채널 정책을 읽기 전용으로 검사 |
| `openclaw security audit --deep` | 위 + 실행 중 Gateway에 WebSocket probe(5초 타임아웃) |
| `openclaw security audit --fix` | 안전한 자동 수정을 먼저 적용한 뒤 감사 수행(시스템 변경 가능) |

`--fix`는 대략 이런 종류를 손댑니다.
- 파일/디렉토리 권한(700/600)
- 로그 민감정보 마스킹 설정
- 일부 채널의 위험한 그룹 정책(`open`)을 더 안전한 쪽으로

주의점:
- 감사가 통과해도 "코드 취약점"까지 보장하는 것은 아닙니다.

---

## 트러블슈팅 초단기 체크

- Control UI unauthorized
  - `openclaw dashboard`로 토큰 포함 URL로 접속
  - 올바른 프로필/호스트인지 확인

- 18789 포트 충돌
  - `openclaw gateway stop` 또는 포트 변경

- 채널이 조용함
  - `channels status/logs`
  - pairing/allowlist 차단 여부
  - 모델 인증이 Gateway 호스트에 설정됐는지

---

## 다음 글

다음 글에서는 OpenClaw의 메시지 흐름과 코드 구조를 한 번에 잡기 위해, 아키텍처와 레포 맵을 요약합니다.

- 다음: [Explain OpenClaw (04) - 아키텍처와 레포 맵](/blog-repo/explain-openclaw-guide-04-architecture-repo-map/)
