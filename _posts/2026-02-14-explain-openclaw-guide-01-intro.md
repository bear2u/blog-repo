---
layout: post
title: "Explain OpenClaw 완벽 가이드 (01) - 소개 및 로드맵"
date: 2026-02-14
permalink: /explain-openclaw-guide-01-intro/
author: centminmod
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, Clawdbot, Moltbot, Security, Deployment, Hardening, Docs]
original_url: "https://github.com/centminmod/explain-openclaw"
excerpt: "Explain OpenClaw 레포가 다루는 범위와, 안전한 읽는 순서(초급->운영->보안)를 정리합니다."
---
## Explain OpenClaw는 무엇인가?

**Explain OpenClaw**는 OpenClaw(구 Moltbot/Clawdbot)를 처음 접하는 사람을 위한 쉬운 설명부터,
운영자가 실제로 부딪히는 배포/하드닝/보안 감사/최적화까지 한 번에 묶어 둔 "살아 있는" 지식베이스입니다.

- 원문: https://github.com/centminmod/explain-openclaw
- OpenClaw 공식 문서(정본): https://docs.openclaw.ai

이 시리즈는 원문을 그대로 복붙하기보다,
"어디를 먼저 읽어야 하는지"와 "운영자가 놓치기 쉬운 포인트"를 한국어 챕터로 재구성합니다.

---

## 30초 모델: OpenClaw를 어떻게 이해할까?

> OpenClaw는 **항상 켜져 있는 Gateway**가 메시징 채널과 LLM을 연결하고,
> 필요하면 도구를 호출해 행동까지 수행하게 만드는 셀프호스팅 플랫폼입니다.

```text
메시징 앱들
   |
   v
Gateway (단일 항상-ON 프로세스)
 - 채널 ingress/egress
 - 라우팅 + 정책
 - 세션/상태(디스크)
 - 에이전트 턴(모델 호출)
 - 도구 호출(옵션)
   |
   v
LLM 프로바이더(또는 로컬 모델)
   |
   v
채널로 응답 반환
```

가장 중요한 운영/보안 전제는 이 한 문장입니다.

- **Gateway 호스트가 신뢰 경계(trust boundary)** 이다.

---

## 이 레포가 집중하는 배포 시나리오

Explain OpenClaw README는 아래를 중심으로 문서를 구성합니다.

1. Standalone Mac mini: 로컬 우선, 노출 최소화, 높은 프라이버시
2. Isolated VPS: 원격 서버 운영(항상 켜짐), 대신 하드닝 난이도 증가
3. Cloudflare Moltworker: Cloudflare 서버리스 런타임에서 Gateway 실행(PoC 성격)
4. Docker Model Runner: 로컬 LLM로 API 비용 0 + 데이터 외부 전송 0 목표

---

## 추천 읽는 순서(보안 우선)

1. 개념 잡기(plain English): 무엇이 Gateway이고, 무엇이 위험 표면인지
2. 프라이버시/보안: 위협 모델 + 하드닝 체크리스트 + 안전 설정 예시
3. 배포 런북: 내 시나리오(Mac mini/VPS/Moltworker/로컬모델) 따라가기
4. 운영 루틴: `openclaw security audit`와 비용/리소스 모니터링

---

## 최소 빠른 시작(안전한 기본값)

```bash
# 설치(권장)
curl -fsSL https://openclaw.ai/install.sh | bash

# 온보딩 + 데몬 설치
openclaw onboard --install-daemon

# 상태 확인
openclaw gateway status
openclaw health

# 보안 감사
openclaw security audit --deep
# 풋건 자동 수정(설정/권한 변경이 발생할 수 있음)
openclaw security audit --fix
```

---

## 다음 글

다음 글에서는 문서에서 반복 등장하는 핵심 개념과 용어를 운영자 기준으로 정리합니다.

- 다음: [Explain OpenClaw (02) - 핵심 개념과 용어](/blog-repo/explain-openclaw-guide-02-concepts-glossary/)
