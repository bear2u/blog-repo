---
layout: post
title: "Synkra AIOS Core 완벽 가이드 (09) - LLM 라우팅: claude-max/claude-free로 비용 최적화"
date: 2026-02-15
permalink: /aios-core-guide-09-llm-routing/
author: Synkra AIOS Team
categories: [AI 코딩 에이전트, AIOS]
tags: [AIOS, LLM Routing, Claude Code, DeepSeek, Cost Optimization, Security]
original_url: "https://github.com/SynkraAI/aios-core"
excerpt: "Claude Max(OAuth)와 DeepSeek(Anthropic 호환 엔드포인트)를 스위칭하는 라우팅 커맨드 개념, 설치/설정/보안 주의점을 정리합니다."
---

## 목적: “개발은 싸게, 중요한 건 프리미엄으로”

AIOS 문서의 LLM 라우팅 가이드는 아주 실용적인 목표를 갖습니다.

- 복잡한 작업/품질이 중요한 작업: `claude-max` (구독 기반)
- 반복 개발/테스트/가벼운 작업: `claude-free` (저비용 API)

---

## 제공 커맨드(문서 기준)

| 커맨드 | 백엔드 | 비용 | 사용처 |
|---|---|---:|---|
| `claude-max` | Claude Max(OAuth) | 구독 | 고난도 분석/중요 작업 |
| `claude-free` | DeepSeek(Anthropic 호환) | 매우 저렴 | 개발/테스트/대량 작업 |

---

## 설치(문서 예시)

레포를 클론했을 때:

```bash
node .aios-core/infrastructure/scripts/llm-routing/install-llm-routing.js
```

---

## DeepSeek API 키 설정

문서는 `.env`에 키를 두는 방식을 제안합니다.

```bash
DEEPSEEK_API_KEY=sk-your-key-here
```

키를 찾는 순서(문서 요지):

1. 프로젝트 `.env`
2. 환경변수

---

## 동작 방식(요지)

- `claude-max`
  - 대체 프로바이더 설정을 제거/초기화
  - Claude 로그인(OAuth) 기반으로 실행

- `claude-free`
  - `.env`를 상위 디렉토리까지 탐색해 키를 로드
  - DeepSeek의 Anthropic 호환 엔드포인트로 설정

문서에 나온 엔드포인트:

```text
https://api.deepseek.com/anthropic
```

---

## 보안 주의: permission bypass 옵션

문서에는 `claude-max`/`claude-free`가 기본적으로

- `--dangerously-skip-permissions`

같은 형태의 “권한 확인 스킵” 플래그를 쓴다는 경고가 포함되어 있습니다.

의미:
- 편하지만, 신뢰하지 않는 레포에서 실행하면 위험할 수 있습니다.

권장 운영 감각:
- 라우팅 커맨드는 **신뢰된 환경/레포**에서만
- 낯선 레포에서는 기본 `claude`로 확인 프롬프트를 켜고 사용

---

## 비용 최적화 팁(실무)

- “디버깅/반복”은 저비용 백엔드
- “아키텍처/보안/중요 PR”은 프리미엄 백엔드
- 팀 기준을 문서화해서 스토리/체크리스트에 포함

---

*다음 글에서는 AIOS 코어 아키텍처(구성요소)와 보안 하드닝/권한 모드/트러블슈팅을 한 챕터로 묶어 운영 관점에서 정리합니다.*
