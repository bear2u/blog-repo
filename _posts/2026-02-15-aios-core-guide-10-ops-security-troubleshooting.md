---
layout: post
title: "Synkra AIOS Core 완벽 가이드 (10) - 아키텍처/보안/운영: 권한 모드와 트러블슈팅"
date: 2026-02-15
permalink: /aios-core-guide-10-ops-security-troubleshooting/
author: Synkra AIOS Team
categories: [AI 코딩 에이전트, AIOS]
tags: [AIOS, Architecture, Security, Permission Modes, Hardening, Troubleshooting]
original_url: "https://github.com/SynkraAI/aios-core"
excerpt: "AIOS 코어 구성(agents/tasks/workflows/templates), Permission Modes(Explore/Ask/Auto), 보안 하드닝 핵심 포인트와 doctor 기반 트러블슈팅 루틴을 정리합니다."
---

## 코어 아키텍처 한 장 요약

AIOS 코어 문서는 “프레임워크 본체가 무엇으로 구성되는지”를 비교적 명확히 설명합니다.

- agents: 역할/페르소나/의존성
- tasks: 실행 레시피(표준 단위)
- workflows: 멀티 스텝 오케스트레이션
- templates/checklists/data: 반복 산출물/품질 기준/지식베이스

이를 프로젝트에 설치된 `.aios-core/` 관점으로 보면 대략 이런 형태입니다.

```
.aios-core/
├── core/               # registry, health-check, orchestration 등
├── development/        # agents/tasks/workflows
├── product/            # templates/checklists
└── infrastructure/     # scripts/integrations
```

---

## Permission Modes: 에이전트 자율성 제어

문서는 에이전트가 시스템에 어느 정도까지 영향 줄 수 있는지 “모드”로 제어하는 방식을 제시합니다.

```
EXPLORE  : 읽기 중심(안전 탐색)
ASK      : 변경/실행은 확인(기본)
AUTO     : 완전 자동(신뢰된 환경)
```

커맨드 예시:

```text
*mode
*mode explore
*mode ask
*mode auto
*yolo
```

운영 감각:
- 처음 보는 레포/환경: `explore`
- 일상 개발: `ask`
- CI/자동화/신뢰된 레포: `auto`

---

## 보안 하드닝: AIOS가 특별히 위험한 지점

AIOS는 “모델과 시스템 사이의 특권 레이어”에 가까워서, 보안 실수가 곧바로 비용/유출/파괴로 이어질 수 있습니다.

문서가 강조하는 대표 항목:

- API 키/토큰 관리(.env 절대 커밋 금지)
- 권한/샌드박싱/격리
- 입력 검증/인젝션 방지
- 로깅/감사(무슨 작업이 실행됐는지 추적)

키 저장 계층(문서 요지):

- 절대 금지: 소스코드/레포/로그
- 개발에서만 허용: gitignore된 `.env`
- 운영 권장: 시크릿 매니저/CI 주입

---

## 트러블슈팅 기본 루틴: doctor부터

문서는 문제 해결의 출발점으로 `doctor`를 강하게 추천합니다.

```bash
npx @synkra/aios-core doctor
npx @synkra/aios-core doctor --fix
npx @synkra/aios-core doctor --verbose
```

추가로 빠른 조치 예시(문서에서 자주 등장):

```text
*memory clear-cache
*memory rebuild
*config --reset
npx @synkra/aios-core update
```

---

## 흔한 문제 유형(요약)

1. 설치 문제
- Node/npm 버전
- 권한(EACCES)
- 네트워크/캐시/레지스트리

2. 에이전트/메타 에이전트 동작 문제
- 설정 파일 손상
- 필요한 파일/디렉토리 누락

3. 메모리 레이어/성능 문제
- 캐시/인덱스 재생성

4. 보안/권한 에러
- 모드 설정 확인
- 위험 플래그(권한 스킵) 사용 범위 제한

---

## 마무리

AIOS를 “잘 쓰는” 핵심은 결국:

- 프로세스를 태스크/워크플로우로 고정하고
- 권한 모드로 리스크를 제어하고
- doctor/체크리스트로 상태를 관리하는

운영 습관을 만드는 데 있습니다.

---

*시리즈 전체 목차는 [Synkra AIOS Core 가이드](/blog-repo/aios-core-guide/)에서 확인할 수 있습니다.*
