---
layout: post
title: "GSD (Get Shit Done) 완벽 가이드 (05) - 명령어 레퍼런스"
date: 2026-02-13
permalink: /gsd-guide-05-commands/
author: TÂCHES
categories: [AI 코딩, 개발 도구]
tags: [Claude Code, Commands, CLI, Reference]
original_url: "https://github.com/gsd-build/get-shit-done"
excerpt: "GSD의 모든 명령어 완벽 가이드"
---

## 핵심 워크플로우 명령어

| 명령어 | 설명 |
|--------|------|
| `/gsd:new-project [--auto]` | 전체 초기화: 질문 → 리서치 → 요구사항 → 로드맵 |
| `/gsd:discuss-phase [N]` | 계획 전 구현 결정 사항 캡처 |
| `/gsd:plan-phase [N]` | 단계 리서치 + 계획 + 검증 |
| `/gsd:execute-phase <N>` | 모든 계획을 병렬 웨이브로 실행 |
| `/gsd:verify-work [N]` | 수동 사용자 수용 테스트 |
| `/gsd:audit-milestone` | 마일스톤 완료 정의 달성 검증 |
| `/gsd:complete-milestone` | 마일스톤 아카이브, 릴리스 태그 |
| `/gsd:new-milestone [name]` | 다음 버전 시작 |

---

## 네비게이션 명령어

| 명령어 | 설명 |
|--------|------|
| `/gsd:progress` | 현재 위치와 다음 단계 표시 |
| `/gsd:help` | 모든 명령어 및 사용법 가이드 표시 |
| `/gsd:update` | 체인지로그 미리보기와 함께 GSD 업데이트 |
| `/gsd:join-discord` | GSD Discord 커뮤니티 참여 |

---

## 브라운필드 명령어

기존 코드베이스에 GSD를 적용할 때 사용:

| 명령어 | 설명 |
|--------|------|
| `/gsd:map-codebase` | new-project 전 기존 코드베이스 분석 |

### map-codebase로 분석되는 항목

- 스택 및 기술
- 아키텍처 패턴
- 코딩 컨벤션
- 주요 관심사

---

## 단계 관리 명령어

| 명령어 | 설명 |
|--------|------|
| `/gsd:add-phase` | 로드맵에 단계 추가 |
| `/gsd:insert-phase [N]` | 단계 사이에 긴급 작업 삽입 |
| `/gsd:remove-phase [N]` | 향후 단계 제거 및 재번호 |
| `/gsd:list-phase-assumptions [N]` | 계획 전 Claude의 의도된 접근 방식 확인 |
| `/gsd:plan-milestone-gaps` | 감사에서 발견된 갭을 닫는 단계 생성 |

---

## 세션 관리 명령어

| 명령어 | 설명 |
|--------|------|
| `/gsd:pause-work` | 단계 중간에 중지할 때 핸드오프 생성 |
| `/gsd:resume-work` | 마지막 세션에서 복원 |

### pause-work 사용 시나리오

```
/gsd:execute-phase 3
... 실행 중 ...
/gsd:pause-work
# → HANDOFF.md 생성

# 나중에
/gsd:resume-work
# → HANDOFF.md에서 복원
```

---

## 유틸리티 명령어

| 명령어 | 설명 |
|--------|------|
| `/gsd:settings` | 모델 프로필 및 워크플로우 에이전트 구성 |
| `/gsd:set-profile <profile>` | 모델 프로필 전환 (quality/balanced/budget) |
| `/gsd:add-todo [desc]` | 나중을 위해 아이디어 캡처 |
| `/gsd:check-todos` | 대기 중인 todo 목록 |
| `/gsd:debug [desc]` | 영구 상태로 체계적 디버깅 |
| `/gsd:quick` | GSD 보장으로 애드혹 태스크 실행 |

---

## 명령어 옵션

### new-project

```bash
/gsd:new-project           # 대화형
/gsd:new-project --auto    # 자동 승인 모드
```

### plan-phase

```bash
/gsd:plan-phase 1              # 기본
/gsd:plan-phase 1 --skip-research   # 리서치 건너뛰기
/gsd:plan-phase 1 --skip-verify     # 계획 검증 건너뛰기
/gsd:plan-phase --gaps         # 갭 클로저 모드
```

### execute-phase

```bash
/gsd:execute-phase 1          # 기본 실행
/gsd:execute-phase 1 --plan 2 # 특정 계획만 실행
```

### settings

```bash
/gsd:settings                 # 대화형 설정
```

---

## 명령어 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                   새 프로젝트 시작                           │
│                         ↓                                    │
│                  /gsd:new-project                           │
│                         ↓                                    │
│              ┌────────┴────────┐                            │
│              │   마일스톤 루프   │                            │
│              └────────┬────────┘                            │
│                       ↓                                      │
│              ┌────────┴────────┐                            │
│              │    단계 루프     │                            │
│              │                 │                            │
│              │ /gsd:discuss-phase → /gsd:plan-phase         │
│              │         ↓                                    │
│              │ /gsd:execute-phase → /gsd:verify-work        │
│              │         ↓                                    │
│              │   (다음 단계 또는 완료)                       │
│              └────────┬────────┘                            │
│                       ↓                                      │
│           /gsd:complete-milestone                           │
│                       ↓                                      │
│           /gsd:new-milestone (또는 완료)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 빠른 참조 카드

| 상황 | 명령어 |
|------|--------|
| 새 프로젝트 시작 | `/gsd:new-project` |
| 현재 상태 확인 | `/gsd:progress` |
| 다음 단계 계획 | `/gsd:discuss-phase N` |
| 계획 실행 | `/gsd:plan-phase N` → `/gsd:execute-phase N` |
| 작업 검증 | `/gsd:verify-work N` |
| 작업 중단 | `/gsd:pause-work` |
| 작업 재개 | `/gsd:resume-work` |
| 빠른 수정 | `/gsd:quick` |
| 버그 디버깅 | `/gsd:debug` |
| 설정 변경 | `/gsd:settings` |

---

*다음 글에서는 GSD 설정 및 구성을 살펴봅니다.*
