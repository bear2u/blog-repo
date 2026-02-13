---
layout: page
title: GSD 가이드
permalink: /gsd-guide/
icon: fas fa-terminal
---

# GSD (Get Shit Done) 완벽 가이드

> **Claude Code의 컨텍스트 품질 저하 문제를 해결하는 메타 프롬프팅 시스템**

**GSD**는 Claude Code, OpenCode, Gemini CLI를 위한 경량화되고 강력한 메타 프롬프팅, 컨텍스트 엔지니어링, 스펙 주도 개발 시스템입니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/gsd-guide-01-intro/) | GSD란? Context Rot 문제 해결 |
| 02 | [설치 및 시작](/blog-repo/gsd-guide-02-installation/) | 설치 방법, 설정, 권한 모드 |
| 03 | [핵심 워크플로우](/blog-repo/gsd-guide-03-workflow/) | 6단계 워크플로우 이해하기 |
| 04 | [멀티 에이전트 시스템](/blog-repo/gsd-guide-04-agents/) | 11개 전문 에이전트 아키텍처 |
| 05 | [명령어 레퍼런스](/blog-repo/gsd-guide-05-commands/) | 모든 명령어 완벽 가이드 |
| 06 | [설정 및 구성](/blog-repo/gsd-guide-06-configuration/) | 모델 프로필, 워크플로우 설정 |
| 07 | [보안 및 문제 해결](/blog-repo/gsd-guide-07-security-troubleshooting/) | 보안 설정, 트러블슈팅 |
| 08 | [템플릿 시스템](/blog-repo/gsd-guide-08-templates/) | 문서 구조와 템플릿 |
| 09 | [심화 활용법](/blog-repo/gsd-guide-09-advanced/) | TDD, Discovery Levels, 모범 사례 |

---

## 주요 특징

- **Context Engineering** — 컨텍스트 윈도우 품질 저하 방지
- **XML 프롬프트 포맷팅** — Claude 최적화된 구조
- **멀티 에이전트 오케스트레이션** — 병렬 처리로 효율성 향상
- **원자적 Git 커밋** — 각 태스크당 개별 커밋
- **자동 검증** — 목표 대비 코드베이스 검증

---

## 빠른 시작

```bash
# 설치
npx get-shit-done-cc@latest

# 권장: 권한 건너뛰기 모드로 실행
claude --dangerously-skip-permissions

# 프로젝트 초기화
/gsd:new-project

# 워크플로우
/gsd:discuss-phase 1
/gsd:plan-phase 1
/gsd:execute-phase 1
/gsd:verify-work 1
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                      GSD 시스템                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│   │ Meta-       │   │ Context     │   │ Spec-Driven │       │
│   │ Prompting   │   │ Engineering │   │ Development │       │
│   └─────────────┘   └─────────────┘   └─────────────┘       │
│                                                              │
│   ┌───────────────────────────────────────────────────┐     │
│   │              멀티 에이전트 시스템                  │     │
│   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │     │
│   │  │플래너  │ │실행자  │ │검증자  │ │리서처  │     │     │
│   │  └────────┘ └────────┘ └────────┘ └────────┘     │     │
│   └───────────────────────────────────────────────────┘     │
│                                                              │
│   → Claude Code, OpenCode, Gemini CLI 지원                  │
│   → Context Rot 문제 해결                                    │
│   → 일관된 고품질 코드 생성                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| Node.js | 런타임 환경 |
| XML | 프롬프트 포맷팅 |
| Markdown | 문서화 |
| Git | 버전 관리 |

---

## 워크플로우 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                    GSD 워크플로우                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Initialize → /gsd:new-project                           │
│       ↓                                                      │
│  2. Discuss   → /gsd:discuss-phase N                        │
│       ↓                                                      │
│  3. Plan      → /gsd:plan-phase N                           │
│       ↓                                                      │
│  4. Execute   → /gsd:execute-phase N                        │
│       ↓                                                      │
│  5. Verify    → /gsd:verify-work N                          │
│       ↓                                                      │
│  6. Repeat    → 다음 단계 또는 마일스톤 완료                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 지원 플랫폼

| 플랫폼 | 설치 경로 |
|--------|----------|
| Claude Code | `~/.claude/` (global) 또는 `./.claude/` (local) |
| OpenCode | `~/.config/opencode/` |
| Gemini CLI | `~/.gemini/` |

---

## 관련 링크

- [GitHub 저장소](https://github.com/gsd-build/get-shit-done)
- [NPM 패키지](https://www.npmjs.com/package/get-shit-done-cc)
- [Discord 커뮤니티](https://discord.gg/5JJgD5svVS)
- [X (Twitter)](https://x.com/gsd_foundation)

---

**Claude Code is powerful. GSD makes it reliable.**
