---
layout: post
title: "Entire CLI 완벽 가이드 (01) - 소개 및 개요"
date: 2026-02-11
permalink: /entire-cli-guide-01-intro/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, CLI, Git, AI Agent, Session Tracking, Claude Code, Gemini CLI]
original_url: "https://github.com/entireio/cli"
excerpt: "Git 워크플로우에 통합되어 AI 에이전트 세션을 자동으로 캡처하는 Entire CLI 소개"
---

## Entire CLI란?

**Entire CLI**는 Git 워크플로우에 통합되어 **AI 에이전트 세션을 자동으로 캡처**하는 도구입니다. Claude Code, Gemini CLI 등의 AI 코딩 도구와 함께 사용하면 코드가 어떻게 작성되었는지에 대한 검색 가능한 기록을 커밋과 함께 인덱싱할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                        Entire CLI                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Your Branch                  entire/checkpoints/v1        │
│        │                              │                      │
│        ▼                              │                      │
│   [Base Commit]                       │                      │
│        │                              │                      │
│        │  ┌──── AI Agent ────┐        │                      │
│        │  │   작업 중...     │        │                      │
│        │  └─────────────────┘        │                      │
│        │                              │                      │
│        ▼                              ▼                      │
│   [Your Commit] ────────────► [Session Metadata]            │
│        │                      (transcript, prompts)          │
│        ▼                                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 주요 특징

### 1. **Git 훅 기반 자동 캡처**

Entire는 Git 훅과 AI 에이전트 훅을 통해 세션 데이터를 자동으로 캡처합니다.

- **Transparent**: 기존 워크플로우를 방해하지 않음
- **Automatic**: 별도의 수동 작업 불필요
- **Complete**: 모든 프롬프트, 응답, 파일 변경사항 기록

### 2. **체크포인트 시스템**

세션 중 여러 체크포인트를 생성하여 언제든지 되돌릴 수 있습니다.

```bash
# 현재 세션 상태 확인
entire status

# 이전 체크포인트로 되돌리기
entire rewind
```

### 3. **다중 전략 지원**

프로젝트와 팀의 요구사항에 맞는 전략 선택 가능:

| 전략 | 코드 커밋 | 적합한 용도 |
|-----|---------|-----------|
| **Manual-commit** (기본) | 없음 | 대부분의 워크플로우 |
| **Auto-commit** | 자동 생성 | 자동 커밋을 원하는 팀 |

### 4. **AI 에이전트 통합**

- **Claude Code** - Anthropic의 공식 CLI
- **Gemini CLI** - Google의 Gemini CLI (프리뷰)
- 확장 가능한 아키텍처로 더 많은 에이전트 지원 예정

---

## 왜 Entire를 사용하는가?

### **문제: AI 코딩의 기록 부재**

AI 에이전트가 코드를 작성할 때, 최종 결과물(커밋)만 남고 **과정은 사라집니다**.

- "왜 이렇게 구현했지?"
- "이전에 시도한 접근법은 무엇이었지?"
- "어떤 프롬프트로 이 코드가 생성되었지?"

### **해결: 전체 세션 기록**

Entire는 모든 것을 캡처합니다:

```
Commit a3f2b1c4
├── 코드 변경사항 (Git)
└── AI 세션 메타데이터 (Entire)
    ├── 사용자 프롬프트
    ├── AI 응답 (전체 transcript)
    ├── 수정된 파일 목록
    ├── 토큰 사용량
    └── 타임스탬프
```

### **이점**

1. **이해도 향상** - 코드가 어떻게 만들어졌는지 추적
2. **디버깅 용이** - 문제 발생 시 세션으로 돌아가기
3. **팀 협업** - 다른 팀원의 AI 세션 이해
4. **학습 자료** - AI와의 효과적인 작업 방식 학습
5. **감사 추적** - 코드 변경 이력의 완전한 컨텍스트

---

## 핵심 개념 미리보기

### **Session (세션)**

AI 에이전트와의 **완전한 상호작용 단위**입니다.

- Session ID: `2026-01-08-abc123de-f456-7890-abcd-ef1234567890`
- 시작부터 끝까지의 모든 프롬프트와 응답 포함

### **Checkpoint (체크포인트)**

세션 내에서 **되돌릴 수 있는 저장 지점**입니다.

- Checkpoint ID: `a3b2c4d5e6f7` (12-hex-char)
- 코드 상태 + 메타데이터의 스냅샷

### **Strategy (전략)**

체크포인트를 **언제, 어떻게 저장할지** 결정합니다.

- **Manual-commit**: 커밋 시 체크포인트 생성
- **Auto-commit**: AI 응답마다 자동 커밋 + 체크포인트

---

## 간단한 예제

### 1. 프로젝트에서 Entire 활성화

```bash
cd your-project
entire enable
```

이 명령은:
- Git 훅 설치 (prepare-commit-msg, post-commit, pre-push)
- AI 에이전트 훅 설정 (Claude Code 기본)
- `.entire/` 디렉토리 생성

### 2. AI 에이전트로 작업

```bash
# Claude Code로 작업
claude "Add user authentication"
```

Entire가 백그라운드에서:
- 프롬프트 캡처
- AI 응답 기록
- 파일 변경사항 추적

### 3. 커밋 생성

```bash
git commit -m "Add user authentication"
```

Entire가 자동으로:
- 세션 메타데이터를 `entire/checkpoints/v1` 브랜치에 저장
- Checkpoint ID를 커밋 메시지에 추가
- Shadow 브랜치 정리

### 4. 나중에 조회

```bash
# 세션 목록 보기
entire status

# 특정 커밋의 세션 확인
entire explain <commit-hash>

# 이전 체크포인트로 되돌리기
entire rewind
```

---

## 기술 스택

| 구성 요소 | 기술 |
|----------|-----|
| **언어** | Go 1.25.x |
| **CLI 프레임워크** | Cobra (github.com/spf13/cobra) |
| **TUI** | Huh (github.com/charmbracelet/huh) |
| **Git 라이브러리** | go-git/go-git/v5 |
| **빌드 도구** | mise |
| **린팅** | golangci-lint |

---

## 요구사항

- **Git** - 버전 관리
- **macOS or Linux** - Windows는 WSL 사용
- **AI 에이전트** - Claude Code 또는 Gemini CLI

---

## 설치 방법 (간단)

```bash
# Homebrew로 설치
brew tap entireio/tap
brew install entireio/tap/entire

# 또는 Go로 설치
go install github.com/entireio/cli/cmd/entire@latest
```

---

## 다음 단계

이제 Entire CLI가 무엇인지 이해했습니다. 다음 챕터에서는:

- **설치 및 시작하기** - 상세한 설치 과정과 첫 세션 만들기
- **핵심 개념** - Session, Checkpoint, Strategy 깊이 이해
- **일반적인 워크플로우** - 실제 사용 시나리오

---

## 참고 링크

- [GitHub 저장소](https://github.com/entireio/cli)
- [공식 문서](https://github.com/entireio/cli#readme)
- [Claude Code 문서](https://docs.anthropic.com/en/docs/claude-code)

---

*다음 글에서는 Entire CLI 설치 방법과 첫 번째 세션을 만드는 방법을 살펴봅니다.*
