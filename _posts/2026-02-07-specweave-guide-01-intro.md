---
layout: post
title: "SpecWeave 완벽 가이드 (01) - 소개 및 개요"
date: 2026-02-07
permalink: /specweave-guide-01-intro/
author: Anton Abyzov
categories: [AI 코딩, 개발 도구]
tags: [SpecWeave, AI Coding, Claude Code, Spec-Driven Development, Enterprise AI]
original_url: "https://github.com/anton-abyzov/specweave"
excerpt: "AI 코딩을 위한 엔터프라이즈 레이어 SpecWeave - 영구 메모리, GitHub/JIRA 동기화, 자율 실행"
---

## SpecWeave란?

**SpecWeave는 AI 코딩을 위한 엔터프라이즈 레이어**입니다. Claude Opus 4.6 기반으로 영구 메모리, GitHub/JIRA 동기화, 품질 게이트, 자율 실행 기능을 제공합니다.

### 슬로건

> *"Ship features while you sleep."* (잠자는 동안 기능을 배포하세요)

### 핵심 개념

모든 AI 코딩 도구는 채팅이 끝나면 컨텍스트를 잃습니다. SpecWeave는 **영구 문서**를 생성합니다:

```
.specweave/increments/0001-oauth/
├── spec.md    <- WHAT: 사용자 스토리, 인수 기준
├── plan.md    <- HOW: 아키텍처 결정, 기술 선택
└── tasks.md   <- DO: 구현 작업 + 테스트
```

**6개월 후**: "OAuth" 검색 → 정확한 결정 사항, 승인자, 구축 이유를 즉시 찾을 수 있습니다.

## 빠른 데모

```bash
/sw:increment "User authentication"
/sw:auto                              # 잠자는 동안 배포
```

### 실제 작동 방식

```
[08:23:41] [Planning]      Analyzing T-003: Implement refresh token rotation
[08:24:12] [Implementing]  Writing src/auth/token-manager.ts
[08:25:33] [Testing]       Running tests... FAILED
[08:25:47] [Fixing]        Adjusting implementation...
[08:26:15] [Testing]       Re-running... PASSED
[08:26:22] [Done]          T-003 complete. Moving to T-004...
```

AI가:
- Spec + Plan + Tasks 생성
- **수 시간 동안** 자율적으로 실행
- 테스트 실행, 실패 수정, GitHub/JIRA 동기화
- 완성된 작업을 리뷰용으로 제공

## 주요 차별화 요소

### 1. Lazy Plugin Loading (99% 토큰 절약)

SpecWeave는 프롬프트 키워드를 기반으로 플러그인을 **필요시에만** 로드합니다:

| 시나리오 | Lazy Loading 없이 | Lazy Loading 사용 |
|----------|-------------------|-------------------|
| 비-SpecWeave 작업 | ~60k 토큰 | ~500 토큰 |
| SpecWeave 작업 | ~60k 토큰 | ~60k (필요시) |

"React frontend" 언급 → 프론트엔드 플러그인 로드
"Kubernetes deploy" 언급 → K8s 플러그인 로드

**수동 설정 불필요!**

### 2. Self-Improving Skills (자가 개선)

SpecWeave는 수정 사항에서 학습합니다:

```markdown
## Skill Memories
<!-- Auto-captured by SpecWeave reflect -->
- Always use `vi.hoisted()` for ESM mocking in Vitest 4.x+
- Prefer native `fs` over fs-extra in new code
```

다음번에는 같은 실수를 반복하지 않습니다.

### 3. 구조화된 문서 (루트 디렉토리 깔끔)

모든 것이 `.specweave/`에 정리됩니다:

```
.specweave/
├── increments/####-name/     # 기능 스펙 + 작업
├── docs/internal/            # 리빙 문서
│   ├── architecture/adr/     # Architecture Decision Records
│   └── specs/                # 기능 명세서
└── config.json               # 프로젝트 설정
```

프로젝트 루트는 깔끔하게 유지. 마크다운 파일 흩어지지 않음.

### 4. Deep Interview Mode (NEW)

복잡한 기능의 경우, 초기화 시 **Deep Interview Mode**를 활성화할 수 있습니다. Claude가 명세서를 작성하기 전에 아키텍처, 통합, UI/UX, 트레이드오프에 대해 **40개 이상의 질문**을 합니다:

```
Deep Interview Mode

For big features, Claude can ask 40+ questions about architecture,
integrations, UI/UX, and tradeoffs before creating specifications.

Enable Deep Interview Mode? [y/N]
```

> Claude Code 창시자 Thariq의 워크플로우에서 영감: *"큰 기능에 대해 Claude가 40개 이상의 질문을 하면 훨씬 더 상세한 스펙을 얻게 됩니다."*

### 5. 68+ AI 에이전트 협업

| 에이전트 | 역할 |
|---------|------|
| **PM** | 요구사항, 사용자 스토리, 인수 기준 |
| **Architect** | 시스템 설계, ADR, 기술 결정 |
| **QA Lead** | 테스트 전략, 품질 게이트 |
| **Security** | OWASP 리뷰, 취약점 스캐닝 |
| **DevOps** | CI/CD, 인프라, 배포 |

컨텍스트에 따라 에이전트 자동 활성화. "security" 언급 → 보안 전문 지식 로드.

**최적**: Claude Opus 4.6 및 Sonnet 4.5

### 6. LSP 통합 (100배 빠른 코드 이해)

SpecWeave는 **Language Server Protocol**을 활용하여 시맨틱 코드 인텔리전스를 제공:

| 작업 | LSP 없이 | LSP 사용 |
|------|----------|----------|
| 모든 참조 찾기 | Grep + 15개 파일 읽기 (~10K 토큰) | 시맨틱 쿼리 (~500 토큰) |
| 타입 에러 확인 | 빌드 + 출력 파싱 (~5K 토큰) | getDiagnostics (~1K 토큰) |
| 정의로 이동 | Grep + 검증 (~8K 토큰) | goToDefinition (~200 토큰) |

**LSP 플러그인 자동 작동**: `.cs` 파일 편집 → `csharp-lsp` 활성화. `.ts` 편집 → `typescript-lsp` 활성화. 설정 불필요.

```bash
# 스택별 언어 서버 설치
npm install -g typescript-language-server typescript  # TypeScript
pip install pyright                                    # Python
dotnet tool install -g csharp-ls                      # C#
```

## 핵심 명령어

| 명령어 | 목적 |
|--------|------|
| `/sw:increment "feature"` | Spec + Plan + Tasks 생성 |
| `/sw:auto` | 자율 실행 (수 시간) |
| `/sw:do` | 한 번에 하나씩 작업 실행 |
| `/sw:grill 0001` | **종료 전 코드 리뷰** |
| `/sw:done 0001` | 품질 검증과 함께 종료 |
| `/sw:sync-progress` | GitHub/JIRA/ADO에 푸시 |
| `/sw:next` | 자동 종료 + 다음 제안 |

**[100개 이상의 명령어 →](https://spec-weave.com/docs/commands/overview)**

## 외부 통합

| 플랫폼 | 기능 |
|--------|------|
| **GitHub** | Issues, PRs, 마일스톤, 양방향 동기화 |
| **JIRA** | 에픽, 스토리, 상태 동기화 |
| **Azure DevOps** | 작업 항목, 영역 경로 |

**자동 동기화**: Increment 종료 시 (`/sw:done`) 외부 도구가 즉시 업데이트됩니다.

## 모든 환경에서 작동

| 시나리오 | 동작 |
|----------|------|
| **10년 된 레거시 코드베이스** | Brownfield 분석으로 문서 갭 감지 |
| **주말 MVP** | 완전한 spec-driven 개발 |
| **50개 팀 엔터프라이즈** | JIRA/ADO로 멀티 프로젝트 동기화 |

## 시스템 요구사항

- **Node.js 20.12.0+** (22 LTS 권장)
- 모든 AI 코딩 도구 (Claude Code + Opus 4.6 권장)
- Git 저장소

## SpecWeave로 구축

> 이 프레임워크는 자기 자신을 구축합니다. 모든 기능, 버그 수정, 릴리스가 spec-driven입니다.

- **배포 빈도**: 매월 다수 배포
- **기능**: 190개 이상
- **Increments**: [Browse our increments →](https://github.com/anton-abyzov/specweave/tree/develop/.specweave/increments)

## 다음 단계

다음 챕터에서는 SpecWeave를 설치하고 첫 Increment를 생성하는 방법을 다룹니다.

---

## 시리즈 네비게이션

- **현재**: (1) 소개 및 개요
- **다음**: [(2) 설치 및 빠른 시작]({{ site.baseurl }}/specweave-guide-02-installation/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/specweave-guide/)
