---
layout: page
title: SpecWeave 가이드
permalink: /specweave-guide/
icon: fas fa-magic
---

# SpecWeave 완벽 가이드

> **AI 코딩을 위한 엔터프라이즈 레이어**

SpecWeave는 Claude Opus 4.6 기반으로 영구 메모리, GitHub/JIRA 동기화, 품질 게이트, 자율 실행을 제공하는 Spec-Driven Development 프레임워크입니다. **잠자는 동안 기능을 배포하세요.**

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요]({{ site.baseurl }}/specweave-guide-01-intro/) | SpecWeave란?, 핵심 차별화 요소, 주요 기능 |
| 02 | [설치 및 빠른 시작]({{ site.baseurl }}/specweave-guide-02-installation/) | 설치, 초기화, 첫 Increment 생성 |
| 03 | [아키텍처 및 핵심 개념]({{ site.baseurl }}/specweave-guide-03-architecture/) | Spec-Driven Development, 디렉토리 구조, 품질 게이트 |
| 04 | [Increment 시스템]({{ site.baseurl }}/specweave-guide-04-increment-system/) | Spec/Plan/Tasks, 라이프사이클, 외부 동기화 |
| 05 | [자율 실행 모드]({{ site.baseurl }}/specweave-guide-05-autonomous-mode/) | /sw:auto, 수 시간 작업, 자동 수정 메커니즘 |
| 06 | [플러그인 시스템]({{ site.baseurl }}/specweave-guide-06-plugin-system/) | Lazy Loading, 68+ 에이전트, 커스텀 플러그인 |
| 07 | [LSP 통합]({{ site.baseurl }}/specweave-guide-07-lsp-integration/) | Language Server Protocol, 100배 빠른 코드 이해 |
| 08 | [외부 도구 연동]({{ site.baseurl }}/specweave-guide-08-external-integrations/) | GitHub, JIRA, Azure DevOps 양방향 동기화 |
| 09 | [AI 에이전트 시스템]({{ site.baseurl }}/specweave-guide-09-ai-agents/) | PM, Architect, QA, Security, DevOps 에이전트 |
| 10 | [실전 활용 및 팁]({{ site.baseurl }}/specweave-guide-10-best-practices/) | 프로덕션 배포, 문제 해결, 최적화 전략 |

---

## 핵심 특징

### 🧠 영구 메모리

채팅이 끝나도 모든 컨텍스트가 `.specweave/` 디렉토리에 영구 보존됩니다.

```
.specweave/increments/0001-oauth/
├── spec.md    <- WHAT: 사용자 스토리, 인수 기준
├── plan.md    <- HOW: 아키텍처 결정, 기술 선택
└── tasks.md   <- DO: 구현 작업 + 테스트
```

**6개월 후**: "OAuth" 검색 → 정확한 결정 사항, 승인자, 구축 이유를 즉시 확인

### 💤 자율 실행

```bash
/sw:increment "User authentication"
/sw:auto                              # 잠자는 동안 배포
```

AI가 수 시간 동안:
- 작업을 순차적으로 실행
- 테스트 실패 시 자동 수정
- GitHub/JIRA 자동 동기화
- 완성된 작업을 리뷰용으로 제공

### ⚡ Lazy Plugin Loading (99% 토큰 절약)

| 시나리오 | Without Lazy | With Lazy |
|----------|-------------|-----------|
| 비-SpecWeave 작업 | ~60k 토큰 | ~500 토큰 |
| SpecWeave 작업 | ~60k 토큰 | ~60k (필요시) |

프롬프트 키워드 기반 자동 로드:
- "React" 언급 → 프론트엔드 플러그인
- "Kubernetes" 언급 → K8s 플러그인
- "security" 언급 → 보안 플러그인

### 🤖 68+ AI 에이전트

| 에이전트 | 역할 |
|---------|------|
| **PM** | 요구사항, 사용자 스토리 |
| **Architect** | 시스템 설계, ADR |
| **QA Lead** | 테스트 전략, 품질 게이트 |
| **Security** | OWASP 리뷰, 취약점 스캔 |
| **DevOps** | CI/CD, 인프라 |

컨텍스트 기반 자동 활성화. Claude Opus 4.6 권장.

### 🔍 LSP 통합 (100배 빠른 코드 이해)

| 작업 | Without LSP | With LSP |
|------|------------|----------|
| 참조 찾기 | Grep + 15파일 (~10K 토큰) | 시맨틱 쿼리 (~500 토큰) |
| 타입 에러 | 빌드 + 파싱 (~5K 토큰) | getDiagnostics (~1K 토큰) |
| 정의 탐색 | Grep (~8K 토큰) | goToDefinition (~200 토큰) |

### 🔗 외부 통합

| 플랫폼 | 기능 |
|--------|------|
| **GitHub** | Issues, PRs, 마일스톤, 양방향 동기화 |
| **JIRA** | 에픽, 스토리, 상태 동기화 |
| **Azure DevOps** | 작업 항목, 영역 경로 |

---

## 빠른 시작

### 1. 설치

```bash
npm install -g specweave   # Node.js 20.12.0+ 필요
```

### 2. 프로젝트 초기화

```bash
# 새 프로젝트
mkdir my-app && cd my-app
specweave init .

# 기존 프로젝트
cd your-project
specweave init .
```

### 3. 첫 Increment

```bash
/sw:increment "Add dark mode"   # Spec + Plan + Tasks 생성
/sw:auto                        # 자율 실행
/sw:grill 0001                  # 코드 리뷰
/sw:done 0001                   # 품질 검증 후 종료
```

---

## 핵심 명령어

| 명령어 | 목적 |
|--------|------|
| `/sw:increment "feature"` | Spec + Plan + Tasks 생성 |
| `/sw:auto` | 자율 실행 (수 시간) |
| `/sw:do` | 작업 하나씩 실행 |
| `/sw:grill 0001` | **코드 리뷰** |
| `/sw:done 0001` | 종료 + 검증 |
| `/sw:sync-progress` | GitHub/JIRA 동기화 |
| `/sw:next` | 자동 종료 + 다음 제안 |

**[100개 이상의 명령어 →](https://spec-weave.com/docs/commands/overview)**

---

## 모든 환경에서 작동

| 시나리오 | 동작 |
|----------|------|
| **10년 된 레거시** | Brownfield 분석으로 문서 갭 감지 |
| **주말 MVP** | 완전한 spec-driven 개발 |
| **50개 팀 엔터프라이즈** | JIRA/ADO 멀티 프로젝트 동기화 |

---

## SpecWeave로 구축

> 이 프레임워크는 자기 자신을 구축합니다. 모든 기능, 버그 수정, 릴리스가 spec-driven입니다.

- **배포 빈도**: 매월 다수
- **기능**: 190개 이상
- **버전**: v1.0.235

**[Browse our increments →](https://github.com/anton-abyzov/specweave/tree/develop/.specweave/increments)**

---

## 시스템 요구사항

- **Node.js 20.12.0+** (22 LTS 권장)
- AI 코딩 도구 (Claude Code + Opus 4.6 권장)
- Git 저장소

---

## 관련 링크

- [GitHub 저장소](https://github.com/anton-abyzov/specweave)
- [공식 문서](https://spec-weave.com)
- [Discord 커뮤니티](https://discord.gg/UYg4BGJ65V)
- [YouTube 튜토리얼](https://www.youtube.com/@antonabyzov)

---

## 시작하기

[🚀 Chapter 1: 소개 및 개요부터 시작하기]({{ site.baseurl }}/specweave-guide-01-intro/)
