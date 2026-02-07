---
layout: post
title: "SpecWeave 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-02-07
permalink: /specweave-guide-02-installation/
author: Anton Abyzov
categories: [AI 코딩, 개발 도구]
tags: [SpecWeave, Installation, Getting Started, CLI, Setup]
original_url: "https://github.com/anton-abyzov/specweave"
excerpt: "SpecWeave 설치부터 첫 Increment 생성까지 단계별 가이드"
---

## 시스템 요구사항

### 필수 요구사항

**Node.js 20.12.0 이상** 필요 (Node.js 22 LTS 권장)

```bash
node --version   # 버전 확인
```

> **`SyntaxError: Unexpected token 'with'` 오류 발생?** Node.js 버전이 너무 낮습니다. [업그레이드 방법 →](https://spec-weave.com/docs/guides/troubleshooting/common-errors#node-version-error)

### 권장 환경

- **AI 코딩 도구**: Claude Code (Opus 4.6 권장)
- **Git**: 버전 관리 시스템
- **운영체제**: macOS, Linux, Windows

## 설치 방법

### 글로벌 설치

```bash
npm install -g specweave
```

설치 확인:

```bash
specweave --version
```

## 새 프로젝트 시작

### 1. 프로젝트 디렉토리 생성

```bash
mkdir my-app && cd my-app
```

### 2. SpecWeave 초기화

```bash
specweave init .
```

초기화 과정에서 다음을 설정합니다:

```
? Project name: my-app
? Primary language: TypeScript
? Enable Deep Interview Mode? (y/N): y
? Integration with GitHub? (y/N): y
? Integration with JIRA? (y/N): n
```

#### Deep Interview Mode

복잡한 기능을 위해 활성화하면:
- 아키텍처에 대한 40개 이상의 질문
- 통합, UI/UX, 트레이드오프 논의
- 훨씬 더 상세한 스펙 생성

### 3. 생성된 구조

```
my-app/
├── .specweave/
│   ├── config.json              # 프로젝트 설정
│   ├── docs/                    # 내부 문서
│   └── increments/              # 기능별 디렉토리
├── .git/                        # Git 저장소
└── README.md
```

## 기존 프로젝트에 추가

### 1. 프로젝트 루트로 이동

```bash
cd your-existing-project
```

### 2. SpecWeave 초기화

```bash
specweave init .
```

기존 프로젝트의 경우 SpecWeave가 자동으로:
- 프로젝트 구조 분석
- 기술 스택 감지
- 적절한 플러그인 제안

### 3. Brownfield 분석

기존 프로젝트에서는 문서 갭을 자동으로 감지:

```bash
/sw:analyze-brownfield
```

누락된 문서를 생성하도록 제안합니다.

## 첫 Increment 생성

### 1. 기능 정의

Claude Code에서:

```bash
/sw:increment "Add dark mode support"
```

### 2. 생성 과정

SpecWeave가 자동으로:

1. **Spec 생성** (`spec.md`)
   - 사용자 스토리
   - 인수 기준
   - 성공 메트릭

2. **Plan 생성** (`plan.md`)
   - 아키텍처 결정
   - 기술 선택
   - 구현 전략

3. **Tasks 생성** (`tasks.md`)
   - 실행 가능한 작업 목록
   - 각 작업의 테스트 기준
   - 우선순위 지정

### 3. 생성된 구조

```
.specweave/increments/0001-dark-mode/
├── spec.md      # 사용자 관점: 무엇을, 왜
├── plan.md      # 개발자 관점: 어떻게
└── tasks.md     # 실행 관점: 구체적 작업
```

## 기본 워크플로우

### 단계별 실행

```bash
/sw:do    # 다음 작업 하나 실행
```

각 작업마다:
1. 구현
2. 테스트 실행
3. 실패 시 자동 수정
4. 성공 시 다음 작업

### 자율 실행 모드

```bash
/sw:auto
```

자율 모드에서 SpecWeave는:
- 모든 작업을 순차적으로 실행
- 테스트 실패 시 자동 수정
- 수 시간 동안 실행 가능
- 백그라운드에서 작동

### 진행 상황 확인

```bash
/sw:status 0001    # 특정 Increment 상태
/sw:list           # 모든 Increment 목록
```

출력 예시:

```
Increment 0001: Dark Mode Support
Status: In Progress (60% complete)
Tasks: 3 done, 2 in progress, 0 blocked
Last activity: 5 minutes ago
```

## 코드 리뷰 및 종료

### 리뷰 요청

```bash
/sw:grill 0001
```

SpecWeave가 수행:
- 코드 품질 검사
- 테스트 커버리지 확인
- 보안 취약점 스캔
- 문서 완성도 평가

### Increment 종료

```bash
/sw:done 0001
```

종료 프로세스:
1. 최종 품질 검증
2. 모든 테스트 통과 확인
3. GitHub/JIRA 자동 동기화
4. 문서 업데이트

## 외부 도구 연동 설정

### GitHub 연동

```bash
/sw-github:setup
```

설정 항목:
- Personal Access Token
- 저장소 URL
- 기본 브랜치
- 레이블 매핑

### JIRA 연동

```bash
/sw-jira:setup
```

설정 항목:
- JIRA URL
- API Token
- 프로젝트 키
- 이슈 타입 매핑

## CLI 명령어

### 업데이트

```bash
specweave update
```

**모든 것을 업데이트**:
- CLI 도구
- 플러그인
- 지침 파일

대부분의 문제는 `specweave update`로 해결됩니다 (98%).

### 검증

```bash
specweave validate
```

프로젝트 설정 및 구조 검증:
- config.json 유효성
- 플러그인 상태
- 외부 통합 연결

## 문제 해결

### Node.js 버전 오류

```bash
# Node.js 버전 확인
node --version

# NVM으로 업그레이드 (macOS/Linux)
nvm install 22
nvm use 22

# Windows: Node.js 설치 프로그램 다운로드
# https://nodejs.org
```

### 플러그인 로드 실패

```bash
# 플러그인 재설치
specweave update

# 플러그인 검증
specweave validate
```

### 외부 통합 오류

```bash
# 연결 테스트
/sw-github:test-connection
/sw-jira:test-connection

# 재설정
/sw-github:setup
```

## 환경 변수

`.env` 파일로 민감한 정보 관리:

```bash
# .env
GITHUB_TOKEN=ghp_...
JIRA_API_TOKEN=...
ANTHROPIC_API_KEY=sk-ant-...
```

`.gitignore`에 추가:

```
.env
.specweave/config.local.json
```

## 다음 단계

다음 챕터에서는 SpecWeave의 아키텍처와 핵심 개념을 자세히 알아봅니다.

---

## 시리즈 네비게이션

- **이전**: [(1) 소개 및 개요]({{ site.baseurl }}/specweave-guide-01-intro/)
- **현재**: (2) 설치 및 빠른 시작
- **다음**: [(3) 아키텍처 및 핵심 개념]({{ site.baseurl }}/specweave-guide-03-architecture/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/specweave-guide/)
