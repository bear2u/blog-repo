---
layout: post
title: "Synkra AIOS Core 완벽 가이드 (02) - 설치와 퀵스타트: init/install/doctor"
date: 2026-02-15
permalink: /aios-core-guide-02-installation-quickstart/
author: Synkra AIOS Team
categories: [AI 코딩 에이전트, AIOS]
tags: [AIOS, aios-core, Installation, NPX, doctor, Node.js]
original_url: "https://github.com/SynkraAI/aios-core"
excerpt: "npx 기반 설치(프로젝트 디렉토리에서 실행), init vs install, doctor로 진단/자동수정까지 빠르게 시작합니다."
---

## 준비물

문서 기준으로 최소 요구사항은 다음입니다.

- Node.js 18+
- npm 9+
- Git
- (선택) GitHub CLI `gh`
- Claude Code/AI IDE

버전 확인:

```bash
node --version
npm --version
git --version
```

---

## 설치 옵션: init vs install

AIOS는 보통 두 가지 진입점을 제공합니다.

1. **새 프로젝트(그린필드)**

```bash
npx @synkra/aios-core init my-project
cd my-project
```

2. **기존 프로젝트(브라운필드)**

```bash
cd existing-project
npx @synkra/aios-core install
```

`init`은 새 디렉토리 생성/초기 구성을 포함하는 경우가 많고, `install`은 현재 프로젝트에 AIOS 구성요소를 주입하는 느낌으로 이해하면 됩니다.

---

## 매우 중요한 포인트: NPX는 “프로젝트 디렉토리에서” 실행

문서에서 반복 경고하는 흔한 실수는 이것입니다.

- 홈 디렉토리나 임의 위치에서 `npx ... install` 실행
- NPX 임시 디렉토리에서 실행되어 설치 대상/IDE 감지가 꼬임

올바른 패턴:

```bash
cd /path/to/your/project
npx @synkra/aios-core install
```

---

## 설치 확인: doctor로 진단하기

설치 후 가장 먼저 돌릴 명령:

```bash
npx @synkra/aios-core doctor
```

문서에는 다음 변형도 나옵니다.

```bash
# 자동 수정(가능한 범위)
npx @synkra/aios-core doctor --fix

# 상세 출력
npx @synkra/aios-core doctor --verbose

# 특정 컴포넌트 점검
npx @synkra/aios-core doctor --component memory-layer
```

---

## 설치 후 구조 확인

AIOS는 프로젝트에 “프레임워크 코어” 디렉토리가 존재하는지로 많은 것을 판단합니다.

예시:

```bash
ls -la .aios-core/
ls -la .aios-core/core/
ls -la .aios-core/development/agents/
```

대표적인 감각:

```
.aios-core/
├── core/               # registry, health-check, orchestration 등
├── development/        # agents, tasks, workflows
├── product/            # templates, checklists
└── infrastructure/     # scripts, integrations
```

---

## 첫 에이전트 실행

설치가 끝났다면 IDE/Claude Code에서 `@`로 에이전트를 활성화합니다.

```
@aios-master
*help
```

다른 에이전트 예:

```
@dev
@qa
@architect
@pm
@sm
```

---

## 흔한 설치 문제 빠른 체크

문서가 제안하는 “가장 먼저 해볼 것”만 모으면 아래 정도입니다.

- `doctor`로 진단부터
- Node/npm 버전 확인
- npm 권한 문제(EACCES)면 npm prefix를 사용자 디렉토리로 변경
- 설치가 멈추면 registry/캐시/timeout 점검

특히 처음에는 “어디에 설치됐는지”가 핵심이라, `.aios-core/`가 현재 프로젝트 안에 생겼는지부터 확인하는 걸 추천합니다.

---

*다음 글에서는 에이전트 활성화(`@dev`)와 에이전트 커맨드(`*help`)가 어떻게 설계되어 있는지, 그리고 커맨드 권한/가시성 개념을 정리합니다.*
