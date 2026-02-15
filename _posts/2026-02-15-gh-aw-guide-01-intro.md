---
layout: post
title: "GitHub Agentic Workflows(gh-aw) 가이드 (01) - 소개: Actions 위에서 에이전트를 안전하게 굴리기"
date: 2026-02-15
permalink: /gh-aw-guide-01-intro/
author: GitHub
categories: [개발 도구, GitHub]
tags: [gh-aw, GitHub Actions, GitHub CLI, Agentic Workflows, Safe Outputs, Security]
original_url: "https://github.com/github/gh-aw"
excerpt: "gh-aw는 자연어 마크다운 워크플로우를 GitHub Actions로 컴파일해 실행하는 GitHub CLI 확장입니다. 이 장에서는 전체 개념과 왜 safe-outputs가 핵심인지 정리합니다."
---

## gh-aw는 무엇인가

**GitHub Agentic Workflows(gh-aw)**는 GitHub CLI 확장(`gh aw`)으로,

- 자연어로 작성한 워크플로우(마크다운 + YAML frontmatter)
- 을 GitHub Actions 실행물(`.lock.yml`)로 컴파일해서
- 리포지토리 자동화를 “에이전트 방식”으로 구현할 수 있게 합니다.

여기서 중요한 관점은 “코딩 에이전트가 로컬에서 코드를 고치는 것”이 아니라,
**레포 운영(이슈/PR/릴리즈/모니터링/리서치)** 같은 작업을 GitHub Actions 안에서 자동화하는 방식이라는 점입니다.

---

## 왜 ‘Agentic Workflows’인가

Actions는 원래도 자동화 플랫폼이지만, 대부분은 “정해진 스크립트” 기반입니다.

gh-aw는 여기에 다음을 추가합니다.

1. **자연어 워크플로우**: 사람이 읽을 수 있는 정책/절차를 그대로 워크플로우 본문에 적는다
2. **컴파일/검증**: 설정(frontmatter)을 스키마로 검증하고 안전한 실행물로 변환한다
3. **가드레일**: 네트워크/권한/샌드박스 등 실행 경계를 명시한다

즉, “LLM이 똑똑하게 한다”가 아니라 “LLM이 **안전한 범위**에서만 하게 만든다”가 핵심입니다.

---

## safe-outputs: 쓰기 권한을 에이전트에서 분리한다

gh-aw 문서에서 가장 중요한 설계는 **safe-outputs**입니다.

- 에이전트(자연어 실행)는 기본적으로 **읽기 전용**으로 돌리고
- 이슈 생성/코멘트/PR 생성 같은 “쓰기”는
- 에이전트가 **구조화된 출력**(safe outputs)을 요청하면
- 별도의 권한을 가진 후처리(job)가 수행하는 방식입니다.

이 구조의 장점:

- 최소 권한(least privilege)
- 프롬프트 인젝션 방어에 유리
- 감사/추적이 쉽고, 위험한 작업을 제한하기 쉽다

---

## 이 시리즈에서 다룰 것

gh-aw는 “쓰기 좋은 워크플로우 문법”만이 아니라,

- 도구(toolsets/MCP)
- 네트워크 권한
- strict 모드
- 컴파일/락파일 운영
- 실행/디버깅(런 모드/로그)

같은 운영 요소가 중요합니다.

다음 장부터는 설치/초기화부터 차근차근 정리합니다.

---

*다음 글에서는 `gh aw` 설치와 `gh aw init`로 레포를 초기화하고, 엔진/시크릿을 어떻게 잡는지 설명합니다.*

