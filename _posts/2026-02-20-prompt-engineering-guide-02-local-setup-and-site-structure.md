---
layout: post
title: "Prompt Engineering Guide 가이드 (02) - 로컬 실행과 사이트 구조"
date: 2026-02-20
permalink: /prompt-engineering-guide-02-local-setup-and-site-structure/
author: Elvis Saravia (DAIR.AI)
categories: ['LLM 학습', '프롬프트 엔지니어링']
tags: [Next.js, Nextra, MDX, Documentation]
original_url: "https://github.com/dair-ai/Prompt-Engineering-Guide"
excerpt: "Node/pnpm 기반 로컬 실행 방법과 pages 중심 문서 구조를 이해해 빠르게 탐색 가능한 상태를 만듭니다."
---

## 로컬 실행

README 기준 로컬 실행 절차는 단순합니다.

```bash
# Node 18 이상
pnpm i
pnpm dev
```

기본 접속 주소는 `http://localhost:3000`입니다.

---

## 기술 스택

`package.json` 기준 문서 사이트는 아래 스택을 사용합니다.

- `next`
- `nextra`
- `nextra-theme-docs`
- `react`

즉, 정적 Markdown 모음이 아니라 **Next.js 기반 문서 애플리케이션**입니다.

---

## 문서 구조 읽는 법

`pages/`는 다국어 MDX 파일을 사용합니다.

- `pages/_meta.en.json`: 영어 기준 전체 내비게이션
- `pages/introduction/*`: 입문 내용
- `pages/techniques/*`: 프롬프팅 기법
- `pages/agents/*`: 에이전트/컨텍스트 엔지니어링
- `pages/guides/*`: 최신 가이드 모음

메타 파일을 먼저 읽고 필요한 챕터만 들어가는 방식이 학습 효율이 높습니다.

---

## 추천 탐색 순서

1. `pages/_meta.en.json`으로 전체 지도 확인
2. `introduction -> techniques`로 기초 고정
3. `applications -> risks -> agents`로 실무 확장
4. `notebooks`로 실습 검증

다음 장에서는 프롬프트 기본기(요소/설정/작성 원칙)를 정리합니다.
