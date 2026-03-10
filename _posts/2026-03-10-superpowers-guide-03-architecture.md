---
layout: post
title: "superpowers 완벽 가이드 (03) - 핵심 개념과 아키텍처"
date: 2026-03-10
permalink: /superpowers-guide-03-architecture/
author: obra
categories: [개발 도구, superpowers]
tags: [Trending, GitHub, superpowers]
original_url: "https://github.com/obra/superpowers"
excerpt: "스킬 라이브러리와 워크플로우 트리거 관점으로 구조를 봅니다."
---

## 레포 구성(파일/디렉토리 관점)

- `.codex/`: Codex용 설치/스킬 관련 문서
- `.claude-plugin/`, `.cursor-plugin/`: 각 플랫폼용 플러그인/설정
- `docs/`: 플랫폼별/상세 사용 문서(README에서 링크)

---

## 워크플로우(README 기반)

```mermaid
flowchart LR
  A[대화/요구사항] --> B[brainstorming]
  B --> C[writing-plans]
  C --> D[subagent-driven-development]
  D --> E[requesting-code-review]
  E --> F[finishing-a-development-branch]
```

---

*다음 글에서는 실전 사용 패턴을 정리합니다.*

