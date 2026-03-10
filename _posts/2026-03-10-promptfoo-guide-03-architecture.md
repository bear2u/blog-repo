---
layout: post
title: "promptfoo 완벽 가이드 (03) - 핵심 개념과 아키텍처"
date: 2026-03-10
permalink: /promptfoo-guide-03-architecture/
author: "Ian Webster"
categories: [개발 도구, promptfoo]
tags: [Trending, GitHub, promptfoo]
original_url: "https://github.com/promptfoo/promptfoo"
excerpt: "promptfoo를 CLI·설정·리포트 뷰어 관점으로 잡습니다."
---

## 빠른 개념 정리

- **CLI 엔트리**: `package.json`의 `bin.promptfoo` → `dist/src/entrypoint.js`
- **웹 UI(앱)**: `workspaces`에 `src/app`가 포함(로컬에서 결과를 보기 위한 UI)
- **서버/백엔드 개발 모드**: `scripts.dev:server`가 `src/server/index.ts`를 사용
- **설정 파일**: 도구/커맨드 문맥에서 기본값으로 `promptfooconfig.yaml`를 사용하도록 구현된 부분이 존재

---

## 구성요소 흐름(개념도)

```mermaid
flowchart LR
  A[promptfooconfig.yaml] --> B[promptfoo CLI]
  B --> C[Eval Runner]
  C --> D[결과/리포트]
  D --> E[promptfoo view (Web UI)]
```

---

## 다음에 볼 것(코드/문서)

- `README.md`: Quick Start / What can you do
- `package.json`: `bin`, `workspaces`, `scripts`
- `src/server/index.ts`, `src/app/`: 로컬 뷰어/웹 UI 구성

---

*다음 글에서는 실전 사용 패턴을 정리합니다.*

