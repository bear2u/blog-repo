---
layout: post
title: "A2UI 완벽 가이드 (01) - 소개 & 위키 맵"
date: 2026-03-13
permalink: /a2ui-guide-01-intro-and-wiki-map/
author: google
categories: [AI 에이전트, A2UI]
tags: [Trending, GitHub, a2ui, UI, Protocol]
original_url: "https://github.com/google/A2UI"
excerpt: "에이전트가 선언형 JSON으로 UI를 기술하고, 클라이언트가 카탈로그 기반으로 안전하게 렌더링하는 A2UI의 목표/철학/구성을 레포 기준으로 정리합니다."
---

## A2UI란?

GitHub Trending(daily, 2026-03-13 기준) 상위에 오른 **google/A2UI**를 레포(README/문서/스펙) 기준으로 정리합니다.

- **한 줄 요약(Trending 표시)**: Agent-to-User Interface
- **언어(Trending 표시)**: TypeScript
- **오늘 스타(Trending 표시)**: +629
- **원본**: https://github.com/google/A2UI

---

## README가 말하는 핵심 철학(요약)

`README.md`는 A2UI를 다음처럼 정의합니다.

- 에이전트가 “UI를 말한다”: 선언형 JSON 포맷으로 UI 의도를 기술
- 클라이언트는 자체 UI 컴포넌트 카탈로그로 이를 렌더링
- 임의 코드 실행이 아니라 데이터 포맷이므로, “신뢰 경계(trust boundary)” 상황에서 안전성을 높임

---

## 근거(파일/경로)

- 프로젝트 개요/샘플 실행: `README.md`
- 공식 문서 소스(MkDocs): `docs/`, `mkdocs.yaml`
- 스펙/버전별 자료: `specification/` (예: `specification/v0_8/*`, `specification/v0_9/*`)
- 클라이언트 렌더러들: `renderers/`
- 에이전트 SDK들: `agent_sdks/`
- 툴링(Composer/Editor/Inspector): `tools/`
- 스펙 예제 검증 워크플로우: `.github/workflows/validate_specifications.yml`

---

## (위키 맵) 문서 구조

```text
01. 소개 & 위키 맵
02. 시작하기(샘플 실행) (README/Quickstart 기반)
03. 스펙 & 데이터 모델 (메시지 타입, JSON Schema, v0.8/v0.9)
04. 렌더러 & 통합 (renderers, agent_sdks, transports)
05. 툴링/기여/자동화 (tools, mkdocs, workflows)
```

---

## A2UI 데이터 흐름(개념)

```mermaid
flowchart LR
  U[User] --> A[Agent / LLM]
  A --> M[A2UI JSON 메시지]
  M --> T[Transport (JSONL 스트림)]
  T --> P[Client Parser]
  P --> R[Renderer (Lit/Angular/React/...)]
  R --> UI[Native UI Components]
```

이 그림은 `docs/concepts/data-flow.md`와 `docs/quickstart.md`가 설명하는 메시지 흐름을 “도식화”한 것입니다.

---

## 위키 링크

- `[[A2UI Guide - Index]]` → [가이드 목차](/blog-repo/a2ui-guide/)
- `[[A2UI Guide - Getting Started]]` → [02. 시작하기(샘플 실행)](/blog-repo/a2ui-guide-02-getting-started-and-samples/)

---

*다음 글에서는 레포가 제공하는 샘플/퀵스타트를 기준으로 “한 번 돌려보기” 플로우를 정리합니다.*

