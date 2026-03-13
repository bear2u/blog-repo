---
layout: post
title: "A2UI 완벽 가이드 (04) - 렌더러 & 통합 (renderers, agent_sdks, transports)"
date: 2026-03-13
permalink: /a2ui-guide-04-renderers-and-integrations/
author: google
categories: [AI 에이전트, A2UI]
tags: [Trending, GitHub, a2ui, Renderer, SDK]
original_url: "https://github.com/google/A2UI"
excerpt: "A2UI의 클라이언트 렌더러(renderers)와 서버/에이전트 SDK(agent_sdks), 문서가 언급하는 transport 개념을 레포 구조 기준으로 연결합니다."
---

## 렌더러는 여러 프레임워크를 대상으로 한다

레포에는 다양한 렌더러가 디렉토리로 분리되어 있습니다.

- `renderers/web_core/`
- `renderers/lit/`
- `renderers/angular/`
- `renderers/react/`
- `renderers/markdown/` (README에서 언급되는 마크다운 렌더러 경로 포함)

각 렌더러는 `package.json`/`tsconfig.json`이 있는 독립 패키지 형태로 존재합니다. (예: `renderers/lit/package.json`)

---

## 에이전트 SDK(서버 측)도 별도로 존재

레포 최상위에 `agent_sdks/`가 있고, 언어별 SDK 구조가 보입니다.

- `agent_sdks/python/src/`
- `agent_sdks/java/src/`

README는 A2UI가 “어떤 LLM이든 JSON을 만들 수 있으면 생성 가능”하다고 설명하지만, 샘플 실행은 특정 SDK/샘플 구조를 통해 안내됩니다. (`README.md`, `docs/quickstart.md`)

---

## Transports(전송) 개념

문서 사이트는 “Transports”를 별도 개념으로 다룹니다. (`mkdocs.yaml`의 `concepts/transports.md` 네비게이션)

즉, A2UI는 “UI 포맷/스펙 + 렌더러/SDK”에 집중하고, 전송은 A2A 등 외부 프로토콜과 결합 가능한 형태로 설계된 것으로 보입니다. (`README.md`, `docs/index.md`)

---

## 통합 포인트를 한 장으로 요약

```mermaid
flowchart LR
  Agent[Agent SDK / App] -->|JSONL| Transport[Transport]
  Transport --> Client[Client Stream Reader]
  Client --> Parser[Message Parser + Catalog Resolver]
  Parser --> Renderer[Renderer (web_core + framework)]
  Renderer --> UI[Native UI]
  UI -->|userAction| Agent
```

---

## 근거(파일/경로)

- 렌더러 디렉토리: `renderers/`
- 에이전트 SDK: `agent_sdks/`
- 문서 네비게이션(Transports): `mkdocs.yaml`
- Quickstart 예시: `docs/quickstart.md`

---

## 위키 링크

- `[[A2UI Guide - Index]]` → [가이드 목차](/blog-repo/a2ui-guide/)
- `[[A2UI Guide - Tooling/Automation]]` → [05. 툴링/기여/자동화](/blog-repo/a2ui-guide-05-tooling-contrib-automation/)

---

*다음 글에서는 tools/와 문서/스펙 검증 워크플로우를 중심으로, “개발자 경험 + 문서 자동화”를 정리합니다.*

