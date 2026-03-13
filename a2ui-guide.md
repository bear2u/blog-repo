---
layout: page
title: A2UI 가이드
permalink: /a2ui-guide/
icon: fas fa-robot
---

# 🧩 A2UI 완벽 가이드

> **Agent-to-User Interface: 에이전트가 “UI를 말하게” 만드는 선언형 프로토콜**

**google/A2UI**는 에이전트가 JSON 형태의 선언형 메시지로 UI를 기술하고, 클라이언트가 이를 **신뢰 가능한 컴포넌트 카탈로그**로 렌더링하는 방식의 프로토콜/스펙 + 렌더러/SDK/툴체인을 제공합니다.

---

## 📚 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 & 위키 맵](/blog-repo/a2ui-guide-01-intro-and-wiki-map/) | 목적/철학/구성요소, 문서 트리 안내 |
| 02 | [시작하기(샘플 실행)](/blog-repo/a2ui-guide-02-getting-started-and-samples/) | README/Quickstart 기반 데모 실행 |
| 03 | [스펙 & 데이터 모델](/blog-repo/a2ui-guide-03-spec-and-data-model/) | 메시지 타입, JSON Schema, 버전(v0.8/v0.9) |
| 04 | [렌더러 & 통합](/blog-repo/a2ui-guide-04-renderers-and-integrations/) | `renderers/*`, `agent_sdks/*`, transport 개요 |
| 05 | [툴링/기여/자동화](/blog-repo/a2ui-guide-05-tooling-contrib-automation/) | Composer/Editor/Inspector, mkdocs, CI |

---

## 빠른 시작 (문서/샘플 기준)

```bash
git clone https://github.com/google/A2UI.git
cd A2UI

# 샘플은 Gemini API Key가 필요하다고 안내됨 (README.md, docs/quickstart.md)
export GEMINI_API_KEY="..."
```

---

## 관련 링크

- GitHub 저장소: https://github.com/google/A2UI
- 문서 소스(레포): `docs/`, `mkdocs.yaml`
- 스펙/스키마(레포): `specification/`

