---
layout: post
title: "A2UI 완벽 가이드 (02) - 시작하기(샘플 실행)와 레포 탐색"
date: 2026-03-13
permalink: /a2ui-guide-02-getting-started-and-samples/
author: google
categories: [AI 에이전트, A2UI]
tags: [Trending, GitHub, a2ui, Quickstart, Samples]
original_url: "https://github.com/google/A2UI"
excerpt: "README와 docs/quickstart.md를 기준으로 레스토랑 파인더 데모를 실행하는 경로와, 레포에서 샘플/렌더러/스펙이 어디에 있는지 연결합니다."
---

## 전제(문서 기준)

`README.md`와 `docs/quickstart.md`는 샘플 실행에 필요한 전제를 명시합니다.

- Node.js(웹 클라이언트)
- Python(에이전트 샘플)
- 샘플은 **Gemini API Key**가 필요할 수 있음 (`GEMINI_API_KEY`)

---

## 레포 클론 및 환경 변수

```bash
git clone https://github.com/google/A2UI.git
cd A2UI

export GEMINI_API_KEY="your_gemini_api_key"
```

---

## Quickstart(문서)에서 안내하는 흐름

`docs/quickstart.md`는 Lit 기반 데모를 “원 커맨드”로 실행하는 형태를 안내합니다.

```bash
cd samples/client/lit
npm install
npm run demo:all
```

이 과정은 “렌더러 빌드 + 에이전트(백엔드) 실행 + dev 서버 실행”을 묶는다고 문서가 설명합니다. (`docs/quickstart.md`)

---

## 레포에서 ‘어디에 뭐가 있나’

빠르게 위치를 기억하기 위한 지도:

```text
A2UI/
  docs/                 # MkDocs 문서 소스 (mkdocs.yaml nav)
  specification/        # 스펙/버전, json schema, 검증 스크립트
  renderers/            # 클라이언트 렌더러(web core, lit, angular, react, ...)
  agent_sdks/           # 에이전트/서버 측 SDK(java/python)
  samples/              # 데모/샘플(클라이언트/에이전트)
  tools/                # composer/editor/inspector 등
```

---

## 근거(파일/경로)

- 샘플 실행/철학: `README.md`
- Quickstart: `docs/quickstart.md`
- docs 네비게이션: `mkdocs.yaml`
- 샘플 위치: `samples/`
- 렌더러: `renderers/`

---

## 위키 링크

- `[[A2UI Guide - Index]]` → [가이드 목차](/blog-repo/a2ui-guide/)
- `[[A2UI Guide - Spec]]` → [03. 스펙 & 데이터 모델](/blog-repo/a2ui-guide-03-spec-and-data-model/)

---

*다음 글에서는 `specification/`와 `docs/`를 연결해서 A2UI 메시지/데이터 모델을 정리합니다.*

