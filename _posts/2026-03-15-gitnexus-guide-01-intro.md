---
layout: post
title: "GitNexus 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-15
permalink: /gitnexus-guide-01-intro/
author: abhigyanpatwari
categories: [GitHub Trending, GitNexus]
tags: [Trending, GitHub, GitNexus, MCP, KnowledgeGraph, CodeIntelligence]
original_url: "https://github.com/abhigyanpatwari/GitNexus"
excerpt: "코드베이스를 지식 그래프로 인덱싱해 AI 에이전트가 의존성/콜체인을 놓치지 않게 만드는 GitNexus의 목표, 사용 모드(CLI+MCP vs Web UI), 주의사항을 README 근거로 정리합니다."
---

## 이 문서의 목적

- GitNexus가 해결하려는 문제(“에이전트가 구조를 놓친다”)와 접근법(사전 계산된 그래프 인텔리전스)을 README 근거로 정리합니다. (`README.md`)
- 사용 모드(“CLI + MCP” vs “Web UI”)를 구분해, 어떤 상황에 무엇을 쓰는지 기준을 잡습니다. (`README.md`)

---

## 빠른 요약(README 기반)

- GitNexus는 코드베이스를 **지식 그래프**로 인덱싱하고(의존성/콜체인/클러스터/프로세스), 그 결과를 **MCP 도구**로 노출해 에이전트가 한 번의 호출로 “구조화된 컨텍스트”를 받도록 설계합니다. (`README.md`)
- 두 사용 모드:
  - **CLI + MCP**: 로컬 인덱싱 + 에이전트 통합(권장) (`README.md`)
  - **Web UI**: 브라우저에서 그래프 탐색/채팅(제한 있음) (`README.md`, `gitnexus-web/`)
- 패키지/엔진 정보(근거): Node 엔진 요구(`node >=18`), MCP SDK 의존성, KuzuDB/tree-sitter 의존성 (`gitnexus/package.json`)

---

## CLI + MCP vs Web UI(README 표 요약)

README는 두 모드를 비교 표로 제시합니다. (`README.md`)

- **CLI + MCP**: 로컬에서 리포지토리를 인덱싱하고, MCP 서버로 에이전트에 도구를 제공합니다.
- **Web UI**: 브라우저에서 그래프 탐색/채팅을 제공하며, 브라우저 메모리 제약이 언급됩니다.
- **Bridge mode**: `gitnexus serve`로 로컬 서버를 띄우면 Web UI가 로컬 인덱스를 재사용하는 흐름을 설명합니다. (`README.md`)

---

## “무엇을 제공하나?”(README의 MCP tools/resources)

README는 MCP로 노출되는 도구로 `list_repos`, `query`, `context`, `impact`, `detect_changes`, `rename`, `cypher`를 소개합니다. (`README.md`)

이 시리즈에서는 다음 챕터에서:
- 설치/인덱싱을 먼저 재현하고(02)
- 파이프라인/코드 구조를 확인한 뒤(03)
- MCP 통합과 운영 자동화로 확장합니다(04~05).

---

## 주의사항/함정(README 기반)

- README 상단에 “GitNexus 이름의 토큰/코인과 무관” 경고가 있습니다. (`README.md`)
- 패키지 라이선스는 `gitnexus/package.json`에 `PolyForm-Noncommercial-1.0.0`로 표시됩니다. (`gitnexus/package.json`)

---

## 근거(파일/경로)

- 전체 개요/모드 비교/도구 목록: `README.md`
- CLI 패키지/엔진/의존성: `gitnexus/package.json`
- MCP 샘플 설정: `.mcp.json`

---

## TODO/확인 필요

- CLI가 생성/사용하는 인덱스 디렉토리(`.gitnexus/`)와 글로벌 레지스트리(`~/.gitnexus/registry.json`)의 실제 구현 위치를 코드에서 확인하고 문서화 (`README.md`는 설명, 코드는 `gitnexus/src/storage/*` 추정)
- MCP tools의 입력 스키마가 코드에서 어떻게 정의되는지 확인 (`gitnexus/src/mcp/tools.ts` 등)

---

## 위키 링크

- `[[GitNexus Guide - Index]]` → [가이드 목차](/blog-repo/gitnexus-guide/)
- `[[GitNexus Guide - Install]]` → [02. 설치 및 분석](/blog-repo/gitnexus-guide-02-install-and-analyze/)

---

*다음 글에서는 README의 Quick Start(`npx gitnexus analyze`)를 기준으로 설치/인덱싱/상태 확인을 최소 경로로 정리합니다.*

