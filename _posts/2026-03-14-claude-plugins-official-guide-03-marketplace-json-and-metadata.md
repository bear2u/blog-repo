---
layout: post
title: "claude-plugins-official 완벽 가이드 (03) - marketplace.json & 메타데이터"
date: 2026-03-14
permalink: /claude-plugins-official-guide-03-marketplace-json-and-metadata/
author: Anthropic
categories: [AI 코딩 에이전트, claude-plugins-official]
tags: [Trending, GitHub, claude-plugins-official, Marketplace, Claude Code, Metadata]
original_url: "https://github.com/anthropics/claude-plugins-official"
excerpt: "플러그인 목록의 단일 근원인 marketplace.json 구조와 source(로컬/URL) 패턴을 정리합니다."
---

## 이 문서의 목적

이 레포에서 “무슨 플러그인이 존재하는지”를 결정하는 핵심 파일인 `.claude-plugin/marketplace.json`을 기준으로, 플러그인 엔트리가 어떤 필드로 구성되고 어떤 “소스 형태(source)”를 갖는지 파악합니다.

---

## 빠른 요약

- 카탈로그 파일: `.claude-plugin/marketplace.json`
- 최상위 키(파일 기준): `$schema`, `name`, `description`, `owner`, `plugins`
- 플러그인 목록: `plugins[]` (이 레포 기준 다수 엔트리)
- 엔트리 핵심 필드(대표): `name`, `description`, `category`, `source`, `homepage`, (선택) `version`, `author`, `tags`, `lspServers`
- `source`는 “레포 내부 경로”뿐 아니라 “원격 Git URL” 객체도 지원한다. (파일 기준)

---

## `.claude-plugin/marketplace.json`의 역할

파일 상단은 스키마와 디렉터리 자체 정보를 선언합니다.

- `$schema`: `https://anthropic.com/claude-code/marketplace.schema.json`
- `name`: `"claude-plugins-official"`
- `owner`: Anthropic(이메일 포함)

그리고 실제 설치/탐색의 근간은 `plugins` 배열입니다.

---

## 플러그인 엔트리 구조(핵심 필드)

이 레포에서 관찰되는 엔트리 필드(파일 기준)는 다음과 같습니다.

- `name`: 설치 이름(예: `github`, `pyright-lsp`)
- `description`: 목록에서 보이는 설명
- `category`: 분류(예: `development`, `productivity`, `testing`, `database`, `design`, `deployment`, `monitoring`)
- `source`: 설치 소스(경로 또는 URL)
- `homepage`: 문서/신뢰 확인용 링크
- `tags`: 일부 플러그인에만 존재(예: `community-managed`)
- `lspServers`: LSP 플러그인에서 언어 서버 실행 설정 포함

---

## `source`의 3가지 패턴(중요)

### 1) 레포 내부: `./plugins/...`

예: `pyright-lsp` 엔트리는 `source: "./plugins/pyright-lsp"` 형태를 가집니다. (`.claude-plugin/marketplace.json`)

### 2) 레포 내부: `./external_plugins/...`

예: `github` 엔트리는 `source: "./external_plugins/github"` 형태이며, 해당 폴더에 `.claude-plugin/plugin.json`과 `.mcp.json`이 존재합니다. (`.claude-plugin/marketplace.json`, `external_plugins/github/`)

### 3) 원격 Git URL: `{ "source": "url", "url": "https://..." }`

예: `figma`, `Notion`, `slack`, `vercel` 등의 엔트리에서 `source`가 객체로 나타납니다. (`.claude-plugin/marketplace.json`)

이 경우 실제 설치 대상 코드는 **이 레포 바깥**에 있으므로, `homepage`와 `url`을 함께 검토해야 합니다.

---

## LSP 플러그인: `lspServers`가 의미하는 것

`typescript-lsp`, `pyright-lsp` 같은 엔트리에는 `lspServers`가 들어있습니다. (`.claude-plugin/marketplace.json`)

- `command`: 실행 파일(예: `pyright-langserver`)
- `args`: `--stdio` 등의 인자(플러그인마다 다름)
- `extensionToLanguage`: 확장자 → 언어 모드 매핑

즉, LSP 플러그인은 “레포 내 코드”라기보다 **(이미 설치된) 언어 서버 바이너리를 Claude Code가 실행할 수 있게 하는 설정** 성격이 강합니다.

---

## Mermaid: 카탈로그 중심 설치 해석 흐름(개념)

```mermaid
flowchart LR
  CC[Claude Code] -->|read| MKT[marketplace.json]
  MKT -->|plugins[].source| SRC{source type?}
  SRC -->|./plugins/*| LOCAL1[internal plugin dir]
  SRC -->|./external_plugins/*| LOCAL2[external plugin dir]
  SRC -->|url object| REMOTE[remote git url]
  LOCAL2 -->|.mcp.json| MCP[auth/env/url review]
```

---

## 근거(파일/경로)

- 카탈로그 파일(스키마/플러그인 엔트리): `.claude-plugin/marketplace.json`
- 외부 GitHub MCP 플러그인(레포 내 소스): `external_plugins/github/.claude-plugin/plugin.json`, `external_plugins/github/.mcp.json`
- LSP 플러그인 엔트리(카탈로그 내): `.claude-plugin/marketplace.json`의 `typescript-lsp`, `pyright-lsp`

---

## 주의사항/함정

- `source`가 URL인 플러그인은 설치 시점에 “원격 레포”를 가져오는 형태일 수 있으므로, **버전 고정/검증 방식**을 별도로 확인해야 합니다. (이 파일만으로는 검증 로직이 드러나지 않음)
- `name`이 대소문자를 포함하는 엔트리(예: `Notion`)가 존재합니다. 이름은 카탈로그 값 그대로 취급되는 편이 안전합니다. (`.claude-plugin/marketplace.json`)

---

## TODO/확인 필요

- `category` 값과 Claude Code UI의 실제 분류/필터링이 1:1로 일치하는지(또는 내부 매핑이 있는지)는 공식 문서 확인이 필요합니다.

---

## 위키 링크

- `[[Claude Plugins Official Guide - Index]]` → [가이드 목차](/blog-repo/claude-plugins-official-guide/)
- `[[Claude Plugins Official Guide - Plugin Package Structure]]` → [04. 플러그인 패키지 구조](/blog-repo/claude-plugins-official-guide-04-plugin-package-structure/)

---

*다음 글에서는 실제 플러그인 디렉터리가 어떤 파일로 구성되는지(필수/선택)와, `commands/`·`skills/`·`.mcp.json` 같은 확장 포인트를 구체적으로 설명합니다.*

