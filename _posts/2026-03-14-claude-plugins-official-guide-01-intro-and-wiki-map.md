---
layout: post
title: "claude-plugins-official 완벽 가이드 (01) - 소개 & 위키 맵"
date: 2026-03-14
permalink: /claude-plugins-official-guide-01-intro-and-wiki-map/
author: Anthropic
categories: [AI 코딩 에이전트, claude-plugins-official]
tags: [Trending, GitHub, claude-plugins-official, Claude Code, Plugins, Marketplace]
original_url: "https://github.com/anthropics/claude-plugins-official"
excerpt: "Claude Code Plugins Directory(공식) 저장소의 목적/구조/검증 포인트를 위키 맵으로 정리합니다."
---

## 이 문서의 목적

GitHub Trending(daily, 2026-03-14 기준) 상위에 오른 **anthropics/claude-plugins-official**을 “Claude Code 플러그인 디렉터리” 관점에서 빠르게 파악할 수 있도록 **탐색 지도(위키 맵)** 를 제공합니다.

---

## 빠른 요약

- 이 레포는 Claude Code에서 설치 가능한 플러그인을 **디렉터리(마켓플레이스)** 형태로 제공한다. (`README.md`)
- 플러그인 소스는 크게 2갈래:
  - 내부: `plugins/` (Anthropic 유지보수)
  - 외부: `external_plugins/` (파트너/커뮤니티/일부는 URL 소스) (`README.md`, `.claude-plugin/marketplace.json`)
- 설치는 Claude Code에서 `/plugin install ...@claude-plugin-directory` 또는 `/plugin > Discover`로 진행한다. (`README.md`)
- “플러그인 신뢰”가 핵심 리스크이며, 설치/업데이트 전에 플러그인이 어떤 MCP/파일/소프트웨어에 접근하는지 점검해야 한다. (`README.md`)

---

## (Trending 표시) 레포 메타

- **언어(Trending 표시)**: Python
- **오늘 스타(Trending 표시)**: +313
- **원본**: https://github.com/anthropics/claude-plugins-official

---

## 개념도: “디렉터리(카탈로그) → 설치(Claude Code) → 실행(플러그인 구성요소)”

```mermaid
flowchart TD
  U[사용자] -->|/plugin install| CC[Claude Code]
  CC -->|카탈로그 조회| MKT[.claude-plugin/marketplace.json]
  MKT -->|source: ./plugins/*| P1[내부 플러그인]
  MKT -->|source: ./external_plugins/*| P2[외부 플러그인(레포 내)]
  MKT -->|source: url| P3[외부 플러그인(원격 Git URL)]

  P1 --> PKG[plugin package]
  P2 --> PKG
  P3 --> PKG

  PKG --> C1[commands/]
  PKG --> C2[skills/]
  PKG --> C3[agents/]
  PKG --> C4[hooks/]
  PKG --> C5[.mcp.json]
```

---

## 근거(파일/경로)

- 레포 목적/구조/설치 안내/주의: `README.md`
- 마켓플레이스 카탈로그(플러그인 목록/메타/소스): `.claude-plugin/marketplace.json`
- 플러그인 디렉터리(내부): `plugins/`
- 플러그인 디렉터리(외부): `external_plugins/`

---

## (위키 맵) 이 시리즈 문서 구조

```text
01. 소개 & 위키 맵
02. 설치 & Discover (설치 전 점검 포함)
03. marketplace.json & 메타데이터 (source=로컬/외부/URL)
04. 플러그인 패키지 구조 (.claude-plugin/plugin.json, .mcp.json, commands/skills)
05. example-plugin 딥다이브 (커맨드/스킬/MCP)
06. LSP 플러그인 딥다이브 (lspServers + typescript/pyright 설치)
07. 외부 MCP 플러그인 딥다이브(GitHub) (토큰/헤더/URL)
08. 보안·문제해결·자동화 (PR 정책/CI 검증/운영 체크리스트)
```

---

## 주의사항/함정

- **플러그인은 강한 권한(도구/MCP/파일/프로세스)을 가질 수 있으므로** “신뢰”가 전제입니다. 레포 README가 이를 명시적으로 경고합니다. (`README.md`)
- 외부 플러그인의 경우 “레포에 포함된 구성”과 “홈페이지/원격 소스의 실제 내용”이 다를 수 있으니, 설치 전에 `source`와 실제 코드를 함께 확인해야 합니다. (`.claude-plugin/marketplace.json`)

---

## TODO/확인 필요

- Claude Code에서 `@claude-plugin-directory`가 어떤 검증(서명/해시/승인)을 수행하는지 여부는 이 레포만으로 단정하기 어렵습니다. 필요 시 공식 문서에서 확인하세요. (`README.md`의 문서 링크)

---

## 위키 링크

- `[[Claude Plugins Official Guide - Index]]` → [가이드 목차](/blog-repo/claude-plugins-official-guide/)

---

*다음 글에서는 Claude Code에서 플러그인을 설치/탐색하는 흐름과, 설치 전 점검 체크리스트를 정리합니다.*
