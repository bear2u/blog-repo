---
layout: post
title: "GitNexus 완벽 가이드 (02) - 설치 및 분석(Quick Start)"
date: 2026-03-15
permalink: /gitnexus-guide-02-install-and-analyze/
author: abhigyanpatwari
categories: [GitHub Trending, GitNexus]
tags: [Trending, GitHub, GitNexus, Quickstart, CLI, MCP]
original_url: "https://github.com/abhigyanpatwari/GitNexus"
excerpt: "README의 `npx gitnexus analyze` / `gitnexus setup` 흐름을 따라, 로컬 인덱싱→MCP 설정→에디터 연결까지의 최소 경로를 정리합니다."
---

## 이 문서의 목적

- “일단 돌아가게” 만드는 최소 경로(인덱싱 + MCP 설정)를 README 근거로 정리합니다. (`README.md`)
- CLI가 무엇을 제공하는지(명령/엔진 요구사항)를 패키지 근거로 확인합니다. (`gitnexus/package.json`)

---

## 빠른 요약(README/package.json 기반)

- Node 엔진: `gitnexus/package.json`의 `engines.node >= 18.0.0`
- Quick Start(README): 리포 루트에서 `npx gitnexus analyze`
- MCP 설정(README): `gitnexus setup` 또는 수동 설정(Claude Code/Cursor/OpenCode 예시)

---

## 1) Quick Start: 분석(인덱싱)

README의 최소 예시는 다음입니다. (`README.md`)

```bash
# 인덱싱할 리포의 루트에서 실행
npx gitnexus analyze
```

---

## 2) MCP 설정: 자동/수동

README는 `gitnexus setup`가 에디터를 자동 감지해 글로벌 MCP 설정을 작성한다고 설명합니다. (`README.md`)

```bash
gitnexus setup
```

수동 설정 예시(README):

- Claude Code:

```bash
claude mcp add gitnexus -- npx -y gitnexus@latest mcp
```

- Cursor: `~/.cursor/mcp.json`
- OpenCode: `~/.config/opencode/config.json`

또한 레포에는 MCP 샘플 설정 파일이 포함되어 있습니다: `.mcp.json` (`.mcp.json`)

---

## 3) 상태/목록/정리(README의 CLI Commands 참고)

README는 `gitnexus list`, `gitnexus status`, `gitnexus clean` 등 관리 명령을 안내합니다. (`README.md`)

> 정확한 옵션/출력은 CLI 구현(`gitnexus/src/cli/*`)을 기준으로 확인하는 것이 안전합니다.

---

## 근거(파일/경로)

- Quick Start / MCP 설정 예시 / CLI Commands: `README.md`
- Node 엔진/CLI bin: `gitnexus/package.json`
- MCP 샘플 설정: `.mcp.json`

---

## 주의사항/함정

- Web UI는 브라우저 메모리 제약이 언급되며, 대규모 리포는 CLI+백엔드(bridge/serve) 경로가 유리할 수 있습니다. (`README.md`)
- 라이선스가 `PolyForm-Noncommercial-1.0.0`로 명시되어 있으므로, 사용/배포 정책은 라이선스를 확인해야 합니다. (`gitnexus/package.json`)

---

## TODO/확인 필요

- 인덱스 저장 위치(`.gitnexus/`)와 레지스트리(`~/.gitnexus/registry.json`)를 코드에서 확인하고 “무엇이 저장되는지” 정리 (`gitnexus/src/storage/*` 추정)
- `gitnexus analyze`가 실행하는 실제 파이프라인 엔트리포인트 추적 (`gitnexus/src/cli/analyze.ts`)

---

## 위키 링크

- `[[GitNexus Guide - Index]]` → [가이드 목차](/blog-repo/gitnexus-guide/)
- `[[GitNexus Guide - Architecture]]` → [03. 아키텍처](/blog-repo/gitnexus-guide-03-architecture/)

---

*다음 글에서는 README가 설명하는 “인덱싱 파이프라인 단계(Structure/Parsing/Resolution/Clustering/Processes/Search)”를 코드 모듈로 매핑합니다.*

