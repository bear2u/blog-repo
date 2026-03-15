---
layout: post
title: "GitNexus 완벽 가이드 (04) - MCP 도구/에디터 통합"
date: 2026-03-15
permalink: /gitnexus-guide-04-mcp-tools-and-integration/
author: abhigyanpatwari
categories: [GitHub Trending, GitNexus]
tags: [Trending, GitHub, GitNexus, MCP, ClaudeCode, Cursor]
original_url: "https://github.com/abhigyanpatwari/GitNexus"
excerpt: "README의 MCP 설정 예시와 `.mcp.json`을 근거로, GitNexus를 Claude Code/Cursor/OpenCode 등에 연결하는 방법과 노출 도구(impact/query/context/…)의 역할을 정리합니다."
---

## 이 문서의 목적

- GitNexus를 “에이전트/IDE에 붙이는 방법”을 README 예시와 `.mcp.json` 근거로 정리합니다.
- MCP 도구가 어떤 범주를 커버하는지(검색/컨텍스트/임팩트/리네임/사이퍼)를 간단히 매핑합니다. (`README.md`)

---

## 빠른 요약(README/.mcp.json 기반)

- MCP 서버 실행 예시(레포 포함): `.mcp.json`
  - `command: npx`
  - `args: ["-y", "gitnexus@latest", "mcp"]`
- README는 Claude Code/Cursor/OpenCode 등 수동 설정 예시를 제공합니다. (`README.md`)

---

## 1) MCP 설정(레포의 예시 파일)

레포 루트의 `.mcp.json`은 다음처럼 GitNexus MCP 서버를 등록하는 형태입니다. (`.mcp.json`)

```json
{
  "mcpServers": {
    "gitnexus": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "gitnexus@latest", "mcp"]
    }
  }
}
```

---

## 2) Claude Code / Cursor / OpenCode 설정(README 예시)

README 예시(근거):

- Claude Code:

```bash
claude mcp add gitnexus -- npx -y gitnexus@latest mcp
```

- Cursor: `~/.cursor/mcp.json`
- OpenCode: `~/.config/opencode/config.json`

근거: `README.md`

---

## 3) MCP tools(README 열거)

README는 MCP로 노출되는 도구로 아래를 소개합니다. (`README.md`)

- `list_repos`
- `query`
- `context`
- `impact`
- `detect_changes`
- `rename`
- `cypher`

> 입력 스키마/응답 스키마는 구현(`gitnexus/src/mcp/tools.ts`)에서 확인하는 것이 안전합니다.

---

## 4) MCP 리소스/프롬프트(README 열거)

README는 리소스 URI(예: `gitnexus://repos`)와 MCP prompts(예: `detect_impact`, `generate_map`)도 언급합니다. (`README.md`)

운영 관점에서 “리포가 여러 개일 때 repo 파라미터가 필요” 같은 규칙도 README에 명시되어 있습니다. (`README.md`)

---

## 근거(파일/경로)

- MCP 설정 예시 파일: `.mcp.json`
- MCP 통합 설명/예시/도구 목록: `README.md`
- MCP tools 구현 위치: `gitnexus/src/mcp/tools.ts`

---

## 주의사항/함정

- `npx -y gitnexus@latest ...`는 최신 버전을 실행하므로, 재현성을 위해 특정 버전을 고정할지 정책이 필요할 수 있습니다. (README 예시 기반, 정책은 사용자 결정)
- 라이선스/사용 범위는 `gitnexus/package.json`의 라이선스 표기를 확인해야 합니다. (`gitnexus/package.json`)

---

## TODO/확인 필요

- `gitnexus/src/mcp/tools.ts`에서 각 tool의 입력/출력 스키마를 문서로 추출
- `gitnexus/src/mcp/resources.ts`의 리소스 목록과 README의 리소스 URI가 일치하는지 확인

---

## 위키 링크

- `[[GitNexus Guide - Index]]` → [가이드 목차](/blog-repo/gitnexus-guide/)
- `[[GitNexus Guide - Ops]]` → [05. Web UI·운영·자동화](/blog-repo/gitnexus-guide-05-web-ui-ops-doc-automation/)

---

*다음 글에서는 `gitnexus serve`와 Web UI, 그리고 `gitnexus wiki`/`detect_changes` 같은 운영 자동화 포인트를 정리합니다.*

