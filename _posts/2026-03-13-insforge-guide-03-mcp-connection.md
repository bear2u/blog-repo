---
layout: post
title: "InsForge 완벽 가이드 (03) - 대시보드 & MCP 연결"
date: 2026-03-13
permalink: /insforge-guide-03-mcp-connection/
author: InsForge
categories: [개발 도구, insforge]
tags: [Trending, GitHub, insforge, MCP, Dashboard, GitHub Trending]
original_url: "https://github.com/InsForge/InsForge"
excerpt: "README.md 및 i18n/README.ko.md의 'Connect InsForge MCP' 안내를 근거로 에이전트 연결 흐름(대시보드 중심)을 정리합니다."
---

## 이 문서의 목적

- InsForge에서 말하는 “MCP 연결”을 **대시보드 기준 사용자 플로우**로 정리합니다.
- (중요) 실제 MCP 설정의 상세 스니펫은 대시보드 UI에 의존하므로, 이 문서는 “어디서 무엇을 따라야 하는지”를 확정합니다.

---

## 빠른 요약 (문서 근거)

- 한국어 README는 “InsForge Dashboard(기본 `http://localhost:7131`) 로그인 → ‘Connect’ 가이드 따라 MCP 설정”을 안내합니다. (`i18n/README.ko.md`)
- 영문 README도 “Connect InsForge MCP Server” 절차를 별도 섹션으로 안내합니다. (`README.md`)
- 연결 테스트용 프롬프트 예시가 제시됩니다. (`i18n/README.ko.md`)

---

## 1) 대시보드 접속/로그인

대시보드 기본 주소:

- `http://localhost:7131`

근거:
- `i18n/README.ko.md`

---

## 2) “Connect” 가이드 따라 MCP 설정

문서가 확정하는 사실:

- MCP 설정은 대시보드의 “Connect” 가이드에서 수행한다.
- 즉, 레포 파일만으로 “정확히 어떤 JSON/설정 블록을 어디에 넣는지”는 고정할 수 없고, UI를 따라야 한다.

근거:
- `i18n/README.ko.md`
- `README.md`

---

## 3) 연결 테스트(문서 예시)

한국어 README는 다음 메시지를 에이전트에게 전송해 연결을 확인하라고 합니다.

```
InsForge is my backend platform, what is my current backend structure?
```

근거:
- `i18n/README.ko.md`

---

## 4) “연결 후 무엇이 가능한가?”(README 설명)

영문 README는 InsForge를 “AI coding agents를 위한 백엔드 플랫폼”으로 설명하며, 에이전트가 백엔드 컨텍스트를 가져오고(configure/inspect) 조작할 수 있다고 설명합니다. (`README.md`)

또한 “InsForge MCP's fetch-docs tool”로 문서/지침을 불러오라는 예시도 포함합니다. (`README.md`)

---

## 주의사항/함정

- MCP 연결은 “InsForge가 로컬에서 정상 기동 + 대시보드 로그인 가능”이 전제입니다. 먼저 02장에서 compose가 정상인지 확인하세요. (`docker-compose.yml`)
- 연결 실패 시 가장 흔한 원인은 “대시보드/백엔드 포트 충돌(7130~7132)” 또는 “API 키/접속 정보 불일치”입니다(구체 설정 값은 대시보드 UI에서 확인).

---

## TODO / 확인 필요

- MCP 서버의 프로토콜/엔드포인트(예: stdio, SSE, HTTP 등)와 실제 도구 목록/스키마는 레포에 `mcp/` 워크스페이스가 언급되지만(루트 `package.json`), 현재 체크아웃에는 디렉토리가 보이지 않습니다. 실제 구현 위치를 최신 레포 기준으로 재확인하고, “도구 목록(이름/입력/출력)”을 문서화하면 이 챕터가 더 실무적으로 완성됩니다. (근거: 루트 `package.json`의 `workspaces`)

---

## 위키 링크

- `[[InsForge Guide - Index]]` → [가이드 목차](/blog-repo/insforge-guide/)
- `[[InsForge Guide - Docker]]` → [02. 설치 및 실행(Docker)](/blog-repo/insforge-guide-02-docker/)
- `[[InsForge Guide - Architecture]]` → [04. 구성요소/아키텍처](/blog-repo/insforge-guide-04-architecture/)

---

*다음 글에서는 디렉토리(backend/frontend/auth/functions)와 compose를 근거로 InsForge 아키텍처를 정리합니다.*

