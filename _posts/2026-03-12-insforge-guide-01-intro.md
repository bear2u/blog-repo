---
layout: post
title: "InsForge 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-12
permalink: /insforge-guide-01-intro/
author: InsForge
categories: [개발 도구, insforge]
tags: [Trending, GitHub, insforge, Backend, MCP, GitHub Trending]
original_url: "https://github.com/InsForge/InsForge"
excerpt: "에이전트 네이티브(Agent-Native) Supabase 대안을 표방하는 InsForge의 개요와 빠른 시작을 정리합니다."
---

## InsForge란?

GitHub Trending(daily, 2026-03-12 기준) 상위에 오른 **InsForge/InsForge**를 한국어로 정리합니다.

- **한 줄 요약(Trending 표시)**: Give agents everything they need to ship fullstack apps. The backend built for agentic development.
- **언어(Trending 표시)**: TypeScript
- **오늘 스타(Trending 표시)**: +260
- **원본**: https://github.com/InsForge/InsForge

---

## 이 문서의 목적

- “무슨 제품인가 / 어떤 기능 묶음인가”를 한국어 README 기준으로 요약합니다.
- Docker 기반 빠른 시작과, AI Agent 연결(MCP 설정) 흐름을 다음 챕터 범위로 확정합니다.

---

## 빠른 요약 (i18n/README.ko 기반)

- InsForge는 “에이전트 네이티브(Agent-Native) Supabase 대안”을 표방합니다. (`i18n/README.ko.md`)
- 핵심 기능 묶음으로 Authentication/Database/Storage/Serverless Functions를 나열합니다. (`i18n/README.ko.md`)
- 빠른 시작은 Docker compose 실행 흐름(클론→`.env`→`docker compose up`)으로 안내합니다. (`i18n/README.ko.md`, `docker-compose.yml`)
- 대시보드에서 “Connect” 가이드를 따라 MCP 연결을 설정하도록 안내합니다. (`i18n/README.ko.md`)

---

## 빠른 시작(README 안내)

```bash
git clone https://github.com/insforge/insforge.git
cd insforge
cp .env.example .env
docker compose up
```

---

## 근거(파일/경로)

- 한국어 개요/빠른 시작: `i18n/README.ko.md`
- 개발자 가이드/규칙: `AGENTS.md`, `CONTRIBUTING.md`
- 실행/배포 단서: `docker-compose.yml`, `Dockerfile`
- 백엔드/프론트 주요 경로: `backend/`, `frontend/`
- 함수/스키마/API: `functions/`, `shared-schemas/`, `openapi/`

---

## 레포 구조(상위)

```text
InsForge/
  backend/
  frontend/
  functions/
  openapi/
  shared-schemas/
  docker-compose.yml
  package.json
```

---

## 위키 링크

- `[[InsForge Guide - Index]]` → [가이드 목차](/blog-repo/insforge-guide/)

---

*다음 글에서는 로컬에서 “대시보드 접속→MCP 연결”까지의 최소 경로를 정리합니다.*

