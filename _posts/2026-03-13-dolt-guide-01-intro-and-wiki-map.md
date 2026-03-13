---
layout: post
title: "Dolt 완벽 가이드 (01) - 소개 & 위키 맵"
date: 2026-03-13
permalink: /dolt-guide-01-intro-and-wiki-map/
author: dolthub
categories: [개발 도구, 데이터베이스]
tags: [Trending, GitHub, dolt, Database, SQL]
original_url: "https://github.com/dolthub/dolt"
excerpt: "Git처럼 clone/branch/merge/push/pull 가능한 SQL 데이터베이스 Dolt를 레포 구조와 운영 관점으로 위키형 정리합니다."
---

## Dolt란?

GitHub Trending(daily, 2026-03-13 기준) 상위에 오른 **dolthub/dolt**를 레포(코드/문서) 기준으로 정리합니다.

- **한 줄 요약(Trending 표시)**: Dolt – Git for Data
- **언어(Trending 표시)**: Go
- **오늘 스타(Trending 표시)**: +58
- **원본**: https://github.com/dolthub/dolt

---

## Dolt의 “두 얼굴”: CLI와 SQL

`README.md`가 설명하는 핵심은 명확합니다.

- Dolt는 **MySQL 호환**으로 연결할 수 있는 SQL 데이터베이스이며,
- Git처럼 **fork/clone/branch/merge/push/pull**을 데이터(테이블)에 적용합니다.

따라서 문서를 읽을 때는 “CLI로 하는 버전관리”와 “SQL로 노출되는 버전관리”를 같이 보게 됩니다. (`README.md`의 설명)

---

## 근거(파일/경로) — 이 시리즈가 기대는 것

- 개요/설치/CLI 개괄: `README.md`
- CLI 엔트리/명령 구조: `go/cmd/dolt/` (예: `go/cmd/dolt/dolt.go`, `go/cmd/dolt/commands/`)
- 핵심 라이브러리(도메인 로직): `go/libraries/doltcore/`
- 스토리지/저장 구조 힌트: `go/store/`
- Docker 문서: `docker/README.md`, `docker/serverREADME.md`
- 워크플로우/자동화 문서: `.github/workflows/README.md`

---

## (위키 맵) 문서 구조

```text
01. 소개 & 위키 맵
02. 설치 & 퀵스타트 (CLI/SQL/서버 모드)
03. 아키텍처(레포 구조) (cmd ↔ doltcore ↔ store)
04. 버전관리 워크플로우 (commit/branch/merge/remote, server/docker)
05. 운영/테스트/자동화 (Actions, integration-tests, troubleshooting)
```

---

## 위키 링크

- `[[Dolt Guide - Index]]` → [가이드 목차](/blog-repo/dolt-guide/)
- `[[Dolt Guide - Install & Quickstart]]` → [02. 설치 & 퀵스타트](/blog-repo/dolt-guide-02-install-and-quickstart/)

---

*다음 글에서는 설치/초기화/SQL 실행/서버 모드까지 “처음 성공하기” 플로우를 정리합니다.*

