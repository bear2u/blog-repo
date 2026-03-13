---
layout: post
title: "Dolt 완벽 가이드 (02) - 설치 & 퀵스타트 (CLI/SQL/서버)"
date: 2026-03-13
permalink: /dolt-guide-02-install-and-quickstart/
author: dolthub
categories: [개발 도구, 데이터베이스]
tags: [Trending, GitHub, dolt, CLI, SQL]
original_url: "https://github.com/dolthub/dolt"
excerpt: "README와 docker 문서를 기준으로 Dolt 설치, repo 초기화, SQL 실행, sql-server(로컬/도커)까지 한 번에 연결합니다."
---

## 설치(README 기준)

`README.md`는 여러 설치 경로를 소개하지만, “최신 릴리스 설치 스크립트”가 가장 직관적입니다.

```bash
sudo bash -c 'curl -L https://github.com/dolthub/dolt/releases/latest/download/install.sh | bash'
```

---

## 퀵스타트: 로컬 repo 초기화 → SQL 실행

```bash
mkdir my-dolt-db && cd my-dolt-db
dolt init

# SQL 실행(레포 내부에서)
dolt sql -q "show tables;"
```

CLI 목록/설명은 `README.md`에 요약이 포함되어 있습니다. (`dolt init`, `dolt add`, `dolt commit`, `dolt branch`, `dolt merge`, `dolt push/pull`, `dolt sql`, `dolt sql-server` 등)

---

## 서버 모드: `dolt sql-server`

`README.md`와 `docker/serverREADME.md`는 MySQL 호환 서버 모드를 핵심 사용 케이스로 다룹니다.

```bash
dolt sql-server --host 0.0.0.0 --port 3306
```

---

## Docker로 실행(레포 문서 기준)

`docker/README.md`는 `dolthub/dolt`(CLI 이미지) 사용을, `docker/serverREADME.md`는 `dolthub/dolt-sql-server`(서버 이미지) 사용을 설명합니다.

```bash
# CLI 이미지(실행 = dolt 명령 실행)
docker run dolthub/dolt:latest

# 서버 이미지(기본 실행 = dolt sql-server ...)
docker run -p 3307:3306 dolthub/dolt-sql-server:latest
```

서버 이미지에서 연결 호스트/비밀번호를 초기화하기 위한 환경변수도 문서에 정리되어 있습니다. (예: `DOLT_ROOT_PASSWORD`, `DOLT_ROOT_HOST`) (`docker/serverREADME.md`)

---

## 근거(파일/경로)

- 설치/CLI 요약: `README.md`
- 도커 사용: `docker/README.md`, `docker/serverREADME.md`
- 도커 파일: `docker/Dockerfile`, `docker/serverDockerfile`

---

## 위키 링크

- `[[Dolt Guide - Index]]` → [가이드 목차](/blog-repo/dolt-guide/)
- `[[Dolt Guide - Architecture]]` → [03. 아키텍처(레포 구조)](/blog-repo/dolt-guide-03-architecture-and-repo-map/)

---

*다음 글에서는 Dolt의 Go 코드 구조를 “어디서부터 읽으면 되는지” 관점으로 정리합니다.*

