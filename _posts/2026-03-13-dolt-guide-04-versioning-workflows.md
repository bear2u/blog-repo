---
layout: post
title: "Dolt 완벽 가이드 (04) - 버전관리 워크플로우 (commit/branch/merge/remote + server/docker)"
date: 2026-03-13
permalink: /dolt-guide-04-versioning-workflows/
author: dolthub
categories: [개발 도구, 데이터베이스]
tags: [Trending, GitHub, dolt, Workflow, Docker]
original_url: "https://github.com/dolthub/dolt"
excerpt: "README가 나열하는 Git형 명령을 기준으로 Dolt의 데이터 버전관리 워크플로우를 정리하고, server/docker 문서와 연결합니다."
---

## Git형 명령 세트(README 기준)

`README.md`는 `dolt` CLI가 Git과 유사한 명령을 제공한다고 명시하고, 실제로 `clone/pull/push/branch/checkout/merge` 등이 목록에 포함됩니다.

---

## (예시) “데이터를 커밋한다”는 것

Dolt는 “파일이 아니라 테이블”을 버전관리한다는 설명을 전면에 둡니다. (`README.md`)

```bash
# (예시) 변경을 스테이징/커밋
dolt status
dolt add .
dolt commit -m "update data"

# 로그 확인
dolt log
```

> 주의: 실제 워크플로우는 데이터 삽입/변경 경로(SQL vs import)에 따라 달라질 수 있습니다. 본 시리즈는 레포 문서/구조에서 확인 가능한 범위를 우선합니다.

---

## 서버/도커와 연결하기

버전관리 워크플로우는 로컬 CLI뿐 아니라 **서버 모드**와도 맞물립니다.

- 서버 모드(로컬): `dolt sql-server` (`README.md`)
- 서버 모드(도커): `dolthub/dolt-sql-server` (`docker/serverREADME.md`)
- 도커 환경변수/볼륨/설정 경로: `docker/serverREADME.md` (예: `/etc/dolt/servercfg.d/`, `/var/lib/dolt/`)

```bash
# 도커로 서버 실행(기본)
docker run -p 3307:3306 dolthub/dolt-sql-server:latest

# 로컬 파일을 데이터 디렉토리로 마운트(문서 기준)
docker run -p 3307:3306 -v /path/to/databases:/var/lib/dolt/ dolthub/dolt-sql-server:latest
```

---

## 근거(파일/경로)

- CLI 명령 목록/설명: `README.md`
- 도커 서버 운영/설정: `docker/serverREADME.md`
- 도커 CLI 이미지 설명: `docker/README.md`

---

## 위키 링크

- `[[Dolt Guide - Index]]` → [가이드 목차](/blog-repo/dolt-guide/)
- `[[Dolt Guide - Ops/Automation]]` → [05. 운영/테스트/자동화](/blog-repo/dolt-guide-05-ops-testing-automation/)

---

*다음 글에서는 `.github/workflows/README.md`와 `integration-tests/`를 중심으로 운영/자동화 관점을 정리합니다.*

