---
layout: post
title: "Dolt 완벽 가이드 (03) - 아키텍처(레포 구조)와 읽는 순서"
date: 2026-03-13
permalink: /dolt-guide-03-architecture-and-repo-map/
author: dolthub
categories: [개발 도구, 데이터베이스]
tags: [Trending, GitHub, dolt, Go, Architecture]
original_url: "https://github.com/dolthub/dolt"
excerpt: "dolt 레포의 주요 엔트리(go/cmd/dolt)와 핵심 라이브러리(go/libraries/doltcore), 스토리지(go/store)의 관계를 레포 구조 기반으로 정리합니다."
---

## 레포 최상위 구조(핵심만)

```text
dolt/
  README.md
  go/
    cmd/
      dolt/
        dolt.go
        commands/
        cli/
    libraries/
      doltcore/
      events/
      utils/
    store/
  docker/
    Dockerfile
    serverDockerfile
    serverREADME.md
  integration-tests/
  proto/
```

---

## “사용자 입력”이 들어오는 경로(개략)

`go/cmd/dolt/`는 CLI 엔트리/명령 구현을 담고, `go/libraries/doltcore/`는 데이터베이스 동작의 중심 도메인 로직을 담는 형태로 보입니다. (디렉토리 분해 기준)

```mermaid
flowchart LR
  U[User] --> C[go/cmd/dolt/*]
  C --> D[go/libraries/doltcore/*]
  D --> S[go/store/*]
  D --> X[proto/ (bindings: go/gen)]
  C --> O[docker/* (packaging)]
```

---

## “어디부터 읽으면 되나” 체크포인트

- CLI 구조 파악
  - `go/cmd/dolt/dolt.go`
  - `go/cmd/dolt/commands/` (명령 단위 구현)
- 핵심 도메인 로직(폴더명만으로도 관심사 분리가 드러남)
  - `go/libraries/doltcore/doltdb/` (DB 레이어)
  - `go/libraries/doltcore/sqle/` (SQL 엔진 접점)
  - `go/libraries/doltcore/merge/`, `diff/`, `rebase/`, `cherry_pick/` 등 (버전관리 기능)
- 스토리지/저장 구조
  - `go/store/`

---

## 근거(파일/경로)

- CLI 엔트리/명령: `go/cmd/dolt/`
- 핵심 라이브러리: `go/libraries/doltcore/`
- 스토리지: `go/store/`
- protobuf 업데이트 안내: `proto/README.md` (예: `bazel run //:update`, 생성물 위치 `go/gen/`)

---

## 위키 링크

- `[[Dolt Guide - Index]]` → [가이드 목차](/blog-repo/dolt-guide/)
- `[[Dolt Guide - Workflows]]` → [04. 버전관리 워크플로우](/blog-repo/dolt-guide-04-versioning-workflows/)

---

*다음 글에서는 “Dolt를 Git처럼 쓰는” 워크플로우(브랜치/머지/리모트) 관점을 정리합니다.*

