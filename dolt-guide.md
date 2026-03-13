---
layout: page
title: Dolt 가이드
permalink: /dolt-guide/
icon: fas fa-database
---

# 🗄️ Dolt 완벽 가이드

> **Dolt — Git for Data (SQL 데이터베이스를 Git처럼 버전관리)**

**dolthub/dolt**는 MySQL 호환 SQL 인터페이스를 제공하면서도, **clone/branch/merge/push/pull** 같은 Git 워크플로우로 **테이블과 데이터를 버전관리**하는 데이터베이스입니다.

이 시리즈는 “사용(설치/CLI/서버)”과 “구현(Go 코드 구조/모듈)”을 함께 묶어, 레포 기반으로 재현 가능한 단서(경로/워크플로우) 중심으로 정리합니다.

---

## 📚 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 & 위키 맵](/blog-repo/dolt-guide-01-intro-and-wiki-map/) | Dolt가 푸는 문제, CLI/SQL 관점 정리 |
| 02 | [설치 & 퀵스타트](/blog-repo/dolt-guide-02-install-and-quickstart/) | 설치, `dolt init`, `dolt sql`, `dolt sql-server` |
| 03 | [아키텍처(레포 구조)](/blog-repo/dolt-guide-03-architecture-and-repo-map/) | `go/cmd/dolt` ↔ `go/libraries/doltcore` ↔ `go/store` |
| 04 | [버전관리 워크플로우](/blog-repo/dolt-guide-04-versioning-workflows/) | 커밋/브랜치/머지/리모트, 서버/도커 힌트 |
| 05 | [운영/테스트/자동화](/blog-repo/dolt-guide-05-ops-testing-automation/) | GitHub Actions, 통합테스트, 보안, 문제해결 |

---

## 빠른 시작

```bash
# 설치 (공식 README 안내: 최신 릴리스 install.sh)
sudo bash -c 'curl -L https://github.com/dolthub/dolt/releases/latest/download/install.sh | bash'

# 새 Dolt DB 초기화
mkdir mydb && cd mydb
dolt init

# SQL 실행
dolt sql -q "show tables;"

# MySQL 호환 서버 모드
dolt sql-server --host 0.0.0.0 --port 3306
```

---

## 관련 링크

- GitHub 저장소: https://github.com/dolthub/dolt
- 공식 문서(외부): https://docs.dolthub.com/
- Docker 문서(레포): `docker/README.md`, `docker/serverREADME.md`

