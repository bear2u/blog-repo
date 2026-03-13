---
layout: post
title: "InsForge 완벽 가이드 (05) - 운영/보안/트러블슈팅"
date: 2026-03-13
permalink: /insforge-guide-05-ops-and-troubleshooting/
author: InsForge
categories: [개발 도구, insforge]
tags: [Trending, GitHub, insforge, Security, Troubleshooting, GitHub Trending]
original_url: "https://github.com/InsForge/InsForge"
excerpt: "docker-compose*.yml, .env.example, GITHUB/GOOGLE_OAUTH_SETUP.md를 근거로 운영 체크리스트와 장애 포인트를 정리합니다."
---

## 이 문서의 목적

- InsForge를 로컬/자체 호스팅으로 운용할 때의 “자주 터지는 지점”을 체크리스트로 정리합니다.
- 특히 비밀번호/키/포트/스토리지/마이그레이션을 중심으로 다룹니다.

---

## 빠른 요약

- 기본 포트: `7130/7131/7132` (compose에서 노출)
- 데이터: Postgres 볼륨(`postgres-data`) + 로컬 스토리지(`storage-data`)
- 보안 핵심: `JWT_SECRET`, `ENCRYPTION_KEY`, `ADMIN_PASSWORD`, OAuth 클라이언트 시크릿(.env)
- 함수 런타임: `deno` 서비스(7133) + `functions/server.ts`

근거:
- `docker-compose.yml`, `docker-compose.prod.yml`
- `.env.example`

---

## 1) 기동 실패: 포트 충돌

compose가 고정으로 쓰는 대표 포트:

- 7130~7133
- 5432(Postgres), 5430(PostgREST)

대응:

- 충돌 포트를 먼저 찾고(예: `lsof -i :7131`), compose의 `ports`를 조정

근거:
- `docker-compose.yml`, `docker-compose.prod.yml`

---

## 2) 로그인/관리자 계정 관련

compose 환경 변수에 관리자 계정이 존재합니다.

- `ADMIN_EMAIL`
- `ADMIN_PASSWORD`

운영 체크리스트:

- 외부 노출 전 반드시 변경
- 비밀번호 정책/초기화 절차를 문서화(팀 운영)

근거:
- `docker-compose.yml`, `docker-compose.prod.yml`

---

## 3) 암호화/토큰 키(JWT/Encryption)

필드(발췌):

- `JWT_SECRET`
- `ENCRYPTION_KEY`

근거:
- `docker-compose.yml`, `.env.example`

운영 체크리스트:

- 키 회전(rotate) 전략 수립
- 키 유출 방지: `.env`를 레포에 커밋하지 않기

---

## 4) OAuth 설정(구글/깃허브 등)

레포에는 OAuth 설정 가이드가 별도 파일로 존재합니다.

- `GOOGLE_OAUTH_SETUP.md`
- `GITHUB_OAUTH_SETUP.md`

근거:
- 레포 파일 목록

---

## 5) 스토리지/로그

prod compose 기준:

- 스토리지: `/insforge-storage` (`storage-data` 볼륨)
- 로그: `/insforge-logs` (`shared-logs` 볼륨) + `vector`가 수집

근거:
- `docker-compose.prod.yml`

---

## 6) 마이그레이션/초기화

개발 compose는 `insforge` 서비스 `command`에서:

- `npm install`
- `cd backend && npm run migrate:up`

과 같은 초기화를 수행합니다(정확 커맨드는 compose 파일의 `command` 정의를 기준으로 확인).

근거:
- `docker-compose.yml`

---

## TODO / 확인 필요

- “보안 헤더/CORS/레이트 리밋/백업” 같은 운영 항목은 백엔드 설정(예: `backend/src/api/middlewares/*`, `deploy/*`)을 읽고 구체 체크리스트로 확정하는 것이 좋습니다.

---

## 위키 링크

- `[[InsForge Guide - Index]]` → [가이드 목차](/blog-repo/insforge-guide/)
- `[[InsForge Guide - Docker]]` → [02. 설치 및 실행(Docker)](/blog-repo/insforge-guide-02-docker/)
- `[[InsForge Guide - Architecture]]` → [04. 구성요소/아키텍처](/blog-repo/insforge-guide-04-architecture/)

