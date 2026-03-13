---
layout: post
title: "Dolt 완벽 가이드 (05) - 운영/테스트/자동화 (GitHub Actions, 테스트, 보안, 문제해결)"
date: 2026-03-13
permalink: /dolt-guide-05-ops-testing-automation/
author: dolthub
categories: [개발 도구, 데이터베이스]
tags: [Trending, GitHub, dolt, CI, Operations]
original_url: "https://github.com/dolthub/dolt"
excerpt: "Dolt 레포의 GitHub Actions 운영 문서와 테스트 디렉토리를 기반으로, CI/릴리스/벤치마크 자동화와 문제해결 단서를 정리합니다."
---

## GitHub Actions 운영 문서가 따로 있다

`dolt/.github/workflows/README.md`는 이 레포의 워크플로우를 4가지로 분류해 설명합니다.

- PR/푸시 CI(접두사 `ci-`)
- 릴리스/배포(접두사 `cd-`)
- 벤치마크/쿠버네티스 잡(접두사 `k8s-`, `repository_dispatch`)
- 기타 dispatch 기반 워크플로우

이 문서가 존재한다는 것 자체가 “운영 자동화가 핵심 관심사”임을 보여줍니다.

---

## 테스트/성능/통합 흐름 단서

레포 상위에 아래 디렉토리가 존재합니다.

- `integration-tests/` (통합 테스트)
- `go/performance/` (성능 관련 코드/벤치)
- `.github/workflows/*` (CI/벤치/릴리스 등)

---

## 프로토콜/코드 생성(Proto) 자동화 단서

`proto/README.md`는 `.proto` 변경 시 업데이트 루틴을 안내합니다.

- `proto/`에서 `bazel run //:update` 실행
- 생성된 Go 바인딩은 `go/gen/`로 업데이트됨(문서 설명)

---

## 운영 관점 문제해결 체크리스트(레포 기준)

- CI가 실패하면, 우선 `.github/workflows/README.md`로 “어떤 종류의 워크플로우인지”를 분류
- 도커 서버 관련 문제는 `docker/serverREADME.md`의 환경변수/볼륨/설정 경로 섹션을 먼저 확인
- 보안 관련 이슈는 `SECURITY.md` 확인

---

## 근거(파일/경로)

- Actions 개요 문서: `.github/workflows/README.md`
- 워크플로우 파일들: `.github/workflows/*.yaml`
- 도커 운영: `docker/serverREADME.md`
- 프로토 업데이트: `proto/README.md`
- 보안: `SECURITY.md`

---

## 위키 링크

- `[[Dolt Guide - Index]]` → [가이드 목차](/blog-repo/dolt-guide/)
- `[[Dolt Guide - Intro]]` → [01. 소개 & 위키 맵](/blog-repo/dolt-guide-01-intro-and-wiki-map/)

---

*이 챕터는 “문서 점검/자동화 노드” 역할을 겸합니다. 이후 필요하면 특정 워크플로우(ci/cd/k8s)를 쪼개서 확장할 수 있습니다.*

