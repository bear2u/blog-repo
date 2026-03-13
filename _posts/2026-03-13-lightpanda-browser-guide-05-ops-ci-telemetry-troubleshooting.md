---
layout: post
title: "Lightpanda Browser 완벽 가이드 (05) - 운영/CI/텔레메트리/문제해결 체크리스트"
date: 2026-03-13
permalink: /lightpanda-browser-guide-05-ops-ci-telemetry-troubleshooting/
author: lightpanda-io
categories: [개발 도구, 브라우저 자동화]
tags: [Trending, GitHub, browser, Lightpanda, CI, Telemetry]
original_url: "https://github.com/lightpanda-io/browser"
excerpt: "Lightpanda 레포의 GitHub Actions(zig-fmt/zig-test/CLA)와 빌드 도구(Makefile/Dockerfile), 텔레메트리 옵션을 기준으로 운영/문제해결 루틴을 정리합니다."
---

## CI 자동화(레포 기준)

`.github/workflows/*`에서 확인되는 자동화 축:

- CLA 서명 프로세스: `.github/workflows/cla.yml` (문서: `CLA.md`, `CONTRIBUTING.md`)
- 포맷 검사: `.github/workflows/zig-fmt.yml` (문서 내 주석: `minimum_zig_version`은 `build.zig.zon`에서 가져온다고 표시)
- 테스트/벤치 아티팩트: `.github/workflows/zig-test.yml`

---

## 빌드/테스트 커맨드 단서

`Makefile`은 다음 타겟을 제공합니다(발췌):

- `build-v8-snapshot`, `build`, `build-dev`
- `run`, `run-debug`
- `test`

Zig 최소 버전은 `build.zig.zon`에 `minimum_zig_version = "0.15.2"`로 명시되어 있습니다.

---

## 텔레메트리(README 기준)

`README.md`는 기본적으로 사용 텔레메트리를 수집/전송할 수 있으며, 아래 환경변수로 비활성화할 수 있다고 안내합니다.

```bash
export LIGHTPANDA_DISABLE_TELEMETRY=true
```

---

## 문제해결 체크리스트(최소)

- 빌드가 안 되면: Zig 버전(`build.zig.zon`)과 의존성(README “Build from sources”)부터 확인
- 실행이 불안정하면: README의 “Status(Beta/WIP)” 안내 및 이슈 트래커에 재현 정보 제공
- 호환성 문제가 있으면: README의 Playwright 호환성 주의사항을 기준으로 “마지막 동작 버전” 포함

---

## 근거(파일/경로)

- 텔레메트리 안내: `README.md`
- Zig 버전/의존성: `build.zig.zon`
- 빌드/테스트 타겟: `Makefile`
- CI 워크플로우: `.github/workflows/cla.yml`, `.github/workflows/zig-fmt.yml`, `.github/workflows/zig-test.yml`

---

## 위키 링크

- `[[Lightpanda Browser Guide - Index]]` → [가이드 목차](/blog-repo/lightpanda-browser-guide/)
- `[[Lightpanda Browser Guide - Intro]]` → [01. 소개 & 위키 맵](/blog-repo/lightpanda-browser-guide-01-intro-and-wiki-map/)

---

*이 챕터는 “문서 점검/자동화 노드” 역할을 겸합니다. 필요 시 zig-test 워크플로우(벤치 업로드 등)를 별도 챕터로 확장할 수 있습니다.*

