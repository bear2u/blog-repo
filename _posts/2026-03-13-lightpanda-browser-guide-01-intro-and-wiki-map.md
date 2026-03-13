---
layout: post
title: "Lightpanda Browser 완벽 가이드 (01) - 소개 & 위키 맵"
date: 2026-03-13
permalink: /lightpanda-browser-guide-01-intro-and-wiki-map/
author: lightpanda-io
categories: [개발 도구, 브라우저 자동화]
tags: [Trending, GitHub, browser, Lightpanda, CDP, Zig]
original_url: "https://github.com/lightpanda-io/browser"
excerpt: "CDP로 Playwright/Puppeteer와 연결 가능한 헤드리스 브라우저 Lightpanda의 목표/기능/레포 구조를 위키형으로 정리합니다."
---

## Lightpanda Browser란?

GitHub Trending(daily, 2026-03-13 기준) 상위에 오른 **lightpanda-io/browser**를 레포(README/구조) 기준으로 정리합니다.

- **한 줄 요약(Trending 표시)**: Lightpanda: the headless browser designed for AI and automation
- **언어(Trending 표시)**: Zig
- **오늘 스타(Trending 표시)**: +1,175
- **원본**: https://github.com/lightpanda-io/browser

---

## README가 명시하는 핵심 기능

`README.md` 기준으로, Lightpanda는 헤드리스 사용을 목표로 합니다.

- JavaScript 실행(V8)
- Web API 일부 지원(WIP)
- CDP 호환을 통한 Playwright/Puppeteer/chromedp 연동(문서 링크 포함)

또한 CLI 예시로 `fetch`(URL 덤프)와 `serve`(CDP 서버)를 제공합니다. (`README.md`)

---

## 근거(파일/경로)

- 개요/Quick start/설치/옵션: `README.md`
- Zig 버전/의존성: `build.zig.zon` (`minimum_zig_version = "0.15.2"`)
- 빌드/테스트 타겟: `Makefile`
- Docker 빌드/런: `Dockerfile`
- 핵심 소스: `src/` (예: `src/main.zig`, `src/browser/*`, `src/cdp/*`, `src/network/*`, `src/telemetry/*`)
- 포맷/테스트/CLA 워크플로우: `.github/workflows/*` (예: `zig-fmt.yml`, `zig-test.yml`, `cla.yml`)

---

## (위키 맵) 문서 구조

```text
01. 소개 & 위키 맵
02. 설치 & 퀵스타트 (nightly/Docker, fetch/serve)
03. 아키텍처(모듈 지도) (browser/cdp/network/telemetry)
04. CDP 자동화 예제 (Puppeteer/Playwright 연결 흐름)
05. 운영/CI/텔레메트리/문제해결 (Actions, fmt/test, disable telemetry)
```

---

## 위키 링크

- `[[Lightpanda Browser Guide - Index]]` → [가이드 목차](/blog-repo/lightpanda-browser-guide/)
- `[[Lightpanda Browser Guide - Quickstart]]` → [02. 설치 & 퀵스타트](/blog-repo/lightpanda-browser-guide-02-install-and-quickstart/)

---

*다음 글에서는 nightly 바이너리/Docker로 빠르게 실행해보는 경로를 정리합니다.*

