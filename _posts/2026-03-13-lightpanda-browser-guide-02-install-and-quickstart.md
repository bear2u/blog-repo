---
layout: post
title: "Lightpanda Browser 완벽 가이드 (02) - 설치 & 퀵스타트 (nightly/Docker, fetch/serve)"
date: 2026-03-13
permalink: /lightpanda-browser-guide-02-install-and-quickstart/
author: lightpanda-io
categories: [개발 도구, 브라우저 자동화]
tags: [Trending, GitHub, browser, Lightpanda, Docker, CDP]
original_url: "https://github.com/lightpanda-io/browser"
excerpt: "README 기준으로 nightly 바이너리 설치, Docker 실행, fetch로 덤프, serve로 CDP 서버 실행까지 연결합니다."
---

## 설치(README 기준): nightly builds

`README.md`는 nightly build 바이너리를 다운로드하는 설치 경로를 제공합니다.

```bash
# Linux x86_64
curl -L -o lightpanda https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-x86_64-linux
chmod a+x ./lightpanda

# MacOS aarch64
curl -L -o lightpanda https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-aarch64-macos
chmod a+x ./lightpanda
```

---

## 설치(README 기준): Docker

`README.md`는 공식 Docker 이미지(`lightpanda/browser:nightly`)로 CDP 서버를 띄우는 예시를 제공합니다.

```bash
docker run -d --name lightpanda -p 9222:9222 lightpanda/browser:nightly
```

---

## URL 덤프: `fetch`

`README.md`의 예시:

```bash
./lightpanda fetch --obey_robots --log_format pretty --log_level info https://demo-browser.lightpanda.io/campfire-commerce/
```

---

## CDP 서버: `serve`

`README.md`의 예시:

```bash
./lightpanda serve --obey_robots --log_format pretty --log_level info --host 127.0.0.1 --port 9222
```

---

## (소스 단서) 엔트리/모드

`src/main.zig`는 실행 모드 분기를 포함합니다(예: `serve`, `fetch`). 따라서 CLI 옵션/동작을 더 깊게 보고 싶으면 `src/main.zig`부터 추적하는 것이 안전합니다.

---

## 위키 링크

- `[[Lightpanda Browser Guide - Index]]` → [가이드 목차](/blog-repo/lightpanda-browser-guide/)
- `[[Lightpanda Browser Guide - Architecture]]` → [03. 아키텍처(모듈 지도)](/blog-repo/lightpanda-browser-guide-03-architecture-and-modules/)

---

*다음 글에서는 `src/` 기준으로 모듈 지도를 그려 “어디를 읽어야 하는지”를 정리합니다.*

