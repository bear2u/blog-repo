---
layout: post
title: "Lightpanda Browser 완벽 가이드 (04) - CDP 자동화 예제 (Puppeteer 연결 흐름)"
date: 2026-03-13
permalink: /lightpanda-browser-guide-04-cdp-automation/
author: lightpanda-io
categories: [개발 도구, 브라우저 자동화]
tags: [Trending, GitHub, browser, Lightpanda, CDP, Puppeteer]
original_url: "https://github.com/lightpanda-io/browser"
excerpt: "README에 포함된 Puppeteer 예제를 기준으로, Lightpanda CDP 서버(browserWSEndpoint)에 연결해 페이지를 제어하는 기본 흐름을 정리합니다."
---

## 핵심 아이디어: Lightpanda는 CDP 서버를 제공한다

`README.md`는 `serve`로 CDP(WebSocket) 서버를 띄우고, Puppeteer에서 `browserWSEndpoint`로 연결하는 예시를 제공합니다.

---

## 1) CDP 서버 시작(README 기준)

```bash
./lightpanda serve --obey_robots --log_format pretty --log_level info --host 127.0.0.1 --port 9222
```

---

## 2) Puppeteer에서 연결(README 예제 요약)

`README.md`는 `puppeteer-core`의 `connect()`에 `browserWSEndpoint`를 넘기는 형태를 보여줍니다.

```js
import puppeteer from 'puppeteer-core';

const browser = await puppeteer.connect({
  browserWSEndpoint: "ws://127.0.0.1:9222",
});
```

이후 `newPage()`, `goto()`, `evaluate()` 등은 기존 Puppeteer 스크립트 패턴 그대로 유지된다고 설명합니다. (`README.md`)

---

## 3) 호환성 주의(README의 Playwright disclaimer)

`README.md`에는 Playwright 호환성에 대한 주의사항이 포함되어 있습니다. 요지는 “Playwright가 기능 감지를 통해 다른 실행 경로를 선택할 수 있고, Lightpanda가 새 Web API를 추가하면 경로가 바뀌면서 아직 구현되지 않은 기능을 요구할 수 있다”는 점입니다.

문제 발생 시 이슈를 열되, “마지막으로 동작하던 버전”을 포함하라고 안내합니다. (`README.md`)

---

## 근거(파일/경로)

- CDP 서버 실행/예제 코드: `README.md`
- 서버 모드 엔트리 단서: `src/main.zig` (mode 분기)
- CDP 관련 구현: `src/cdp/*`, `src/network/websocket.zig`

---

## 위키 링크

- `[[Lightpanda Browser Guide - Index]]` → [가이드 목차](/blog-repo/lightpanda-browser-guide/)
- `[[Lightpanda Browser Guide - Ops/CI]]` → [05. 운영/CI/텔레메트리/문제해결](/blog-repo/lightpanda-browser-guide-05-ops-ci-telemetry-troubleshooting/)

---

*다음 글에서는 CI(zig fmt/test), 텔레메트리, robots 옵션 등 운영 관점 체크리스트를 정리합니다.*

