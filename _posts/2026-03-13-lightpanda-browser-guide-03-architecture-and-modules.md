---
layout: post
title: "Lightpanda Browser 완벽 가이드 (03) - 아키텍처(모듈 지도): browser/cdp/network/telemetry"
date: 2026-03-13
permalink: /lightpanda-browser-guide-03-architecture-and-modules/
author: lightpanda-io
categories: [개발 도구, 브라우저 자동화]
tags: [Trending, GitHub, browser, Lightpanda, Architecture, Zig]
original_url: "https://github.com/lightpanda-io/browser"
excerpt: "Lightpanda의 핵심 관심사(페이지/DOM/스크립트, CDP 서버, 네트워크, robots/telemetry)를 src/ 디렉토리 구조 기반으로 정리합니다."
---

## 레포 구조(상위)

```text
browser/
  README.md
  build.zig
  build.zig.zon
  Makefile
  Dockerfile
  src/
    main.zig
    browser/
    cdp/
    network/
    telemetry/
```

---

## 모듈 지도를 “디렉토리”로 그리기

아래는 `src/`의 디렉토리/파일명만으로 확인 가능한 관심사 분리입니다.

- 페이지/브라우저 동작: `src/browser/` (예: `Browser.zig`, `Page.zig`, `ScriptManager.zig`, `EventManager.zig`, `dump.zig`)
- CDP: `src/cdp/` (예: `cdp.zig`, `Node.zig`, `AXNode.zig`)
- 네트워크/로봇: `src/network/` (예: `http.zig`, `websocket.zig`, `Robots.zig`)
- 텔레메트리: `src/telemetry/`

---

## (개략) 실행 흐름

```mermaid
flowchart LR
  CLI[src/main.zig] --> MODE{mode}
  MODE -->|fetch| FETCH[lp.fetch\n(src/lightpanda.zig)]
  MODE -->|serve| SRV[lp.Server\n(src/Server.zig)]
  SRV --> CDP[src/cdp/*]
  FETCH --> B[src/browser/*]
  SRV --> B[src/browser/*]
  B --> NET[src/network/*]
  CLI --> TEL[src/telemetry/*]
```

> 이 도식은 “모듈 연결”을 빠르게 잡기 위한 지도입니다. 실제 호출 관계는 `src/main.zig` / `src/lightpanda.zig` / `src/Server.zig`에서 확인하는 것이 안전합니다.

---

## 근거(파일/경로)

- 모드 분기/서버 생성/에러 처리: `src/main.zig`
- fetch 옵션/핸들러 단서: `src/lightpanda.zig` (예: `FetchOpts`, `fetch`)
- 브라우저 구성요소: `src/browser/*`
- CDP 프로토콜/노드 모델: `src/cdp/*`
- robots/http/websocket: `src/network/*`
- 텔레메트리: `src/telemetry/*`

---

## 위키 링크

- `[[Lightpanda Browser Guide - Index]]` → [가이드 목차](/blog-repo/lightpanda-browser-guide/)
- `[[Lightpanda Browser Guide - CDP Automation]]` → [04. CDP 자동화 예제](/blog-repo/lightpanda-browser-guide-04-cdp-automation/)

---

*다음 글에서는 `serve`로 띄운 CDP 서버에 Puppeteer를 연결하는 예제를 README 기준으로 정리합니다.*

