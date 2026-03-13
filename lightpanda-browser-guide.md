---
layout: page
title: Lightpanda Browser 가이드
permalink: /lightpanda-browser-guide/
icon: fas fa-globe
---

# 🕶️ Lightpanda Browser 완벽 가이드

> **Headless browser designed for AI and automation (CDP compatible)**

**lightpanda-io/browser**는 헤드리스 사용을 목표로 하는 오픈소스 브라우저입니다. `fetch`로 URL을 덤프하거나, `serve`로 **CDP(WebSocket) 서버**를 띄워 Puppeteer/Playwright 같은 클라이언트로 제어할 수 있도록 구성되어 있습니다. (`README.md`)

---

## 📚 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 & 위키 맵](/blog-repo/lightpanda-browser-guide-01-intro-and-wiki-map/) | 목표/기능/레포 구조 |
| 02 | [설치 & 퀵스타트](/blog-repo/lightpanda-browser-guide-02-install-and-quickstart/) | nightly 바이너리, Docker, `fetch/serve` |
| 03 | [아키텍처(모듈 지도)](/blog-repo/lightpanda-browser-guide-03-architecture-and-modules/) | `src/browser`, `src/cdp`, `src/network` 등 |
| 04 | [CDP 자동화 예제](/blog-repo/lightpanda-browser-guide-04-cdp-automation/) | Puppeteer 연결 흐름, 옵션/주의 |
| 05 | [운영/CI/텔레메트리/문제해결](/blog-repo/lightpanda-browser-guide-05-ops-ci-telemetry-troubleshooting/) | Actions, 빌드/포맷, 텔레메트리, 체크리스트 |

---

## 빠른 시작 (README 기준)

```bash
# Docker로 CDP 서버 실행 (9222)
docker run -d --name lightpanda -p 9222:9222 lightpanda/browser:nightly

# 로컬 바이너리로 URL 덤프
./lightpanda fetch --obey_robots --log_format pretty --log_level info https://demo-browser.lightpanda.io/campfire-commerce/
```

---

## 관련 링크

- GitHub 저장소: https://github.com/lightpanda-io/browser
- 벤치마크/데모(README 링크): https://github.com/lightpanda-io/demo

