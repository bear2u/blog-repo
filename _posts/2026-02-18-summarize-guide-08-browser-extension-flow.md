---
layout: post
title: "Summarize 가이드 (08) - 브라우저 확장 동작: Side Panel/Sidebar 스트리밍 UX"
date: 2026-02-18
permalink: /summarize-guide-08-browser-extension-flow/
author: steipete
categories: ['개발 도구', '브라우저 확장']
tags: [Chrome Extension, Firefox Sidebar, Side Panel, Streaming, UX]
original_url: "https://github.com/steipete/summarize/blob/main/docs/chrome-extension.md"
excerpt: "Chrome Side Panel + Firefox Sidebar에서 auto summarize가 어떤 이벤트 흐름으로 동작하는지 정리합니다."
---

## 확장 아키텍처 요약

문서 기준 확장은 세 파트로 나뉩니다.

1. Side Panel/Sidebar UI
2. Background service worker
3. Content script(Readability 기반 추출)

패널은 SSE를 직접 수신해 스트리밍 Markdown을 렌더링합니다.

---

## 데이터 흐름

대표 흐름:

1. 사용자가 패널 열기
2. 탭 이동/URL 변경 감지
3. content script가 `{url,title,text}` 추출
4. background가 daemon `/v1/summarize` 호출
5. 패널이 `/events` SSE 구독

패널이 열려 있을 때만 auto summarize를 수행해 비용/노이즈를 줄입니다.

---

## Firefox 포인트

`apps/chrome-extension/README.md` 기준 Firefox는 131+에서 Sidebar 형태로 지원합니다.

- toolbar icon으로 토글
- 단축키 `Ctrl+Shift+U` (macOS는 Cmd)
- Temporary Add-on 로드 방식 안내 제공

Chrome Side Panel과 UX는 조금 다르지만 핵심 파이프라인은 동일합니다.

---

## 운영/디버그

문서에서 강조하는 문제 해결 경로:

- daemon 상태 확인
- extension 로그 확인
- site access 권한 재점검
- 확장 reload 후 재시도

다음 장에서 media/youtube/slides 파이프라인을 집중 분석합니다.

