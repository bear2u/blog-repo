---
layout: page
title: Summarize 가이드
permalink: /summarize-guide/
icon: fas fa-terminal
---

# Summarize 완벽 가이드

> **Chrome/Firefox 사이드패널 + 로컬 데몬 + CLI를 결합해 URL/파일/미디어를 빠르게 요약하는 실전 도구 해설**

**Summarize (`@steipete/summarize`)**는 링크/문서/오디오·비디오를 요약하는 TypeScript 기반 프로젝트입니다. CLI 단독 사용도 가능하고, 브라우저 확장과 로컬 데몬을 연결해 스트리밍 요약/채팅/슬라이드 추출까지 처리할 수 있습니다.

- 원문 저장소: https://github.com/steipete/summarize
- 공식 문서 사이트: https://summarize.sh

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개/핵심 포지션](/blog-repo/summarize-guide-01-intro/) | Summarize가 해결하는 문제와 제품 구조 |
| 02 | [설치/빠른 시작](/blog-repo/summarize-guide-02-install-and-quickstart/) | CLI, Chrome/Firefox 확장, daemon 설치 흐름 |
| 03 | [입력/실행 플로우](/blog-repo/summarize-guide-03-input-and-run-pipeline/) | URL/파일/stdin 처리와 run 파이프라인 |
| 04 | [모델/프로바이더 전략](/blog-repo/summarize-guide-04-model-and-provider-strategy/) | auto 모델 선택, OpenRouter fallback, CLI 모델 |
| 05 | [설정/환경변수 우선순위](/blog-repo/summarize-guide-05-config-and-env-precedence/) | config.json, .env, CLI 플래그 병합 규칙 |
| 06 | [캐시/비용/출력 제어](/blog-repo/summarize-guide-06-cache-cost-output-controls/) | SQLite 캐시, media 캐시, 길이/토큰/메트릭 |
| 07 | [로컬 데몬 아키텍처](/blog-repo/summarize-guide-07-daemon-architecture/) | daemon install/run, 토큰 인증, SSE 엔드포인트 |
| 08 | [브라우저 확장 동작](/blog-repo/summarize-guide-08-browser-extension-flow/) | Side Panel/Sidebar UX와 auto summarize 흐름 |
| 09 | [미디어/유튜브/슬라이드](/blog-repo/summarize-guide-09-media-youtube-slides/) | transcript-first, yt-dlp/Whisper fallback, OCR 슬라이드 |
| 10 | [테스트/릴리즈/트러블슈팅](/blog-repo/summarize-guide-10-testing-release-troubleshooting/) | 운영 체크리스트와 소스 기준 주의사항 |
