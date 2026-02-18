---
layout: post
title: "Summarize 가이드 (09) - 미디어/유튜브/슬라이드: transcript-first와 OCR 카드"
date: 2026-02-18
permalink: /summarize-guide-09-media-youtube-slides/
author: steipete
categories: ['개발 도구', '미디어 처리']
tags: [YouTube, Transcript, yt-dlp, Whisper, Slides, OCR]
original_url: "https://github.com/steipete/summarize/blob/main/docs/media.md"
excerpt: "media.md와 youtube.md를 바탕으로 캡션 우선 추출, yt-dlp fallback, 슬라이드 OCR 흐름을 정리합니다."
---

## transcript-first 전략

`docs/media.md`의 핵심은 "먼저 텍스트(캡션/전사)를 확보"입니다.

해결 순서:

1. 내장 캡션/자막
2. 필요 시 yt-dlp + Whisper 계열 전사 fallback

텍스트 기반 요약 품질을 우선하는 설계입니다.

---

## YouTube 모드

`docs/youtube.md` 기준 `--youtube` 모드:

- `auto`(기본)
- `web`
- `no-auto`
- `apify`
- `yt-dlp`

즉 API/HTML/다운로드 전사 경로를 선택적으로 강제할 수 있습니다.

---

## 슬라이드 추출

`--slides`를 켜면 비디오에서 장면 전환 기반으로 스크린샷을 뽑고,
옵션으로 OCR(`--slides-ocr`) 텍스트를 붙일 수 있습니다.

필요 도구:

- `yt-dlp`
- `ffmpeg`
- OCR 사용 시 `tesseract`

---

## 확장 UX 연결

확장에서는 media 감지 시 Page/Video 선택 드롭다운을 제공하고,
Video/Audio를 선택하면 transcript-first URL 모드로 강제됩니다.

이 방식은 일반 문서 요약과 미디어 요약 경로를 명확히 분리해줍니다.

다음 장에서 테스트/릴리즈/장애 대응 포인트로 마무리합니다.

