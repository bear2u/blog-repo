---
layout: post
title: "notebooklm-py 완벽 가이드 (04) - 실전 사용 패턴"
date: 2026-03-09
permalink: /notebooklm-py-guide-04-usage/
author: teng-lin
categories: [개발 도구, notebooklm-py]
tags: [notebooklm-py, Python, GitHub Trending]
original_url: "https://github.com/teng-lin/notebooklm-py"
excerpt: "notebooklm-py CLI/Python 실전 사용 패턴"
---

## CLI 패턴: 소스 → 질의 → 생성 → 다운로드

```bash
# 로그인 후 노트북 선택
notebooklm login
notebooklm use <notebook_id>

# 소스 추가(여러 개 가능)
notebooklm source add "https://en.wikipedia.org/wiki/Artificial_intelligence"
notebooklm source add "./paper.pdf"

# 질문
notebooklm ask "핵심 테마를 요약해줘"

# 생성(예: 오디오) 후 완료까지 대기
notebooklm generate audio "더 재미있게" --wait

# 결과물 다운로드
notebooklm download audio ./podcast.mp3
```

---

## Python 패턴: 비동기 워크플로우

README의 예시처럼 `NotebookLMClient`를 사용해 노트북 생성 → 소스 추가 → Q&A → 아티팩트 생성/대기/다운로드를 코드로 자동화할 수 있습니다.

- 배치 파이프라인(리서치 자동화, 대량 임포트/다운로드)
- CI에서 `NOTEBOOKLM_AUTH_JSON`로 인증 정보 주입(문서 참고)

---

## 웹 UI에 없는 “API/CLI 특화” 포인트(README 기준)

- `download <type> --all` 같은 **배치 다운로드**
- 퀴즈/플래시카드 **JSON/Markdown/HTML 내보내기**
- 마인드맵 **JSON 추출**
- 슬라이드 덱 **PPTX 다운로드**, 슬라이드 개별 수정

---

*다음 글에서는 운영/확장/베스트 프랙티스를 정리합니다.*

