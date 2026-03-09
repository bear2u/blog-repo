---
layout: post
title: "notebooklm-py 완벽 가이드 (03) - 핵심 개념과 아키텍처"
date: 2026-03-09
permalink: /notebooklm-py-guide-03-architecture/
author: teng-lin
categories: [개발 도구, notebooklm-py]
tags: [notebooklm-py, Python, GitHub Trending]
original_url: "https://github.com/teng-lin/notebooklm-py"
excerpt: "notebooklm-py의 핵심 개념과 구조"
---

## 핵심 개념(문서/README 기반)

notebooklm-py는 NotebookLM을 “코드로” 다루기 위해 크게 아래 영역을 제공합니다.

1. **Notebooks**: 노트북 생성/조회/삭제/이름 변경
2. **Sources**: URL/파일/YouTube/Drive/텍스트 등 소스 추가 및 fulltext/guide 조회
3. **Chat**: 소스 기반 Q&A, 히스토리, 페르소나(모드) 설정
4. **Artifacts(Studio)**: 오디오/비디오/슬라이드/퀴즈/플래시카드/리포트/마인드맵 등 생성·다운로드
5. **Research**: 웹/Drive 리서치 에이전트 + 자동 임포트
6. **Sharing/Notes**: 공유/권한 및 노트 저장

---

## 코드 구조(리포 기준 빠른 지도)

리포지토리 `src/notebooklm/` 아래에 도메인별 모듈이 분리되어 있습니다.

- `client.py`: 클라이언트 엔트리(세션/핸들)
- `_notebooks.py`, `_sources.py`, `_chat.py`, `_artifacts.py`, `_research.py`, `_sharing.py`, `_notes.py`: 기능별 서브 클라이언트
- `auth.py`, `_settings.py`, `paths.py`: 인증/설정/경로 및 상태 저장
- `notebooklm_cli.py`: `notebooklm` CLI 엔트리

---

## 다음에 볼 것

- `docs/python-api.md` (async API 개요)
- `docs/cli-reference.md` (명령어 전체)
- `docs/configuration.md` (스토리지/환경변수/CI)

---

*다음 글에서는 실전 사용 패턴을 정리합니다.*

