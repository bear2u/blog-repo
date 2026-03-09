---
layout: post
title: "notebooklm-py 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-09
permalink: /notebooklm-py-guide-01-intro/
author: teng-lin
categories: [개발 도구, notebooklm-py]
tags: [notebooklm-py, Python, GitHub Trending]
original_url: "https://github.com/teng-lin/notebooklm-py"
excerpt: "notebooklm-py 프로젝트 소개와 핵심 포인트"
---

## notebooklm-py란?

GitHub Trending 기준으로 주목받는 **teng-lin/notebooklm-py**를 한국어로 정리합니다.

- **한 줄 요약**: Unofficial Python API for Google NotebookLM
- **언어**: Python
- **오늘 스타**: +457
- **원본**: https://github.com/teng-lin/notebooklm-py

---

## 왜 주목받나(README 기반)

- NotebookLM 기능을 **Python API / CLI**로 자동화할 수 있습니다.
- 웹 UI에 없는 기능(예: 배치 다운로드, 퀴즈/플래시카드 JSON 내보내기 등)을 다룹니다.

---

## 주의사항(중요)

README에서 명시하듯, **비공식(undocumented) Google API**를 사용합니다.

- Google 내부 API 변경으로 갑자기 동작이 깨질 수 있습니다.
- 과도한 호출은 레이트 리밋/차단 리스크가 있습니다.

---

## 이 가이드에서 다룰 것

- 설치/인증(로그인) 빠른 시작
- 구성요소(Notebook/Sources/Chat/Artifacts) 관점의 구조
- CLI/Python 실전 사용 패턴
- 운영/보안/트러블슈팅 체크리스트

---

*다음 글에서는 설치 및 빠른 시작을 정리합니다.*

