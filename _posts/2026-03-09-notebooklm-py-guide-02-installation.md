---
layout: post
title: "notebooklm-py 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-03-09
permalink: /notebooklm-py-guide-02-installation/
author: teng-lin
categories: [개발 도구, notebooklm-py]
tags: [notebooklm-py, Python, GitHub Trending]
original_url: "https://github.com/teng-lin/notebooklm-py"
excerpt: "notebooklm-py 설치와 빠른 시작"
---

## 요구사항 체크(README 기준)

- Python 3.10+
- 첫 인증(로그인)을 위해 브라우저 자동화가 필요할 수 있음

---

## 설치

```bash
# 기본 설치
pip install notebooklm-py

# 브라우저 로그인 지원 포함(초기 설정에 필요할 수 있음)
pip install "notebooklm-py[browser]"
playwright install chromium
```

---

## CLI 빠른 시작(README/Docs 기준)

```bash
# 1) 인증(브라우저 열림)
notebooklm login

# 2) 노트북 생성/선택
notebooklm create "My Research"
notebooklm use <notebook_id>

# 3) 소스 추가 → 질의
notebooklm source add "https://example.com"
notebooklm ask "이 소스를 요약해줘"
```

---

## 팁

- 저장 위치를 바꾸고 싶으면 `--storage PATH` 또는 `NOTEBOOKLM_HOME`을 확인하세요.
- 인증/세션 문제는 문서의 Troubleshooting을 먼저 보세요.

---

*다음 글에서는 핵심 개념과 아키텍처를 정리합니다.*

