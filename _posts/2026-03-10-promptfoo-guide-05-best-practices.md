---
layout: post
title: "promptfoo 완벽 가이드 (05) - 운영/확장/베스트 프랙티스"
date: 2026-03-10
permalink: /promptfoo-guide-05-best-practices/
author: "Ian Webster"
categories: [개발 도구, promptfoo]
tags: [Trending, GitHub, promptfoo]
original_url: "https://github.com/promptfoo/promptfoo"
excerpt: "팀/CI에 붙일 때 체크할 포인트를 정리합니다."
---

## 체크리스트

- API 키/시크릿은 `.env` 파일 커밋 금지, CI Secret으로 관리
- eval 실행이 “항상 같은 입력”을 보도록(데이터/프롬프트/모델 버전) 고정
- 실패 케이스를 재현 가능한 형태로 남기기(설정 파일, 커맨드 로그, 리포트 링크)

---

## 자주 겪는 함정

- 로컬에선 되는데 CI에서 안 되는 경우: 런타임(Node 버전), 환경 변수 누락을 먼저 확인
- eval 속도가 느릴 때: 테스트 범위를 작은 스모크→확장(전체) 순으로 분리

---

## TODO / 확인 필요

- 조직 표준 템플릿(설정 파일 기본형)을 만들고, 신규 프로젝트는 템플릿에서 시작하도록 정리
- “레드팀 결과를 어떤 포맷으로 리뷰/승인할지” 팀 규칙 확정

