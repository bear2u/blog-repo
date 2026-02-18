---
layout: post
title: "Summarize 가이드 (05) - 설정/환경변수 우선순위: config.json과 .env 병합 규칙"
date: 2026-02-18
permalink: /summarize-guide-05-config-and-env-precedence/
author: steipete
categories: ['개발 도구', '설정']
tags: [config.json, .env, Environment Variables, Precedence, summarize]
original_url: "https://github.com/steipete/summarize/blob/main/docs/config.md"
excerpt: "docs/config.md와 src/config.ts를 기준으로 설정 우선순위, env 병합, 주요 키 구조를 해설합니다."
---

## 기본 설정 파일

기본 경로는 `~/.summarize/config.json`입니다.

주요 섹션:

- `model`, `models`(preset)
- `output.language`
- `prompt`
- `cache`, `cache.media`
- `cli` (CLI provider 제어)
- `env`/`apiKeys` (legacy)
- `ui.theme`, `slides`, `logging`

---

## 우선순위 핵심

문서의 우선순위는 다음이 핵심입니다.

- 모델: CLI flag > env > config > built-in default
- 언어: CLI flag > config > default(auto)
- 프롬프트: CLI flag > config > built-in
- env 값: process env > config.env > legacy apiKeys 매핑

즉 운영 환경에서는 "실행 시점 env"가 가장 강합니다.

---

## .env 로드 포인트

`src/cli-main.ts`에서 현재 작업 디렉토리의 `.env`를 읽고 환경에 병합합니다.

이 설계 덕분에 프로젝트별 키/옵션을 분리하기 쉽습니다.

---

## 실무 팁

- 팀 공통 기본값은 `config.json`
- CI/배포 비밀값은 프로세스 env
- 임시 실험값은 CLI 플래그

이 3계층을 분리해 쓰면 설정 충돌을 줄일 수 있습니다.

다음 장에서 캐시와 출력/비용 제어를 봅니다.

