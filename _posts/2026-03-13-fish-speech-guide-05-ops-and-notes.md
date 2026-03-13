---
layout: post
title: "fish-speech 완벽 가이드 (05) - 운영/주의사항"
date: 2026-03-13
permalink: /fish-speech-guide-05-ops-and-notes/
author: fishaudio
categories: [AI, fish-speech]
tags: [Trending, GitHub, fish-speech, TTS, License, Troubleshooting, GitHub Trending]
original_url: "https://github.com/fishaudio/fish-speech"
excerpt: "LICENSE, docker/Dockerfile, compose.base.yml, API_FLAGS.txt를 근거로 배포/운영 시 체크해야 할 핵심을 정리합니다."
---

## 이 문서의 목적

- fish-speech를 실제 제품/서비스/팀 운영 환경에 올릴 때 “놓치기 쉬운” 포인트를 체크리스트로 정리합니다.
- 특히 체크포인트/라이선스/인증/API 포맷을 중심으로 다룹니다.

---

## 빠른 요약

- Docker 이미지에는 체크포인트가 포함되지 않으며, 반드시 볼륨 마운트가 필요합니다. (`docker/Dockerfile`)
- Compose는 `./checkpoints`와 `./references`를 컨테이너에 마운트합니다. (`compose.base.yml`)
- API 서버는 선택적으로 Bearer 토큰 인증을 걸 수 있습니다. (`tools/api_server.py`)
- 라이선스는 “Fish Audio Research License”이며, 재배포/서비스 제공 시 의무 조항이 존재합니다. (`LICENSE`)

---

## 1) 체크포인트 운영(핵심)

### 체크포인트 미포함

Dockerfile 주석에 명시:

- “docker images do not contain the checkpoints” (`docker/Dockerfile`)

### 볼륨 마운트 경로

Compose 베이스가 마운트하는 경로:

- `./checkpoints:/app/checkpoints`
- `./references:/app/references`

근거:
- `compose.base.yml`

---

## 2) API/네트워크 운영

### 리슨 주소/포트

- 기본 `--listen 127.0.0.1:8080` (`tools/server/api_utils.py`)
- 외부 공개 시 `0.0.0.0:8080`로 바꾸는 패턴이 레포 예시(`API_FLAGS.txt`)에 존재합니다.

### 인증(선택)

`tools/api_server.py`는 `--api-key`가 설정되면 Bearer 토큰을 검증합니다.

운영 권장 체크리스트:

- 외부 노출 시 `--api-key`를 반드시 설정
- 리버스 프록시(예: nginx)에서 요청 크기 제한/타임아웃 설정

---

## 3) 라이선스/법적 고지(반드시 확인)

`LICENSE`에는 제3자 배포/제공 시 필요한 고지/표시 의무가 포함되어 있습니다(예: “Built with Fish Audio” 표기 등). 본문은 요약이 아니라 “읽어야 할 위치”를 명확히 남깁니다.

근거:
- `LICENSE` (Fish Audio Research License)

---

## 4) 디버깅 포인트(코드 기준)

- 헬스체크: `GET /v1/health` (`tools/server/views.py`)
- 요청 포맷 분기: JSON vs msgpack (`tools/server/api_utils.py`)
- 모델 로딩/워밍업: WebUI는 시작 시 dry-run inference를 수행합니다. (`tools/run_webui.py`)

---

## TODO / 확인 필요

- 체크포인트 다운로드/배치 방법은 레포 내부 문서가 아니라 공식 사이트로 안내됩니다. 최신 안내는 공식 문서를 확인하세요. (`docs/README.ko.md`)

---

## 위키 링크

- `[[fish-speech Guide - Index]]` → [가이드 목차](/blog-repo/fish-speech-guide/)
- `[[fish-speech Guide - Docker]]` → [03. Docker로 실행](/blog-repo/fish-speech-guide-03-docker/)
- `[[fish-speech Guide - Inference]]` → [04. 추론(WebUI/서버)](/blog-repo/fish-speech-guide-04-inference/)

