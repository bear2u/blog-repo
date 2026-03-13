---
layout: post
title: "fish-speech 완벽 가이드 (03) - Docker로 실행"
date: 2026-03-13
permalink: /fish-speech-guide-03-docker/
author: fishaudio
categories: [AI, fish-speech]
tags: [Trending, GitHub, fish-speech, TTS, Docker, Deployment, GitHub Trending]
original_url: "https://github.com/fishaudio/fish-speech"
excerpt: "compose.yml/compose.base.yml 및 docker/Dockerfile의 주석 예시를 근거로 WebUI/Server 컨테이너 실행 루트를 정리합니다."
---

## 이 문서의 목적

- fish-speech를 **Docker로 실행**하는 표준 경로를 “레포 파일에 적힌 그대로” 정리합니다.
- WebUI(Gradio)와 API 서버(8080)의 두 실행 타깃을 구분합니다.

---

## 빠른 요약

- `compose.yml`은 `webui`/`server` 서비스를 제공하며, 각각 `ports`로 `7860`/`8080`을 노출합니다.
- 두 서비스는 공통 베이스(`compose.base.yml`의 `app-base`)를 `extends`로 공유합니다.
- 체크포인트/레퍼런스는 호스트 디렉토리를 컨테이너 `/app/checkpoints`, `/app/references`로 마운트합니다. (`compose.base.yml`)

---

## 1) 체크포인트가 먼저다

Dockerfile 상단 주석:

- “The docker images do not contain the checkpoints. You need to mount the checkpoints to the container.” (`docker/Dockerfile`)

즉, 실행 전 다음이 필요합니다.

- 호스트 `./checkpoints` 디렉토리에 체크포인트를 준비
- compose가 이를 `/app/checkpoints`로 마운트

근거:
- `compose.base.yml`
- `docker/Dockerfile`

---

## 2) Docker Compose로 실행(레포 기준)

### WebUI 프로필

`compose.yml`에서 `webui`는 `profiles: ["webui"]`로 정의되어 있으므로, 실행 시 프로필을 명시하는 편이 안전합니다.

```bash
docker compose --profile webui up --build
```

노출 포트:

- 기본 `7860:7860` (`GRADIO_PORT`로 변경 가능)

### API 서버 프로필

```bash
docker compose --profile server up --build
```

노출 포트:

- 기본 `8080:8080` (`API_PORT`로 변경 가능)

근거:
- `compose.yml`

---

## 3) BACKEND(cu* vs cpu)와 GPU 설정

`compose.base.yml`은 Docker build args로 `BACKEND`를 받습니다.

- 기본값: `BACKEND=cuda` (주석: “or cpu”) (`compose.base.yml`)

또한 `deploy.resources.reservations.devices`에 nvidia GPU 예약 블록이 존재합니다. CPU-only 환경에서는 이 블록이 문제를 일으킬 수 있으므로, 주석에 “remove this block if CPU-only”가 적혀 있습니다. (`compose.base.yml`)

---

## 4) 단일 docker run(문서 주석 예시)

`docker/Dockerfile`에는 build/run 예시가 주석으로 제공됩니다.

예: webui 이미지 실행(포트 7860, 체크포인트 마운트)

```bash
docker run --gpus all \
  -v ./checkpoints:/app/checkpoints \
  -e COMPILE=1 \
  -p 7860:7860 \
  fish-speech-webui:cuda
```

---

## 실행 토폴로지(개략)

```mermaid
flowchart LR
  H[Host ./checkpoints] -->|volume mount| C[/app/checkpoints]
  subgraph Compose[Docker Compose]
    W[webui (target: webui)] --> P1[7860]
    S[server (target: server)] --> P2[8080]
    W --> C
    S --> C
  end
```

---

## 주의사항/함정

- `compose.yml`은 서비스에 `profiles`가 걸려 있으므로, “아무 프로필도 지정하지 않고 `docker compose up`”을 하면 기대와 다르게 동작할 수 있습니다(환경/버전에 따라 다름). 운영에서는 `--profile webui`/`--profile server`를 명시하세요. (근거: `compose.yml`)
- 체크포인트가 없으면 WebUI/서버는 로딩 단계에서 실패할 가능성이 큽니다. (근거: `docker/Dockerfile`, 기본 체크포인트 경로는 `tools/run_webui.py`)

---

## TODO / 확인 필요

- 서버 타깃이 어떤 엔트리포인트로 실행되는지는 Dockerfile의 `target server` 스테이지 정의를 읽고 확정하는 것이 좋습니다(이 문서는 compose/주석 예시 중심으로 정리).

---

## 위키 링크

- `[[fish-speech Guide - Index]]` → [가이드 목차](/blog-repo/fish-speech-guide/)
- `[[fish-speech Guide - Local Install]]` → [02. 설치(로컬)](/blog-repo/fish-speech-guide-02-local-install/)
- `[[fish-speech Guide - Inference]]` → [04. 추론(WebUI/서버)](/blog-repo/fish-speech-guide-04-inference/)

---

*다음 글에서는 `tools/run_webui.py`, `tools/api_server.py`, `tools/api_client.py`를 기준으로 WebUI/서버/클라이언트 실행을 정리합니다.*

