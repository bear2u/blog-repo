---
layout: post
title: "PicoClaw 가이드 (04) - Docker Compose: agent(단발)와 gateway(상시 봇)로 실행하기"
date: 2026-02-15
permalink: /picoclaw-guide-04-docker-compose/
author: Sipeed
categories: [AI 에이전트, 개발 도구]
tags: ["PicoClaw", "Docker", "Compose", "Gateway", "Agent Mode"]
original_url: "https://github.com/sipeed/picoclaw#-docker-compose"
excerpt: "PicoClaw는 docker compose로 agent(단발/대화형)와 gateway(장기 실행 봇)를 실행할 수 있습니다. config.json 마운트와 workspace 볼륨이 핵심입니다."
---

## compose가 제공하는 2개 서비스

레포의 `docker-compose.yml`은 크게 두 서비스를 제공합니다.

1. `picoclaw-agent`
   - `docker compose run --rm picoclaw-agent ...`
   - 단발 질문 또는 대화형 CLI 용도
2. `picoclaw-gateway`
   - `docker compose --profile gateway up -d`
   - Telegram/Discord 같은 채널 봇을 상시 실행하는 용도

---

## config와 workspace가 핵심

compose 파일에서 중요한 부분은 두 가지입니다.

- `./config/config.json`을 컨테이너의 `/root/.picoclaw/config.json`으로 마운트(읽기 전용)
- `picoclaw-workspace` 볼륨을 `/root/.picoclaw/workspace`로 마운트(영속 데이터)

즉, 컨테이너를 갈아엎어도:

- 설정(config)은 로컬 파일로 유지되고
- 세션/메모리/스케줄링 DB는 볼륨에 남습니다.

---

## 실행 예시(README 흐름)

```bash
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw

cp config/config.example.json config/config.json
vim config/config.json   # API 키, 채널 토큰 등 설정

docker compose --profile gateway up -d
docker compose logs -f picoclaw-gateway
```

agent 모드(단발):

```bash
docker compose run --rm picoclaw-agent -m "What is 2+2?"
```

다음 장에서는 config.json을 “덩어리”로 이해할 수 있게, 주요 블록(agents/channels/providers/tools/heartbeat/gateway)을 해부합니다.

