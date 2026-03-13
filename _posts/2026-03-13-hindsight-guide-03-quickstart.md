---
layout: post
title: "Hindsight 완벽 가이드 (03) - 빠른 시작(로컬)"
date: 2026-03-13
permalink: /hindsight-guide-03-quickstart/
author: vectorize-io
categories: [AI 에이전트, hindsight]
tags: [Trending, GitHub, hindsight, Docker, SDK, Quickstart, GitHub Trending]
original_url: "https://github.com/vectorize-io/hindsight"
excerpt: "README.md의 Docker run/Docker compose(Client install) 예시를 근거로 로컬에서 API(8888)/UI(9999)까지 띄우는 최소 루트를 정리합니다."
---

## 이 문서의 목적

- Hindsight를 “지금 당장 써보기” 위해, README에 적힌 커맨드만으로 로컬 API/UI를 띄우는 방법을 정리합니다.
- (중요) LLM 공급자 키를 어디에 넣는지, 어떤 포트를 확인해야 하는지 명확히 합니다.

---

## 빠른 요약(README 기준)

- 가장 쉬운 실행: Docker run (내장 저장소/볼륨을 사용)
- 외부 Postgres 사용: `docker/docker-compose`에서 `docker compose up`
- 클라이언트: `pip install hindsight-client -U` 또는 `npm install @vectorize-io/hindsight-client`

---

## 1) Docker run(권장: README “recommended”)

README 예시:

```bash
export OPENAI_API_KEY=sk-xxx

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

접속:

- API: `http://localhost:8888`
- UI: `http://localhost:9999`

근거:
- `README.md` Quick Start

---

## 2) LLM Provider 변경

README는 다음 환경 변수로 provider를 바꿀 수 있다고 합니다.

- `HINDSIGHT_API_LLM_PROVIDER`
- 값: `openai`, `anthropic`, `gemini`, `groq`, `ollama`, `lmstudio`

근거:
- `README.md` Quick Start

---

## 3) Docker(외부 PostgreSQL)

README 예시:

```bash
export OPENAI_API_KEY=sk-xxx
export HINDSIGHT_DB_PASSWORD=choose-a-password
cd docker/docker-compose
docker compose up
```

근거:
- `README.md` Quick Start

---

## 4) 클라이언트 설치 + 최소 호출

### 설치(README)

```bash
pip install hindsight-client -U
# or
npm install @vectorize-io/hindsight-client
```

### Python 최소 예시(README 발췌 흐름)

```python
from hindsight_client import Hindsight
client = Hindsight(base_url="http://localhost:8888")
client.retain(bank_id="my-bank", content="Alice works at Google as a software engineer")
client.recall(bank_id="my-bank", query="What does Alice do?")
```

근거:
- `README.md` Client 섹션

---

## 주의사항/함정

- `HINDSIGHT_API_LLM_API_KEY`는 Docker run에서 필수로 전달됩니다(예시 기준). 키가 없으면 retain/recall 과정에서 LLM 호출이 실패할 수 있습니다. (`README.md`)
- 포트 8888/9999가 이미 점유되어 있으면 컨테이너 기동이 실패합니다.

---

## TODO / 확인 필요

- UI에서 제공하는 설정/관측 기능(예: bank 관리, 필터, 로그)은 `hindsight-docs/` 및 UI 코드 위치를 확인해 챕터로 확장하면 좋습니다.

---

## 위키 링크

- `[[Hindsight Guide - Index]]` → [가이드 목차](/blog-repo/hindsight-guide/)
- `[[Hindsight Guide - Components]]` → [02. 구성요소 맵](/blog-repo/hindsight-guide-02-components/)
- `[[Hindsight Guide - Memory Design]]` → [04. 메모리 설계/데이터 흐름](/blog-repo/hindsight-guide-04-memory-design/)

---

*다음 글에서는 README의 “World/Experiences/Mental Models” 구조와 retain/recall/reflect 동작을 데이터 흐름으로 정리합니다.*

