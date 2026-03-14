---
layout: post
title: "dimos 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-14
permalink: /dimos-guide-01-intro/
author: dimensionalOS
categories: [로보틱스, dimos]
tags: [Trending, GitHub, dimos, Robotics, Agents, MCP]
original_url: "https://github.com/dimensionalOS/dimos"
excerpt: "Dimensional(dimos)의 목표(로보틱스용 agentive OS), 주요 구성(blueprints/CLI/MCP)과 레포 구조를 README/docs 기반으로 정리합니다."
---

## dimos란?

GitHub Trending(daily, 2026-03-14 기준) 상위에 오른 **dimensionalOS/dimos**를 한국어로 정리합니다.

- **한 줄 요약(README 기반)**: 로보틱스(휴머노이드/4족/드론 등)에서 “에이전트 네이티브” 애플리케이션을 Python으로 만들고 실행하는 SDK/OS를 표방합니다. (`README.md`)
- **설치 포인트**: 설치 스크립트(`scripts/install.sh`) 및 OS별 가이드(`docs/installation/*`)가 있습니다. (`README.md`)
- **개발자 안내**: 레포에 `AGENTS.md`가 존재하며, 에이전트/CLI/MCP 인터페이스를 강조합니다. (`README.md`, `AGENTS.md`)

---

## 이 문서의 목적

- README에 흩어진 “무엇을 할 수 있나(유스케이스)”와 “어디에 뭐가 있나(레포 지도)”를 한 번에 잡습니다.
- 다음 챕터(설치/첫 실행)에서 바로 따라 할 수 있게, “run” 커맨드/블루프린트 개념을 연결합니다.

---

## 레포 구조(상위)

```text
dimos/
  README.md
  pyproject.toml
  docs/                 # 설치/사용법/개발 문서
  scripts/              # 설치 스크립트 등
  dimos/                # Python 패키지(코어/로봇/에이전트/웹)
  assets/               # 데모/시각화 자산
```

패키지 내부(관심사 기준, 디렉토리명 기반):

- 코어/런타임: `dimos/core/*`
- 로봇/플랫폼/CLI: `dimos/robot/*`, `dimos/robot/cli/*`
- 제어/태스크: `dimos/control/*`
- 맵핑/메모리: `dimos/mapping/*`, `dimos/memory/*`
- 에이전트/MCP: `dimos/agents/*`, `dimos/agents/mcp/*`
- 스킬: `dimos/skills/*`
- 웹/시각화: `dimos/web/*`, `dimos/visualization/*`

---

## (개략) “블루프린트 실행” 개념도

```mermaid
flowchart LR
  CLI[dimos CLI\n(dimos/robot/cli/dimos.py)] --> Run[run <blueprint>\n(dimos.core.blueprints...)]
  Run --> Modules[Native Modules\n(docs/usage/native_modules.md)]
  Modules --> Streams[Streams\n(video/audio/tf/pubsub)\n(dimos/stream/*, dimos/protocol/*)]
  Run --> Skills[Skills\n(dimos/skills/*)]
  Run --> Agents[MCP/Agents\n(dimos/agents/mcp/*)]
  Run --> Viz[Visualization\n(docs/usage/visualization.md)\n(dimos/visualization/*)]
```

> 위 도식은 “문서 내비게이션”용 지도입니다. 실제 호출 흐름은 `dimos/robot/cli/dimos.py`의 `run` 커맨드에서 시작해 확인하는 것이 안전합니다.

---

## 근거(파일/경로)

- 개요/설치/블루프린트: `README.md`
- 설치 가이드: `docs/installation/*`, `docs/requirements.md`
- 사용법(blueprints/cli/modules): `docs/usage/*`
- CLI 엔트리: `pyproject.toml` (`[project.scripts]`의 `dimos`)
- CLI 구현: `dimos/robot/cli/dimos.py`
- 패키지 구조: `dimos/*`

---

## 주의사항/함정

- README에 “Git LFS 다운로드” 관련 주의가 있어(리플레이/데모), 첫 실행 시 대용량 다운로드가 발생할 수 있습니다. (`README.md`)
- 하드웨어/시뮬레이션/리플레이 모드가 섞여 있으니, “내가 지금 하려는 실행”이 어떤 모드인지 먼저 고르세요. (`README.md`, `docs/usage/cli.md`, `docs/usage/blueprints.md`)

---

## TODO/확인 필요

- `docs/usage/blueprints.md`의 정의와 코드(`dimos/core/blueprints*`)를 1:1로 매핑해 “블루프린트 레지스트리” 구조를 더 정확히 정리하기
- MCP 서버/어댑터 구성(문서/코드)을 `dimos/agents/mcp/*` 기준으로 정리하기

---

## 위키 링크

- `[[dimos Guide - Index]]` → [가이드 목차](/blog-repo/dimos-guide/)
- `[[dimos Guide - Install & First Run]]` → [02. 설치 및 첫 실행](/blog-repo/dimos-guide-02-install-and-first-run/)

---

*다음 글에서는 설치 문서(`docs/installation/*`)와 `scripts/install.sh`를 기준으로 “첫 실행” 루트를 정리합니다.*

