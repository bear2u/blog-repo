---
layout: post
title: "dimos 완벽 가이드 (03) - 아키텍처: core/robot/control/mapping/agents/web"
date: 2026-03-14
permalink: /dimos-guide-03-architecture/
author: dimensionalOS
categories: [로보틱스, dimos]
tags: [Trending, GitHub, dimos, Architecture, Robotics, Modules]
original_url: "https://github.com/dimensionalOS/dimos"
excerpt: "dimos 패키지의 관심사 분리(코어 런타임, 로봇 플랫폼, 제어/맵핑, 스트림/프로토콜, 에이전트/MCP, 웹/시각화)를 디렉토리 구조 기반으로 정리합니다."
---

## 이 문서의 목적

- “어디에서 무엇을 봐야 하는지” 모듈 지도를 확정합니다.
- CLI(`dimos run ...`)가 어떤 서브시스템을 조립하는지, 코드 탐색 루트를 제공합니다.

---

## 빠른 요약 (디렉토리 구조 기반)

```text
dimos/
  core/            # 런타임/설정/레지스트리/리소스 모니터링
  robot/           # 플랫폼/블루프린트/CLI
  control/         # 태스크/컨트롤 루프/예제
  mapping/         # pointcloud/occupancy/지도 유틸
  memory/          # timeseries 등 메모리 구성요소
  stream/          # audio/video providers
  protocol/        # pubsub/rpc/tf/encode
  agents/          # 에이전트/skills/mcp
  skills/          # 로봇 스킬(플랫폼별/도메인별)
  visualization/   # rerun 등
  web/             # 웹 UI/확장/시각화 도구
```

문서 축:

- 사용법: `docs/usage/*` (modules/native_modules/cli/blueprints/visualization 등)
- 에이전트: `docs/agents/index.md`

---

## (개략) 런타임 컴포넌트 지도

```mermaid
flowchart TB
  CLI[dimos CLI\n(dimos/robot/cli/dimos.py)] --> BP[Blueprints\n(dimos/core/blueprints...)]
  BP --> Robot[Robot Platform\n(dimos/robot/*)]
  BP --> Control[Control/Tasks\n(dimos/control/*)]
  Robot --> Stream[Streams\n(dimos/stream/*)]
  Stream --> Proto[Protocol\n(dimos/protocol/*)]
  Control --> Mapping[Mapping\n(dimos/mapping/*)]
  Control --> Memory[Memory\n(dimos/memory/*)]
  BP --> Skills[Skills\n(dimos/skills/*)]
  BP --> Agents[Agents/MCP\n(dimos/agents/*)]
  Agents --> MCP[dimos/agents/mcp/*]
  BP --> Viz[Visualization\n(dimos/visualization/*)]
  BP --> Web[Web\n(dimos/web/*)]
```

> 위 도식은 “패키지 탐색”을 위한 지도입니다. 실제 조립 순서는 `dimos/robot/cli/dimos.py`의 `run` 커맨드에서 시작해, `dimos.core.blueprints` 계층을 따라가며 확인하는 것이 안전합니다.

---

## 핵심 파일(첫 탐색 추천)

- CLI 진입점: `dimos/robot/cli/dimos.py`
- 글로벌 설정/오버라이드: `dimos/core/global_config.py` (CLI가 `GlobalConfig` 필드를 동적으로 옵션으로 노출)
- MCP 어댑터: `dimos/agents/mcp/mcp_adapter.py`
- 문서(모듈/블루프린트/CLI): `docs/usage/modules.md`, `docs/usage/blueprints.md`, `docs/usage/cli.md`

---

## 근거(파일/경로)

- 패키지 구조: `dimos/*`
- CLI/런타임 조립 단서: `dimos/robot/cli/dimos.py`
- 문서: `docs/usage/*`, `docs/agents/index.md`

---

## 주의사항/함정

- dimos는 하드웨어/시뮬레이션/리플레이 모드가 공존하므로, “내가 실행한 blueprint가 어떤 IO/스트림을 쓰는지” 먼저 확인해야 디버깅이 쉬워집니다. (`docs/usage/blueprints.md`, `docs/usage/modules.md`)
- 패키지가 크고 도메인이 넓어서, README만으로는 “정답 경로”가 잘 안 보입니다. CLI(`dimos/robot/cli/dimos.py`)에서 역추적하는 방식이 가장 안정적입니다.

---

## TODO/확인 필요

- `dimos/core/blueprints` 계층(파일/함수)을 열어 “블루프린트 조립 규칙”을 코드 근거로 정리하기
- `docs/usage/transforms.md`와 `dimos/protocol/tf/*`의 관계 확인

---

## 위키 링크

- `[[dimos Guide - Index]]` → [가이드 목차](/blog-repo/dimos-guide/)
- `[[dimos Guide - CLI & MCP]]` → [04. CLI & MCP & Blueprints](/blog-repo/dimos-guide-04-cli-mcp-blueprints/)

---

*다음 글에서는 `dimos/robot/cli/dimos.py`를 기준으로 CLI 옵션/서브커맨드와 MCP 어댑터 흐름을 정리합니다.*

