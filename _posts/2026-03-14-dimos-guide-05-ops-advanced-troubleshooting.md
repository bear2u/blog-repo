---
layout: post
title: "dimos 완벽 가이드 (05) - 운영/고급/트러블슈팅: 테스트/도커/LFS/성능"
date: 2026-03-14
permalink: /dimos-guide-05-ops-advanced-troubleshooting/
author: dimensionalOS
categories: [로보틱스, dimos]
tags: [Trending, GitHub, dimos, Operations, Testing, Docker]
original_url: "https://github.com/dimensionalOS/dimos"
excerpt: "개발/운영 문서(docs/development/*)와 CLI 런레지스트리/로그 구조를 기준으로, 실무 체크리스트와 트러블슈팅 루트를 정리합니다."
---

## 이 문서의 목적

- “개발자/운영자 관점”에서 문서/코드가 어디에 있는지 바로 찾게 합니다.
- 테스트/도커/프로파일링/LFS 등, 규모가 큰 레포에서 자주 막히는 지점을 체크리스트화합니다.

---

## 빠른 요약 (Docs 기반)

- 테스트: `docs/development/testing.md`
- 도커: `docs/development/docker.md`
- 프로파일링: `docs/development/profiling_dimos.md`
- 대용량 파일(LFS): `docs/development/large_file_management.md`

---

## 1) 운영 체크리스트(최소)

### 실행/로그/레지스트리

- `dimos` CLI는 run 레지스트리/로그 디렉토리를 사용합니다. (`dimos/robot/cli/dimos.py`의 `dimos.core.run_registry` import 및 log-dir 설정)
- 데몬 실행을 쓴다면, “중지/정리” 루틴을 문서화해두는 편이 안전합니다. (근거: `dimos/robot/cli/dimos.py`에서 `stop_entry`, `cleanup_stale` 등을 참조)

### 데이터/자산(LFS)

- 리플레이/데모에서 LFS 다운로드가 걸릴 수 있으므로, 네트워크/스토리지 여유를 먼저 확인하세요. (`README.md`, `docs/development/large_file_management.md`)

---

## 2) 테스트/검증 루트

문서가 제공하는 테스트 가이드를 1차 기준으로 삼는 편이 안전합니다.

- `docs/development/testing.md`

코드에서 힌트를 얻고 싶다면:

- MCP 테스트: `dimos/agents/mcp/test_mcp_server.py`, `dimos/agents/mcp/test_mcp_client.py`
- 코어 테스트: `dimos/core/tests/*`

---

## 3) 도커/개발환경

개발환경 관련 문서/설정의 위치:

- 도커 문서: `docs/development/docker.md`
- devcontainer: `.devcontainer/` (레포 루트)

---

## 4) 트러블슈팅(자주 막히는 축)

- 실행이 안 될 때:
  - CLI 엔트리/옵션 확인: `pyproject.toml`, `dimos/robot/cli/dimos.py`
  - 설정 스키마 확인: `dimos/core/global_config.py`
- 스트림/시각화가 안 될 때:
  - 사용법 문서: `docs/usage/visualization.md`, `docs/usage/modules.md`
  - 구현: `dimos/visualization/*`, `dimos/stream/*`
- MCP 연동이 안 될 때:
  - 구현/테스트: `dimos/agents/mcp/*`

---

## 근거(파일/경로)

- 개발 문서: `docs/development/*`
- 사용법: `docs/usage/*`
- CLI/레지스트리 단서: `dimos/robot/cli/dimos.py`, `dimos/core/run_registry.py`
- MCP: `dimos/agents/mcp/*`

---

## TODO/확인 필요

- “로그 디렉토리 구조/파일명”을 실제 실행 후(리플레이 모드) 샘플로 남기기
- `docs/development/profiling_dimos.md`에 나온 프로파일링 도구를 “blueprint 단위”로 연결(어느 모듈이 CPU/GPU를 쓰는지)

---

## 위키 링크

- `[[dimos Guide - Index]]` → [가이드 목차](/blog-repo/dimos-guide/)
- `[[dimos Guide - CLI & MCP]]` → [04. CLI & MCP & Blueprints](/blog-repo/dimos-guide-04-cli-mcp-blueprints/)

