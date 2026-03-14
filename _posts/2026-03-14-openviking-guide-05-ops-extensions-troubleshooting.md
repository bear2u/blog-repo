---
layout: post
title: "OpenViking 완벽 가이드 (05) - 운영/확장/트러블슈팅: Deploy/Auth/MCP/Examples"
date: 2026-03-14
permalink: /openviking-guide-05-ops-extensions-troubleshooting/
author: volcengine
categories: [AI 에이전트, OpenViking]
tags: [Trending, GitHub, OpenViking, Operations, MCP, Troubleshooting]
original_url: "https://github.com/volcengine/OpenViking"
excerpt: "배포/모니터링/인증/MCP 연동 문서와 examples/ 디렉토리를 기준으로 운영 체크리스트를 정리합니다."
---

## 이 문서의 목적

- 운영 단계에서 필요한 문서 위치(배포/모니터링/인증/MCP)를 한 번에 찾게 합니다.
- examples/를 “어떤 상황에서 어떤 예제를 봐야 하는지”로 분류합니다.

---

## 빠른 요약 (Docs 기반)

- 배포: `docs/en/guides/03-deployment.md`
- 인증: `docs/en/guides/04-authentication.md`
- 모니터링: `docs/en/guides/05-monitoring.md`
- MCP 연동: `docs/en/guides/06-mcp-integration.md`

---

## 1) examples/ 디렉토리 빠른 내비게이션

```text
examples/
  mcp-query/
  openclaw-memory-plugin/
  claude-memory-plugin/
  opencode-memory-plugin/
  opencode/
  multi_tenant/
  k8s-helm/
  server_client/
  cloud/
  skills/
```

추천 탐색 순서(목적별):

- “서버 켜고 클라이언트 붙이기”: `examples/server_client/`
- “에이전트/도구 통합(메모리 플러그인)”: `examples/openclaw-memory-plugin/`, `examples/claude-memory-plugin/`
- “MCP”: `examples/mcp-query/` + 문서 `docs/en/guides/06-mcp-integration.md`
- “멀티 테넌시/클라우드/헬름”: `examples/multi_tenant/`, `examples/cloud/`, `examples/k8s-helm/`

---

## 2) 운영 체크리스트

### 배포/런타임

- 서버 모드 기본 확인: `/health` 응답 확인 (`docs/en/getting-started/03-quickstart-server.md`)
- 포트/방화벽: 서비스 포트(문서 기준 1933) 노출 최소화
- 데이터 영속성/백업: 스토리지 경로/권한/백업 정책 점검 (`docs/en/concepts/05-storage.md`)

### 인증/권한

- 인증 기능을 켠 경우, 클라이언트/CLI의 api_key 전달 경로를 문서대로 통일 (`docs/en/guides/04-authentication.md`, `docs/en/getting-started/03-quickstart-server.md`)

### 모니터링

- 메트릭/로그 수집 방식 확인 (`docs/en/guides/05-monitoring.md`)

---

## 3) 문서 점검 자동화(추천)

레포에는 테스트가 넓게 분포해 있어, “변경 후 최소 점검”을 루틴화하는 것이 좋습니다.

- 테스트: `tests/*` (engine/client/integration/storage/retrieve 등)
- 프리커밋: `.pre-commit-config.yaml`

예시(로컬에서의 최소 점검 루틴):

```bash
# 1) 설정 파일 경로 확인
ls -la ~/.openviking/ov.conf

# 2) 서버 기동/헬스체크
openviking-server &
sleep 2
curl -sS http://localhost:1933/health
```

---

## 4) 트러블슈팅 체크리스트

- 설정 로딩 문제:
  - 모델 설정: `OPENVIKING_CONFIG_FILE`, 기본 경로 `~/.openviking/ov.conf` (`docs/en/getting-started/02-quickstart.md`)
  - CLI 설정: `OPENVIKING_CLI_CONFIG_FILE`, 기본 경로 `~/.openviking/ovcli.conf` (`docs/en/getting-started/03-quickstart-server.md`)
- 서버 기동 실패:
  - `openviking_cli/server_bootstrap.py`에서 로깅/초기화 경로 확인
- 네이티브 확장/빌드:
  - 빌드 단서: `pyproject.toml`(cmake/pybind11), `src/CMakeLists.txt`, `third_party/*`, `openviking/pyagfs/*`

---

## 근거(파일/경로)

- 운영 문서: `docs/en/guides/03-deployment.md`, `docs/en/guides/04-authentication.md`, `docs/en/guides/05-monitoring.md`, `docs/en/guides/06-mcp-integration.md`
- examples: `examples/*`
- 테스트/자동화: `tests/*`, `.pre-commit-config.yaml`

---

## TODO/확인 필요

- `examples/` 중 “가장 짧은 MCP end-to-end”를 골라, 필요한 파일/설정 목록을 블로그에 확정 버전으로 남기기
- `docs/en/guides/06-mcp-integration.md`의 권장 구성(프로세스 관리/보안)을 실제 예제 디렉토리와 대조해 표로 정리하기

---

## 위키 링크

- `[[OpenViking Guide - Index]]` → [가이드 목차](/blog-repo/openviking-guide/)
- `[[OpenViking Guide - Usage & API]]` → [04. 사용법 & API](/blog-repo/openviking-guide-04-usage-and-api/)

