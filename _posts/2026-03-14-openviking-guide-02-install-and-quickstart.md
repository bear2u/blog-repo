---
layout: post
title: "OpenViking 완벽 가이드 (02) - 설치 및 빠른 시작(로컬/서버)"
date: 2026-03-14
permalink: /openviking-guide-02-install-and-quickstart/
author: volcengine
categories: [AI 에이전트, OpenViking]
tags: [Trending, GitHub, OpenViking, Installation, Quickstart, FastAPI]
original_url: "https://github.com/volcengine/OpenViking"
excerpt: "OpenViking 설치→모델 설정(ov.conf)→로컬 예제→서버 모드(openviking-server)까지, 공식 문서 경로를 따라 빠르게 실행합니다."
---

## 이 문서의 목적

- “일단 돌아가게” 만드는 최소 경로(설치 → 설정 → 예제 실행)를 정리합니다.
- 로컬 모드(Python SDK)와 서버 모드(HTTP 서버 + 클라이언트)를 구분해 헷갈림을 줄입니다.

---

## 빠른 요약 (Docs 기반)

- 설치: `pip install openviking --upgrade --force-reinstall` (`docs/en/getting-started/02-quickstart.md`)
- 설정: `~/.openviking/ov.conf`에 embedding/vlm 설정 작성 (`docs/en/getting-started/02-quickstart.md`, `docs/en/guides/01-configuration.md`)
- 서버: `openviking-server` 실행 후 `/health`로 확인 (`docs/en/getting-started/03-quickstart-server.md`)

---

## 1) 설치

```bash
pip install openviking --upgrade --force-reinstall
```

선택적으로 Rust CLI를 설치할 수 있습니다. (`README.md`, `crates/ov_cli/install.sh`)

---

## 2) 모델 설정(필수): `~/.openviking/ov.conf`

OpenViking은 최소한 **Embedding**과 **VLM** 설정을 요구합니다. 설정 템플릿/예시는 아래 문서를 기준으로 합니다.

- 템플릿: `docs/en/getting-started/02-quickstart.md`
- 예시 모음: `docs/en/guides/01-configuration.md#configuration-examples`

기본 경로를 쓰면 자동 로딩됩니다.

```bash
mkdir -p ~/.openviking
${EDITOR:-vi} ~/.openviking/ov.conf
```

기본 경로가 아니라면:

```bash
export OPENVIKING_CONFIG_FILE=/path/to/ov.conf
```

---

## 3) 로컬 모드: Python SDK로 첫 실행

공식 문서의 “Run Your First Example” 흐름은 대략 아래 API 조합으로 구성됩니다. (`docs/en/getting-started/02-quickstart.md`)

- 초기화: `client.initialize()`
- 리소스 추가: `client.add_resource(...)`
- 트리 탐색/읽기: `ls`, `glob`, `read`
- 처리 대기: `wait_processed`
- 요약/검색: `abstract`, `overview`, `find`

코드 근거:

- 클라이언트/SDK: `openviking/client.py`, `openviking/sync_client.py`, `openviking/async_client.py`
- 리소스/FS/검색 서비스: `openviking/service/*`

---

## 4) 서버 모드: `openviking-server` 실행/확인

```bash
openviking-server
curl http://localhost:1933/health
```

문서에 따르면 서버 모드에서는 Python SDK 외에 CLI/curl로도 연결할 수 있습니다. (`docs/en/getting-started/03-quickstart-server.md`)

---

## 5) (옵션) CLI로 연결하기

서버 모드에 연결하려면 CLI 설정 파일(`~/.openviking/ovcli.conf`)을 만들고, `openviking ...` 명령을 사용합니다. (`docs/en/getting-started/03-quickstart-server.md`)

CLI 엔트리포인트/브리지:

- 스크립트 엔트리: `pyproject.toml`의 `[project.scripts]` (`openviking`, `ov`, `openviking-server`)
- Rust CLI 래퍼: `openviking_cli/rust_cli.py`
- Rust 구현: `crates/ov_cli/src/main.rs`

---

## 근거(파일/경로)

- 퀵스타트(로컬): `docs/en/getting-started/02-quickstart.md`
- 퀵스타트(서버): `docs/en/getting-started/03-quickstart-server.md`
- 설정 가이드: `docs/en/guides/01-configuration.md`
- 엔트리포인트: `pyproject.toml`
- 서버 부트스트랩: `openviking_cli/server_bootstrap.py`

---

## 주의사항/함정

- 설정 파일의 provider/model/api_key 등은 환경/벤더에 따라 달라 “복붙”으로 해결되지 않습니다. 예시를 참고해 **자신의 endpoint**로 채우세요. (`docs/en/guides/01-configuration.md`)
- 서버 포트는 기본이 1933으로 문서에 명시되어 있습니다. (`docs/en/getting-started/03-quickstart-server.md`)

---

## TODO/확인 필요

- `openviking/server/config.py`의 실제 설정 키/기본값과 문서 예시의 대응 관계 표로 정리하기
- `examples/` 중 “가장 짧은 end-to-end 예제”를 1개 골라 재현 절차(환경/파일)로 정리하기

---

## 위키 링크

- `[[OpenViking Guide - Index]]` → [가이드 목차](/blog-repo/openviking-guide/)
- `[[OpenViking Guide - Architecture]]` → [03. 아키텍처](/blog-repo/openviking-guide-03-architecture/)

---

*다음 글에서는 `openviking/` / `openviking_cli/` / `crates/ov_cli/`의 책임 경계를 기준으로 아키텍처를 그립니다.*

