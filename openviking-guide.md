---
layout: page
title: OpenViking 가이드
permalink: /openviking-guide/
icon: fas fa-database
---

# OpenViking 완벽 가이드

> **AI 에이전트를 위한 Context Database (Filesystem Paradigm)**

**OpenViking**은 메모리/리소스/스킬을 “파일 시스템 패러다임”으로 통합 관리해, 에이전트의 컨텍스트 로딩/검색/회상을 구조화하려는 오픈소스 프로젝트입니다. (`README.md`, `docs/en/concepts/*`)

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/openviking-guide-01-intro/) | 프로젝트 목표, 핵심 개념, 레포 지도 |
| 02 | [설치 및 빠른 시작](/blog-repo/openviking-guide-02-install-and-quickstart/) | 설치, 모델 설정, 로컬/서버 모드 퀵스타트 |
| 03 | [아키텍처](/blog-repo/openviking-guide-03-architecture/) | Python SDK/서버/FastAPI, 네이티브 확장, Rust CLI 구성 |
| 04 | [사용법 & API](/blog-repo/openviking-guide-04-usage-and-api/) | 리소스/FS/검색/세션 API, CLI/HTTP 연결 포인트 |
| 05 | [운영/확장/트러블슈팅](/blog-repo/openviking-guide-05-ops-extensions-troubleshooting/) | 배포/모니터링/인증/MCP, 예제, 문제 해결 체크리스트 |

---

## 빠른 시작(공식 문서)

```bash
pip install openviking --upgrade --force-reinstall
openviking-server
curl http://localhost:1933/health
```

---

## 관련 링크

- [GitHub 저장소](https://github.com/volcengine/OpenViking)
- [Docs](https://www.openviking.ai/docs)

