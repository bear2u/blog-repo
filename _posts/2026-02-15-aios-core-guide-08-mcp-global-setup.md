---
layout: post
title: "Synkra AIOS Core 완벽 가이드 (08) - MCP 글로벌 셋업: ~/.aios로 도구 설정 공유"
date: 2026-02-15
permalink: /aios-core-guide-08-mcp-global-setup/
author: Synkra AIOS Team
categories: [AI 코딩 에이전트, AIOS]
tags: [AIOS, MCP, Model Context Protocol, Tooling, Docker, Credentials]
original_url: "https://github.com/SynkraAI/aios-core"
excerpt: "MCP 서버 설정을 프로젝트마다 반복하지 않도록, ~/.aios 기반 글로벌 구성과 템플릿 추가/커스텀 등록 흐름을 정리합니다."
---

## MCP(Global) 시스템이 필요한 이유

프로젝트가 늘어나면 “같은 도구(MCP 서버) 설정을 매번 복사”하는 일이 반복됩니다.

AIOS 문서는 이를 해결하기 위해:

- MCP 서버 구성을 **한 번만** 해두고
- 모든 AIOS 프로젝트에서 공유

하는 **글로벌 MCP 시스템**을 제안합니다.

---

## 글로벌 디렉토리 구조

문서가 제시하는 기본 구조는 다음입니다.

```
~/.aios/
├── mcp/
│   ├── global-config.json
│   ├── servers/
│   └── cache/
└── credentials/
    └── .gitignore
```

포인트:
- `credentials/`를 별도로 두고, 실수로 커밋되지 않도록 방지합니다.

---

## 권장 아키텍처(2-레이어)

문서는 보통 “직접 MCP + docker-gateway”의 2-레이어 구성을 추천합니다.

- 직접 MCP: 호스트 접근이 필요한 것(파일/터미널 등)
- docker-gateway: API 기반 도구(컨테이너에서 돌려 컨텍스트 비용/격리 이점)

개념 그림:

```
[클라이언트(~/.claude.json 등)]
  - desktop-commander
  - playwright
  - docker-gateway
         |
         v
[docker-gateway 내부 MCP들]
  - Context7, EXA, Apify ...
```

---

## 초기 설정

문서 기준으로는 다음 명령이 등장합니다.

```bash
# 글로벌 구조 및 설정 생성
aios mcp setup

# 상태 확인
aios mcp status
```

---

## 템플릿으로 MCP 서버 추가

문서에 나온 예:

```bash
aios mcp add context7
aios mcp add exa
aios mcp add github
aios mcp add puppeteer
aios mcp add filesystem
aios mcp add memory
aios mcp add desktop-commander
```

---

## 커스텀 MCP 서버 등록

JSON으로 직접 등록하는 형태도 소개합니다.

```bash
aios mcp add my-server --config='{"command":"npx","args":["-y","my-mcp-server"]}'
```

또는 파일로:

```bash
aios mcp add my-server --config-file=./my-server-config.json
```

---

## 보안 관점 체크

MCP는 도구 실행 권한과 연결되기 쉬우므로 최소한 아래는 지키는 편이 좋습니다.

- 키/토큰은 `~/.aios/credentials/` 같은 안전한 위치에만 저장
- 프로젝트 레포에 키가 섞이지 않도록 `.gitignore` 점검
- 컨테이너에서 돌릴 수 있는 MCP는 격리(docker-gateway)를 우선

---

*다음 글에서는 비용을 크게 낮추기 위한 LLM 라우팅(claude-max/claude-free, DeepSeek) 구성과 주의점을 정리합니다.*
