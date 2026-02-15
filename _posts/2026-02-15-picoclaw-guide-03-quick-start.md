---
layout: post
title: "PicoClaw 가이드 (03) - Quick Start: onboard, config.json, 첫 agent 실행"
date: 2026-02-15
permalink: /picoclaw-guide-03-quick-start/
author: Sipeed
categories: [AI 에이전트, 개발 도구]
tags: ["PicoClaw", "Quick Start", "Config", "OpenRouter", "Zhipu"]
original_url: "https://github.com/sipeed/picoclaw#-quick-start"
excerpt: "가장 빠른 시작은 picoclaw onboard로 홈 디렉토리에 config/workspace를 만들고, providers에 API 키를 넣은 뒤 agent를 실행하는 것입니다."
---

## 1) 초기화: `picoclaw onboard`

README가 제시하는 첫 단계는 `onboard`입니다.

```bash
picoclaw onboard
```

이 과정은 보통:

- `~/.picoclaw/config.json` 생성/초기화
- 기본 워크스페이스 디렉토리 준비

를 포함합니다(구체 산출물은 버전에 따라 달라질 수 있으니 생성된 파일을 확인합니다).

---

## 2) 설정 파일 위치: `~/.picoclaw/config.json`

README는 설정 파일 경로를 `~/.picoclaw/config.json`으로 안내합니다.

핵심은 다음 두 블록입니다.

- `agents.defaults`: 작업 디렉토리(workspace), 모델, 토큰/온도, 반복 횟수 등
- `providers`: OpenRouter/Zhipu 등 LLM 프로바이더 API 키/베이스 URL

최소 구성(개념 예시):

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.picoclaw/workspace",
      "model": "glm-4.7",
      "max_tokens": 8192,
      "temperature": 0.7,
      "max_tool_iterations": 20
    }
  },
  "providers": {
    "openrouter": {
      "api_key": "sk-or-v1-..."
    }
  }
}
```

---

## 3) 첫 실행: agent 모드

단발 실행:

```bash
picoclaw agent -m "What is 2+2?"
```

대화형:

```bash
picoclaw agent
```

여기서 목표는 “설정 파일을 읽고, 선택한 프로바이더로 LLM 호출이 성공한다”까지 확인하는 것입니다.

다음 장에서는 로컬에 아무것도 설치하지 않고도 실행할 수 있는 Docker Compose 구성을 정리합니다.

