---
layout: post
title: "PicoClaw 가이드 (05) - 설정 파일: ~/.picoclaw/config.json 구조 해부"
date: 2026-02-15
permalink: /picoclaw-guide-05-config/
author: Sipeed
categories: [AI 에이전트, 개발 도구]
tags: ["PicoClaw", "Config", "Agents", "Channels", "Providers", "Tools"]
original_url: "https://github.com/sipeed/picoclaw/blob/main/config/config.example.json"
excerpt: "config.example.json을 기준으로 agents, channels, providers, tools, heartbeat, gateway 등의 의미와 실전에서 먼저 손대야 할 항목을 정리합니다."
---

## 전체 구조(큰 그림)

`config/config.example.json` 기준으로, 주요 블록은 다음과 같습니다.

- `agents.defaults`: 모델/토큰/온도/도구 반복, 워크스페이스 제한
- `channels`: Telegram/Discord/LINE/Slack 등 입력 채널
- `providers`: Anthropic/OpenAI/OpenRouter/Zhipu/Groq 등 LLM 프로바이더
- `tools`: 웹 검색 같은 도구 설정
- `heartbeat`: 주기 작업(기본 30분)
- `gateway`: 게이트웨이 서버 바인딩(host/port)

---

## agents.defaults: 실행 기본값

여기서 가장 중요한 옵션은 두 가지입니다.

- `workspace`: 에이전트 작업 디렉토리(세션/메모리/스킬 포함)
- `restrict_to_workspace`: 기본값 `true`로, 파일/명령 접근을 워크스페이스 내부로 제한

그 외 모델/토큰/온도/도구 반복은 “비용/품질/속도”에 직접 영향을 줍니다.

---

## providers: LLM 연결

README는 다음을 예로 듭니다.

- OpenRouter(여러 모델 접근)
- Zhipu(GLM 계열)
- Groq(빠른 추론, 보이스 트랜스크립션도 언급)

운영 팁:

- 처음에는 **프로바이더 1개**만 정상 동작시키고(키/베이스 URL)
- 그 다음에 “모델/비용”을 비교하며 확장하는 게 안전합니다.

---

## channels: 게이트웨이 입력면

게이트웨이를 띄우면, “어디서 메시지를 받을지”가 channels로 결정됩니다.

- Telegram은 토큰 + allow list(사용자 ID)
- Discord는 토큰 + intents 설정(README에 안내)
- LINE은 webhook host/port/path까지 필요(HTTPS 요구)

allow list(`allow_from`, `allowFrom`)는 기본적인 방어선이므로, 사내/개인 운영에서는 먼저 켜는 편이 안전합니다.

---

## heartbeat: 주기 작업

README는 `HEARTBEAT.md`를 워크스페이스에 두면 주기적으로 읽고 작업한다고 설명합니다.

- 주기 기본값: 30분(최소 5분)
- 길게 걸리는 작업은 subagent(spawn)로 비동기 처리하는 흐름을 안내

이 기능은 “에이전트가 스스로 일을 한다”를 만들 수 있지만, 동시에 운영 리스크도 올리므로 9장에서 샌드박스/가드레일과 함께 다룹니다.

다음 장에서는 “프로바이더/모델”을 좀 더 구체적으로 정리합니다.

