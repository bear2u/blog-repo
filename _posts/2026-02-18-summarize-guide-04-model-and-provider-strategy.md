---
layout: post
title: "Summarize 가이드 (04) - 모델/프로바이더 전략: auto 선택, fallback, CLI 모델"
date: 2026-02-18
permalink: /summarize-guide-04-model-and-provider-strategy/
author: steipete
categories: ['개발 도구', 'LLM']
tags: [Model Auto, OpenRouter, OpenAI, Gemini, Claude, Codex]
original_url: "https://github.com/steipete/summarize/blob/main/docs/model-auto.md"
excerpt: "auto 모델 선택과 provider fallback, CLI 모델 연동 규칙을 문서 기준으로 실무 관점에서 정리합니다."
---

## 기본 모델은 auto

Summarize의 기본값은 `--model auto`입니다.

auto 모드에서 하는 일:

- 입력 종류/토큰 수를 보고 candidate 순서 생성
- API 키가 없는 후보는 건너뜀
- 실패 시 다음 후보 재시도
- 최종적으로 모델이 없으면 추출 텍스트를 그대로 반환

---

## 모델 ID 규칙

문서 기준 모델 식별자는 크게 두 계열입니다.

- native: `openai/...`, `google/...`, `anthropic/...`, `xai/...`, `zai/...`
- openrouter 강제: `openrouter/<author>/<model>`

native 후보에 대해 조건이 맞으면 OpenRouter fallback attempt를 자동으로 붙일 수 있습니다.

---

## CLI 모델 백엔드

`docs/cli.md` 기준으로 로컬 CLI를 모델 백엔드로 사용할 수 있습니다.

- `cli/claude/...`
- `cli/codex/...`
- `cli/gemini/...`
- `cli/agent/...` (Cursor Agent)

`--cli` 플래그 또는 `cli.enabled`, `cli.autoFallback` 설정으로 auto 체인에 포함시킬 수 있습니다.

---

## 실무에서 중요한 규칙

1. 명시 모델(`--model`)이 있으면 그것이 최우선
2. 키가 없으면 후보가 자동 필터링됨
3. auto 모드 실패 시 graceful fallback이 기본 철학

다음 장에서 config/env/플래그 우선순위를 정리합니다.

