---
layout: post
title: "Claude Code System Prompts 가이드 (02) - 프롬프트 카탈로그 구조: 네이밍, 메타데이터, 변수"
date: 2026-02-15
permalink: /claude-code-system-prompts-guide-02-taxonomy-and-metadata/
author: Piebald AI
categories: [AI 코딩 에이전트, Claude Code]
tags: [Claude Code, System Prompts, Metadata, Variables, Token Count, README]
original_url: "https://github.com/Piebald-AI/claude-code-system-prompts"
excerpt: "system-prompts/의 파일 네이밍 규칙과 HTML 주석 메타데이터(ccVersion, variables)를 읽는 법을 정리합니다."
---

## system-prompts/는 ‘프롬프트 카탈로그’다

이 레포의 핵심은 `system-prompts/` 디렉토리입니다. 여기에는 “Claude Code가 런타임에 조립해 쓰는 프롬프트 조각”이 **마크다운 파일**로 정리돼 있습니다.

대표적인 파일 네이밍은 다음 패턴으로 시작합니다.

```text
agent-prompt-*.md          # 서브에이전트/유틸리티용 프롬프트
system-prompt-*.md         # 메인 시스템 프롬프트 조각들
system-reminder-*.md       # 런타임 이벤트 알림(Plan Mode/파일/훅 등)
tool-description-*.md      # 내장 툴 설명
data-*.md                  # 내장 템플릿/데이터 파일(워크플로우 템플릿 등)
skill-*.md                 # “스킬”로 제공되는 기능의 시스템 프롬프트
tool-parameter-*.md        # 특정 툴 파라미터/프로토콜 관련 조각(레포에 존재할 수 있음)
```

예시(실제 파일):

- `system-prompts/system-prompt-main-system-prompt.md`
- `system-prompts/system-prompt-tool-usage-policy.md`
- `system-prompts/tool-description-bash.md`
- `system-prompts/system-reminder-plan-mode-is-active-iterative.md`
- `system-prompts/agent-prompt-explore.md`

---

## 파일 상단의 HTML 주석 메타데이터

이 레포는 YAML front matter 대신, 파일 상단에 **HTML 주석**으로 메타데이터를 붙입니다.

예를 들어 `system-prompts/system-prompt-tool-usage-policy.md`의 시작은 대략 이런 형태입니다.

```text
<!--
name: 'System Prompt: Tool usage policy'
description: Policies and guidelines for tool usage
ccVersion: 2.1.41
variables:
  - WEBFETCH_ENABLED_SECTION
  - MCP_TOOLS_SECTION
  ...
-->
```

여기서 중요한 필드는 3개입니다.

1. `name`: README에 노출되는 “사람이 읽는 이름”
2. `description`: README의 한 줄 설명으로 쓰이는 문구(자동 생성/정규화)
3. `ccVersion`: 해당 프롬프트 조각이 추출된 Claude Code 버전

그리고 `variables:`는, 프롬프트 본문에 `${...}` 형태로 **삽입되는 변수(식별자)** 목록입니다.

---

## variables는 ‘런타임 보간 자리 표시자’다

이 레포는 Claude Code의 번들에서 문자열을 추출한 결과라서, `${BASH_TOOL_NAME}` 같은 표기가 그대로 남아 있습니다.

이건 “마크다운 템플릿 문법”이 아니라:

- Claude Code가 내부에서 문자열을 조립할 때
- 런타임 환경(도구 이름, 옵션, 활성화된 기능) 등에 따라
- 해당 자리를 다른 문자열로 채우는

**보간(interpolation) 위치**를 의미합니다.

예: `system-prompts/system-prompt-main-system-prompt.md`를 보면 다음처럼 조건부 문장 조립이 들어 있습니다.

```text
You are an interactive CLI tool that helps users ${OUTPUT_STYLE_CONFIG!==null? ... }
${SECURITY_POLICY}
```

즉, “프롬프트가 단일 문자열”인 게 아니라 “조각 + 변수 + 조건”의 조립 결과라는 사실이 여기서 드러납니다.

---

## README의 토큰 카운트는 ‘대략적인 감각’이다

README에는 각 항목 옆에 `(**516** tks)` 같은 토큰 수가 붙습니다. 이 숫자는:

- 레포의 업데이트 스크립트가
- 특정 모델(예: `claude-sonnet-4-20250514`)로
- 프롬프트 텍스트를 사용자 메시지로 넣어
- Anthropic의 토큰 카운트 API로 계산한 값입니다.

다만 README에서도 언급하듯, 실제 Claude Code 세션에서는:

- 변수 보간 결과가 달라지고
- 환경/도구 목록/서브에이전트 목록이 달라져서

±(수십 토큰) 정도 차이가 날 수 있습니다.

이 숫자는 “정확한 과학”이라기보다:

- 어떤 조각이 큰지(컨텍스트 비용)
- 어떤 조각이 자주 변하는지

감 잡는 지표로 보는 게 좋습니다.

---

## 카테고리(Agent/System/Reminder/Tool/Data)를 왜 나눴을까

이 레포를 이해하는 가장 빠른 프레임은 다음입니다.

- **System Prompt**: Claude Code라는 제품의 “정체성/정책/작업 방식”
- **Tool Description**: 도구를 호출할 때의 “계약(Contract)”
- **System Reminder**: UI/상태 변화 이벤트에 대한 “런타임 규칙 주입”
- **Agent Prompt**: 서브에이전트/유틸리티의 “역할 분리”
- **Data**: 제품 안에 박혀 있는 템플릿/문서 조각

같은 “시스템 프롬프트 텍스트”라도, 언제/어디서/어떤 맥락으로 주입되느냐가 다르기 때문에, 레포도 이를 구분해 정리합니다.

---

## 다음 장 예고

2장에서 “읽는 법(분류/메타/변수)”을 잡았으니, 다음은 “어떻게 만들어지는가(추출/갱신)”입니다.

3장에서는 `tools/updatePrompts.js`를 중심으로:

- 프롬프트 JSON을 어떻게 마크다운 파일로 재구성하는지
- 변경/신규/삭제를 어떻게 감지하는지
- README의 버전/토큰 카운트를 어떻게 자동 갱신하는지

를 따라가 봅니다.

---

*다음 글에서는 업데이트 스크립트의 핵심 로직(name→filename, pieces+identifiers 재조립, 토큰 카운트)을 해부합니다.*

