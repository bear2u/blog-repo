---
layout: post
title: "Claude Code System Prompts 가이드 (03) - 추출/업데이트 파이프라인: tools/updatePrompts.js"
date: 2026-02-15
permalink: /claude-code-system-prompts-guide-03-extraction-pipeline/
author: Piebald AI
categories: [AI 코딩 에이전트, Claude Code]
tags: [Claude Code, Extraction, Node.js, Anthropic API, Token Count, Changelog]
original_url: "https://github.com/Piebald-AI/claude-code-system-prompts"
excerpt: "prompts-*.json에서 system-prompts/*.md와 README 토큰 카운트를 생성/갱신하는 updatePrompts.js의 핵심 흐름을 정리합니다."
---

## 전체 흐름(한 장 요약)

이 레포는 “Claude Code 번들에서 뽑아낸 프롬프트 JSON”을 입력으로 받아, 다음을 자동 생성/갱신합니다.

1. `system-prompts/*.md` 파일들(메타데이터 + 본문)
2. `README.md`의 목록/토큰 카운트/버전 표기
3. `system-prompts/`에서 사라진 파일 정리(삭제)

이 작업의 중심이 `tools/updatePrompts.js`입니다.

---

## 실행 인터페이스

스크립트는 이렇게 실행되도록 설계돼 있습니다.

```bash
node tools/updatePrompts.js /path/to/prompts-2.1.42.json
```

중요한 전제:

- `ANTHROPIC_API_KEY` 환경 변수가 필요합니다.
- 토큰 카운트를 계산하기 위해 Anthropic API(`messages/count_tokens`)를 호출합니다.

```bash
export ANTHROPIC_API_KEY="..."
node tools/updatePrompts.js /path/to/prompts-*.json
```

---

## name → filename 변환 규칙

README에 등장하는 인간 친화적 이름은, 파일명으로 변환됩니다.

예:

- `Agent Prompt: Explore` → `agent-prompt-explore.md`
- `System Prompt: Tool usage policy` → `system-prompt-tool-usage-policy.md`
- `Tool Description: Bash` → `tool-description-bash.md`
- `System Reminder: Plan mode is active (iterative)` → `system-reminder-plan-mode-is-active-iterative.md`

스크립트는 prefix를 정한 뒤 소문자/하이픈으로 정규화합니다.

이 규칙 덕분에, “어떤 파일이 무엇을 담는지”가 직관적으로 보입니다.

---

## pieces + identifiers: ‘분해된 문자열’을 다시 조립하는 이유

프롬프트가 단일 문자열이라면 그대로 쓰면 되겠지만, Claude Code 내부 구현에서는 프롬프트를 다음처럼 “조각(pieces) + 식별자(identifiers)”로 들고 있는 경우가 있습니다.

`updatePrompts.js`는 이를 다음 로직으로 복원합니다.

1. pieces를 순서대로 붙인다
2. pieces 사이에 identifierMap으로 매핑된 변수명을 `${...}` 형태로 삽입한다

결과적으로 마크다운 본문에는 다음이 그대로 남습니다.

- `${BASH_TOOL_NAME}`
- `${SECURITY_POLICY}`
- `${CONDITIONAL_DELEGATE_CODEBASE_EXPLORATION}`

이 레포가 “런타임 보간 지점”을 보여주는 이유가 여기 있습니다.

---

## HTML 주석 메타데이터 생성

스크립트는 파일 상단에 HTML 주석 블록을 붙입니다.

- `name`: 프롬프트 이름
- `description`: 프롬프트 설명
- `ccVersion`: Claude Code 버전
- `variables`: identifierMap에서 수집한 변수 목록

이 메타는:

- README 목록을 생성할 때도 쓰이고
- 변경 감지(기존 파일과 비교)에도 간접적으로 기여합니다.

---

## 변경 감지: changed/new/unchanged

업데이트의 핵심은 “필요한 것만 다시 토큰 카운트”하는 최적화입니다.

- 기존 파일이 있고 내용이 동일하면: unchanged
  - README에 이미 있는 토큰 값을 재사용
- 기존 파일이 있고 내용이 다르면: changed
  - 파일을 교체하고 토큰을 다시 계산
- 파일이 없으면: new
  - 새로 생성하고 토큰을 계산

그리고 JSON에 없는데 `system-prompts/`에 남아 있는 `.md`는 deleted로 보고 제거합니다.

---

## 토큰 카운트는 어떻게 계산하나

스크립트는 Anthropic의 `messages/count_tokens` 엔드포인트로 프롬프트 텍스트의 입력 토큰 수를 계산합니다.

핵심 포인트:

- 모델은 코드에 하드코딩돼 있습니다(`claude-sonnet-4-20250514`).
- 5개 단위로 배치 처리하고, 배치 사이에 딜레이(기본 100ms)를 둡니다.
- 실패 시 해당 파일 토큰은 0으로 기록될 수 있습니다(로그 출력).

즉, README의 토큰 카운트는 “세밀한 측정 장치”라기보다는:

- 대략의 컨텍스트 비용
- 커다란 조각이 어디인지

를 잡는 용도에 가깝습니다.

---

## README 자동 갱신: 버전/릴리즈 날짜/카테고리 재구성

`updateReadme()`는 다음을 합니다.

1. 상단 소개 줄의 Claude Code 버전/날짜를 갱신
2. 프롬프트를 카테고리별로 다시 분류하고(Agent/System/Reminder/Tool/Data)
3. 각 카테고리 내에서 알파벳 정렬
4. README의 해당 섹션을 재생성해서 덮어쓰기

릴리즈 날짜는 npm 레지스트리의 `@anthropic-ai/claude-code` 패키지 메타데이터에서 가져옵니다.

이 설계 덕분에 README는 “사람이 업데이트하는 문서”가 아니라 “빌드 산출물”에 가까워집니다.

---

## 이 파이프라인이 주는 실전적인 이점

Claude Code의 소스는 공개되지 않고, 프롬프트는 번들 JS 속에서 계속 이동합니다.

그래서 “어느 버전에서 어떤 문구가 바뀌었는지”를 수동으로 추적하는 건 사실상 불가능에 가깝습니다.

이 레포의 파이프라인은:

- 추출 결과를 파일로 고정하고
- 토큰 비용과 설명을 붙여 카탈로그화하고
- CHANGELOG로 변화의 의미를 요약하는 방식으로

**관측 가능성(observability)**을 만들어 줍니다.

---

## 다음 장 예고

이제 “어떻게 생성되는가”를 알았으니, 다음은 “무엇이 들어 있는가”를 파고들 차례입니다.

4장에서는 System Prompt 계열을 중심으로:

- 메인 정체성 프롬프트가 최소한으로 어떤 것을 말하는지
- 톤/스타일, 조심스러운 실행, 학습모드 같은 가드레일이 어떻게 분리돼 있는지

를 살펴봅니다.

---

*다음 글에서는 `System Prompt: Main system prompt`, `Tone and style`, `Executing actions with care`, `Learning mode` 조각을 묶어 “Claude Code의 기본 인격”을 해부합니다.*

