---
layout: post
title: "ClawRouter 완벽 가이드 (03) - 라우팅 프로필과 티어"
date: 2026-02-15
permalink: /clawrouter-guide-03-routing-profiles/
author: BlockRun
categories: [AI 에이전트, ClawRouter]
tags: [Routing, Profiles, Tiers, blockrun/auto, Savings, Agentic]
original_url: "https://github.com/BlockRunAI/ClawRouter"
excerpt: "auto/eco/premium/free 프로필, SIMPLE/MEDIUM/COMPLEX/REASONING 티어, agentic auto-detect와 tool detection을 정리합니다."
---

## 프로필: auto / eco / premium / free

README 기준으로 라우팅 전략은 4가지 프로필로 정리됩니다.

- `auto`: 기본(균형)
- `eco`: 비용 최우선
- `premium`: 품질 최우선
- `free`: 무료 티어만

채팅에서 프로필을 바꾸는 명령 예시:

```text
/model eco
```

---

## 티어: SIMPLE / MEDIUM / COMPLEX / REASONING

ClawRouter는 요청을 분류해 티어를 정하고, 티어에 매핑된 모델로 라우팅합니다.

README에 나온 대표 예시(요지):

- SIMPLE: 사실/짧은 질문
- MEDIUM: 요약/설명/추출
- COMPLEX: 코드/복합 작업
- REASONING: 증명/형식 논리/다단계 추론

---

## 자동 기능: agentic auto-detect / tool detection

`docs/features.md`는 "자동으로 동작하는 고급 기능"을 강조합니다.

- agentic auto-detect: 멀티스텝 작업 신호(읽기/수정/실행/테스트 등)가 2개 이상이면 agentic tier로 전환
- tool detection: 요청에 `tools` 배열(function calling)이 있으면 tool-use에 강한 티어로 전환

---

## 세션 핀ning(Session persistence)

멀티턴 대화에서는 모델이 중간에 바뀌면 작업 일관성이 깨질 수 있습니다.
문서는 이를 막기 위해 "세션 동안 모델을 핀"하는 동작을 설명합니다.

---

## 다음 글

다음 글에서는 실제 운영자가 건드릴 설정(환경변수, 포트, overrides, scoring weights)을 레퍼런스 형태로 정리합니다.

- 다음: [ClawRouter (04) - 설정 레퍼런스](/blog-repo/clawrouter-guide-04-configuration/)
