---
layout: post
title: "Acontext 완벽 가이드 (3) - 세션 & 메시지 저장"
date: 2026-02-06
permalink: /acontext-guide-03-session-messages/
author: memodb-io
categories: [AI 에이전트, Acontext]
tags: [Acontext, Sessions, Messages, Multi-provider, Multi-modal]
original_url: "https://docs.acontext.io/store/messages/multi-provider"
excerpt: "Acontext 세션의 메시지 저장 모델과 고급 옵션을 정리합니다."
---

## 멀티 프로바이더 메시지

Acontext는 OpenAI/Anthropic/Gemini 형식 메시지를 저장하고 조회 시 자동 변환합니다.

- 저장 시 `format` 지정
- 조회 시 원하는 `format`으로 반환

## 멀티모달 메시지

이미지/오디오/PDF를 메시지에 포함할 수 있습니다.

- URL 기반 입력
- base64 기반 입력
- 프로바이더 형식 간 변환 지원

## 세션 설정(Configs)

`configs`는 세션 수명 동안 유지되는 설정 키-값입니다.

- 생성 시 초기 configs 부여
- `update_configs`: 전체 교체
- patch/filter로 조건 조회 가능

## 메시지 메타데이터

각 메시지에 `meta`를 붙여서 추적 정보를 분리 관리합니다.

- 예: `source`, `request_id`, `user_agent`
- 조회 결과에서 `items`, `ids`, `metas`를 함께 다룸

## Anthropic 특수 플래그

Acontext는 Anthropic의 `cache_control` 같은 플래그를 보존합니다. 즉, 프롬프트 캐시 관련 제어를 저장/재조회 흐름에서 유지할 수 있습니다.
