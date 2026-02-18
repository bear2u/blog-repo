---
layout: post
title: "Summarize 가이드 (03) - 입력/실행 플로우: runCli 파이프라인 해부"
date: 2026-02-18
permalink: /summarize-guide-03-input-and-run-pipeline/
author: steipete
categories: ['개발 도구', 'CLI']
tags: [runCli, Input Routing, URL Flow, File Flow, Stdin]
original_url: "https://github.com/steipete/summarize/blob/main/src/run/runner.ts"
excerpt: "src/run/runner.ts를 기준으로 summarize 명령이 입력을 해석하고 실제 요약 흐름으로 분기하는 구조를 정리합니다."
---

## CLI 엔트리 체인

엔트리 파일 흐름은 단순합니다.

- `src/cli.ts` -> `runCliMain(...)`
- `src/cli-main.ts` -> `.env` 로드 + 에러/파이프(EPIPE) 처리
- `src/run.ts` -> `runCli(...)`

실제 기능 분기는 `src/run/runner.ts`에 집중돼 있습니다.

---

## runCli에서 먼저 처리하는 것

`runCli`는 본 실행 전에 다음을 우선 처리합니다.

- help/version
- `refresh-free`
- daemon 관련 하위 커맨드
- slides/transcriber 하위 커맨드
- `--clear-cache`, `--cache-stats`

즉 한 바이너리에서 운영 명령까지 함께 처리하는 구조입니다.

---

## 입력 타입 분기

입력은 크게 세 갈래입니다.

1. URL
2. 로컬 파일
3. stdin (`-`)

문서/코드 기준으로 media 확장자, YouTube URL, podcast RSS, PDF 등의 분기 처리가 세밀하게 들어가 있습니다.

---

## 출력 모드

핵심 모드는 아래 두 가지입니다.

- summarize(기본): LLM 호출 포함
- extract(`--extract`): 추출 결과만 출력

`--json`, `--stream`, `--plain`, `--format`, `--markdown-mode` 같은 플래그가 출력 UX를 세분화합니다.

다음 장에서 모델 선택/실패 fallback 전략을 다룹니다.

