---
layout: post
title: "GitHub Agentic Workflows(gh-aw) 가이드 (06) - 컴파일/락파일: gh aw compile, strict 검증, 스캐너"
date: 2026-02-15
permalink: /gh-aw-guide-06-compilation-and-lockfiles/
author: GitHub
categories: [개발 도구, GitHub]
tags: [gh-aw, Compilation, Lockfiles, Strict, actionlint, zizmor, poutine]
original_url: "https://github.github.com/gh-aw/reference/compilation-process/"
excerpt: "gh-aw는 .md 워크플로우를 .lock.yml로 컴파일합니다. 이 장에서는 compile의 핵심 옵션(--watch/--validate/--strict/--purge)과 스캐너(actionlint/zizmor/poutine)를 정리합니다."
---

## 컴파일이 왜 필요한가

`.md` 워크플로우는 사람이 읽기 좋지만, GitHub Actions가 직접 실행할 수는 없습니다.
그래서 gh-aw는 `.md`를 `.lock.yml`로 변환합니다.

컴파일 단계에서 동시에 일어나는 것들:

- frontmatter 스키마 검증
- imports 병합/해석
- job 구성(activation/agent/safe-outputs 등)
- 액션 pinning(SHA 고정)
- `.lock.yml` 생성

---

## 기본 사용법

```bash
# 전체 워크플로우 컴파일
gh aw compile

# 특정 워크플로우만 컴파일(이름)
gh aw compile my-workflow
```

---

## 자주 쓰는 옵션

- `--watch`: 파일 변경을 감지해 자동 재컴파일
- `--validate`: 스키마 검증을 강화
- `--strict`: 보안/관행을 “에러”로 강제(프로덕션 권장)
- `--fix`: 컴파일 전에 codemod 기반 자동 수정
- `--purge`: orphaned `.lock.yml` 제거

실전 루틴:

```bash
gh aw compile --fix --validate --strict
```

---

## 스캐너: actionlint / zizmor / poutine

문서는 보안/품질 검증을 컴파일 파이프라인에 붙이는 옵션들을 제공합니다.

- `--actionlint`: Actions 문법/셸 관련 검증(상황에 따라 shellcheck 포함)
- `--zizmor`: 보안 취약점 스캐너(경고 또는 strict에서 실패)
- `--poutine`: 공급망/의존성/핀ning 관련 검증

이들은 “운영에 올리기 전에” 자동으로 걸러내는 쪽이 가치가 큽니다.

---

## 락파일 운영 원칙

1. `.lock.yml`을 직접 고치지 않는다
2. `.md`를 고친 뒤 `gh aw compile`로 재생성한다
3. `.md`와 `.lock.yml`은 함께 커밋한다
4. 삭제한 워크플로우는 `--purge`로 orphan lock을 정리한다

---

*다음 글에서는 워크플로우를 실제로 실행하는 방법(`gh aw run`, 인터랙티브 모드, 입력 수집)을 정리합니다.*

