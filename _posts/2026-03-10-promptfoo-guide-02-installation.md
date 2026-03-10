---
layout: post
title: "promptfoo 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-03-10
permalink: /promptfoo-guide-02-installation/
author: "Ian Webster"
categories: [개발 도구, promptfoo]
tags: [Trending, GitHub, promptfoo]
original_url: "https://github.com/promptfoo/promptfoo"
excerpt: "promptfoo 설치와 첫 평가 실행 흐름"
---

## 요구사항 체크(README 기준)

- Node.js(프로젝트 `package.json`의 `engines.node` 참고)
- LLM Provider API Key(예: `OPENAI_API_KEY`)

---

## 설치

README의 Quick Start 기준으로 시작합니다.

```bash
npm install -g promptfoo
promptfoo init --example getting-started
```

대안:

- `brew install promptfoo`
- `pip install promptfoo`
- `npx promptfoo@latest <command>`

---

## 실행/첫 사용

예제 디렉토리에서 eval → 결과 뷰어 순서로 확인합니다.

```bash
cd getting-started
promptfoo eval
promptfoo view
```

---

## 팁

- API 키는 보통 환경 변수로 주입합니다(예: `OPENAI_API_KEY`).
- 팀에서 돌릴 땐 “샘플 예제”로 먼저 성공 경험을 만든 뒤, 설정 파일(`promptfooconfig.yaml`)로 확장하세요.

---

*다음 글에서는 핵심 개념과 아키텍처를 정리합니다.*

