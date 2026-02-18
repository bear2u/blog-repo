---
layout: post
title: "Summarize 가이드 (01) - 소개/핵심 포지션: CLI와 브라우저를 묶은 요약 워크플로우"
date: 2026-02-18
permalink: /summarize-guide-01-intro/
author: steipete
categories: ['개발 도구', 'CLI']
tags: [summarize, Chrome Side Panel, CLI, Daemon, AI Summary]
original_url: "https://github.com/steipete/summarize"
excerpt: "Summarize 프로젝트가 어떤 문제를 푸는지, CLI/확장/데몬 3계층 구조로 어떤 경험을 만드는지 정리합니다."
---

## Summarize가 지향하는 경험

README 기준 Summarize의 제품 구조는 단순 요약 스크립트가 아닙니다.

- 터미널에서 바로 쓰는 CLI
- Chrome Side Panel / Firefox Sidebar 확장
- 로컬에서 동작하는 daemon 서비스

즉 "요약 기능"을 실행 엔진(CLI)과 UI(확장)로 분리해 빠른 피드백 루프를 만든 구조입니다.

---

## 핵심 기능 축

프로젝트 소개를 보면 기능이 아래 4축으로 정리됩니다.

1. 다양한 입력: URL, 로컬 파일, PDF, 이미지, 오디오/비디오, YouTube, Podcast
2. 출력 모드: Markdown/Text, JSON 진단, extract-only
3. 모델 전략: 유료 API + 로컬/CLI + OpenRouter free preset
4. 확장 UX: 스트리밍 출력, chat, slides(스크린샷+OCR)

---

## 모노레포 구조

루트 트리 기준으로 관심 폴더는 다음입니다.

- `src/`: CLI + run 엔진 + daemon 로직
- `apps/chrome-extension/`: Chrome/Firefox 확장
- `packages/core/`: 라이브러리 API (`@steipete/summarize-core`)
- `docs/`: 기능별 문서
- `tests/`: 대규모 테스트 스위트

CLI 앱과 core 라이브러리를 분리해 의존성 경량화/재사용성을 확보한 형태입니다.

---

## 버전 문서 관점

저장소 기준으로 `package.json`은 `0.11.2`이고 README는 "0.11.0 preview" 문구가 포함되어 있습니다.

실무에서는 README의 릴리즈 노트성 설명보다 실제 패키지 버전/CHANGELOG를 함께 보는 습관이 안전합니다.

다음 장에서 설치와 실제 첫 실행 경로를 잡습니다.

