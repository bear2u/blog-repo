---
layout: post
title: "airi 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-12
permalink: /airi-guide-01-intro/
author: moeru-ai
categories: [AI 에이전트, airi]
tags: [Trending, GitHub, airi, TypeScript, GitHub Trending]
original_url: "https://github.com/moeru-ai/airi"
excerpt: "Project AIRI(airi) 소개와 저장소 구조(모노레포)를 빠르게 훑습니다."
---

## airi란?

GitHub Trending(weekly) 기준으로 주목받는 **moeru-ai/airi**를 한국어로 정리합니다.

- **한 줄 요약(Trending 설명 기반)**: Self hosted, you-owned Grok Companion… (AI 동반자/가상 캐릭터를 목표로 하는 프로젝트)
- **언어(Trending 표시)**: TypeScript
- **이번 주 스타(Trending 표시)**: +8,962
- **원본**: https://github.com/moeru-ai/airi

---

## 저장소 구조(Repo Map)

airi는 루트에서부터 모노레포 형태를 갖습니다.

- `apps/`: 사용자-facing 앱(예: `apps/stage-web`, `apps/stage-tamagotchi`)과 서버 런타임 관련 앱이 함께 존재합니다.
- `packages/`: UI/오디오/플러그인 SDK/서버 공유 모듈 등 다수의 패키지로 쪼개져 있습니다.
- `crates/`: `tauri-plugin-*` 형태의 Rust 크레이트가 포함됩니다.
- `services/`, `plugins/`, `integrations/`: 외부 연동(봇/서비스/에디터/확장) 축이 별도 디렉토리로 분리돼 있습니다.
- 워크스페이스/빌드 도구: `pnpm-workspace.yaml`, `turbo.json`, 루트 `package.json`의 스크립트(예: `dev`, `build`, `test`, `lint`)가 엔트리 역할을 합니다.

---

## 이 가이드에서 다룰 것(예정)

- 설치/실행(웹/데스크톱/서버 런타임 관점)
- 모노레포 구조와 핵심 패키지들(앱/패키지/플러그인/서비스)
- 플러그인/연동 포인트(확장 지점)
- 운영/트러블슈팅 체크리스트

---

## 위키 링크

- `[[airi Guide - Index]]` → [가이드 목차](/blog-repo/airi-guide/)
