---
layout: post
title: "seomachine 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-12
permalink: /seomachine-guide-01-intro/
author: TheCraigHewitt
categories: [AI 코딩 에이전트, seomachine]
tags: [Trending, GitHub, seomachine, Claude Code, SEO, GitHub Trending]
original_url: "https://github.com/TheCraigHewitt/seomachine"
excerpt: "Claude Code 기반 SEO 콘텐츠 생산 워크스페이스(SEO Machine)의 구조와 사용 흐름을 요약합니다."
---

## seomachine란?

GitHub Trending(weekly) 기준으로 주목받는 **TheCraigHewitt/seomachine**를 한국어로 정리합니다.

- **한 줄 요약(README 기반)**: Claude Code 워크스페이스로서, 리서치→작성→최적화까지 “SEO 최적화 장문 블로그 콘텐츠” 생산 흐름을 프리셋(커맨드/에이전트/컨텍스트)로 제공합니다.
- **언어(Trending 표시)**: Python
- **이번 주 스타(Trending 표시)**: +1,507
- **원본**: https://github.com/TheCraigHewitt/seomachine

---

## 핵심 구성요소(README/구조 기반)

- 커스텀 슬래시 커맨드: `/research`, `/write`, `/rewrite`, `/analyze-existing`, `/optimize` 등(README의 “Custom Commands”)
- 에이전트/스킬 리소스: `.claude/commands/`, `.claude/agents/`, `.claude/skills/`
- 컨텍스트 템플릿: `context/` (브랜드 보이스, 스타일 가이드, 키워드, 내부 링크 맵 등)
- 산출물 디렉토리: `research/`, `drafts/`, `rewrites/`, `published/`
- 데이터 소스/분석 모듈: `data_sources/` (예: `data_sources/requirements.txt`)

---

## 빠른 시작(README 기반)

1) 레포 클론 후(README “Installation”)  
2) 분석 모듈 파이썬 의존성 설치: `pip install -r data_sources/requirements.txt`  
3) Claude Code에서 열기: `claude-code .`  
4) `context/` 아래 템플릿을 “우리 회사 정보”로 채운 뒤 커맨드로 실행

---

## 위키 링크

- `[[seomachine Guide - Index]]` → [가이드 목차](/blog-repo/seomachine-guide/)

