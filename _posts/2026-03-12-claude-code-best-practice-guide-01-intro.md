---
layout: post
title: "claude-code-best-practice 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-12
permalink: /claude-code-best-practice-guide-01-intro/
author: shanraisshan
categories: [AI 코딩 에이전트, claude-code-best-practice]
tags: [Trending, GitHub, claude-code-best-practice, Claude Code, Best Practice, GitHub Trending]
original_url: "https://github.com/shanraisshan/claude-code-best-practice"
excerpt: "Claude Code를 더 ‘일관되게 잘 쓰는’ 방법을 커맨드/에이전트/스킬/워크플로우 단위로 정리한 레포를 소개합니다."
---

## claude-code-best-practice란?

GitHub Trending(weekly) 기준으로 주목받는 **shanraisshan/claude-code-best-practice**를 한국어로 정리합니다.

- **한 줄 요약(README 기반)**: Claude Code의 기능(Commands/Sub-Agents/Skills/Workflows/Hooks/MCP/Settings/Memory 등)을 “베스트 프랙티스 문서”와 “레포 내 구현 예시”로 함께 제공하는 지식베이스입니다.
- **언어(Trending 표시)**: HTML
- **이번 주 스타(Trending 표시)**: +4,400
- **원본**: https://github.com/shanraisshan/claude-code-best-practice

---

## 저장소 구조(README/디렉토리 기반)

- `.claude/`: Claude Code에서 바로 쓸 수 있는 리소스(예: `.claude/commands/`, `.claude/agents/`, `.claude/skills/`, `.claude/hooks/`)
- `best-practice/`: 베스트 프랙티스 문서 모음
- `implementation/`: 베스트 프랙티스를 “이 레포에서 어떻게 구현했는지”에 대한 예시
- `orchestration-workflow/`: Command → Agent → Skill 오케스트레이션 흐름 문서
- `tips/`, `reports/`, `development-workflows/`: 운영 팁/리포트/워크플로우 레시피

---

## 이 가이드에서 다룰 것(예정)

- 이 레포를 “템플릿”으로 복사/참조하는 방법(내 프로젝트에 적용)
- `.claude/` 리소스(커맨드/에이전트/스킬/훅) 설계 패턴
- MCP/세팅/메모리 정리 전략(충돌 방지, 재현성 확보)

---

## 위키 링크

- `[[claude-code-best-practice Guide - Index]]` → [가이드 목차](/blog-repo/claude-code-best-practice-guide/)

