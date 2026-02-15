---
layout: page
title: Cline 가이드
permalink: /cline-guide/
icon: fas fa-robot
---

# Cline 완벽 가이드

> **IDE 안에서 파일 편집, 터미널 실행, 브라우저 자동화, MCP 연동까지 “승인 기반”으로 굴리는 오픈소스 코딩 에이전트**

**Cline**은 VS Code/JetBrains 등 IDE 내부에서 동작하는 오픈소스 AI 코딩 에이전트입니다. 단순 자동완성이나 챗봇이 아니라, 작업을 계획(Plan)하고 실행(Act)하면서 파일 변경과 터미널 명령을 단계별로 제안하고 사용자가 승인하는 “human-in-the-loop” 워크플로우에 초점을 둡니다.

- 원문 저장소: https://github.com/cline/cline
- 공식 문서: https://docs.cline.bot/
- VS Code Marketplace: https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/cline-guide-01-intro/) | Cline이 하는 일, 철학(투명성/모델 선택/Zero Trust) |
| 02 | [설치와 시작](/blog-repo/cline-guide-02-installation-and-first-run/) | 설치, 모델 선택, 첫 작업 루프 |
| 03 | [Plan & Act](/blog-repo/cline-guide-03-plan-and-act/) | 계획과 실행 모드, 언제 어떻게 전환할까 |
| 04 | [컨텍스트 관리](/blog-repo/cline-guide-04-context-management/) | 컨텍스트 윈도우, Focus Chain/Auto Compact, @멘션 전략 |
| 05 | [모델/프로바이더](/blog-repo/cline-guide-05-models-and-providers/) | Cline Provider vs API 키, 로컬 모델, 비용/컨텍스트 감각 |
| 06 | [MCP 연동](/blog-repo/cline-guide-06-mcp-integration/) | MCP 개념, 서버 설치/구성, 보안 고려 |
| 07 | [Rules & Workflows](/blog-repo/cline-guide-07-rules-and-workflows/) | `.clinerules`, `AGENTS.md`, 조건부 룰, 워크플로우 |
| 08 | [Hooks & 승인 정책](/blog-repo/cline-guide-08-hooks-and-approvals/) | Hooks로 가드레일 만들기, Auto-approve 설계 |
| 09 | [서브에이전트/스킬/CLI](/blog-repo/cline-guide-09-subagents-skills-cli/) | 병렬 탐색(subagents), 스킬, Cline CLI 개요 |
| 10 | [트러블슈팅](/blog-repo/cline-guide-10-troubleshooting/) | 터미널 통합 이슈, 프록시/네트워크, 히스토리 복구 |

---

## 주요 특징(요약)

- **Plan & Act 모드**로 “생각”과 “실행”을 분리해 품질과 안정성을 올립니다.
- **승인 기반 도구 실행**으로 파일 변경/터미널 명령의 가시성을 확보합니다.
- **모델/프로바이더 선택 자유도**가 높고, 로컬 모델도 연결할 수 있습니다.
- **MCP**로 외부 도구/데이터 소스를 연결해 에이전트의 능력을 확장합니다.
- **Rules/Hooks**로 조직/프로젝트의 정책을 일관되게 적용할 수 있습니다.

