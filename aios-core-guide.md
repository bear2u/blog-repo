---
layout: page
title: Synkra AIOS Core 가이드
permalink: /aios-core-guide/
icon: fas fa-robot
---

# Synkra AIOS Core 완벽 가이드

> **CLI First → Observability Second → UI Third**

**Synkra AIOS Core(aios-core)**는 개발(또는 다른 도메인) 작업을 위해 **에이전트 역할/태스크/워크플로우/템플릿**을 표준화하고, 한 번 만든 구조를 반복 가능한 프로세스로 실행할 수 있게 하는 **CLI 중심 프레임워크**입니다.

- 원문 저장소: https://github.com/SynkraAI/aios-core
- 문서(레포 내): `docs/`

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/aios-core-guide-01-intro/) | AIOS가 풀려는 문제, CLI First 철학 |
| 02 | [설치와 퀵스타트](/blog-repo/aios-core-guide-02-installation-quickstart/) | `npx @synkra/aios-core` 설치, `doctor`로 점검 |
| 03 | [CLI/에이전트 사용법](/blog-repo/aios-core-guide-03-cli-and-agents/) | `@dev`, `*help`, 커맨드/가시성/권한 |
| 04 | [태스크/워크플로우](/blog-repo/aios-core-guide-04-tasks-and-workflows/) | Task-First 아키텍처, 워크플로우 유형 |
| 05 | [플래닝 워크플로우](/blog-repo/aios-core-guide-05-planning-workflow/) | Brief→PRD→Architecture, Web UI에서 IDE로 전환 |
| 06 | [개발 사이클(스토리)](/blog-repo/aios-core-guide-06-story-development-cycle/) | SM/Dev/QA 루프, sharding과 품질 게이트 |
| 07 | [Squads/확장팩](/blog-repo/aios-core-guide-07-squads-expansion-packs/) | 에이전트 팀 번들, 도메인 확장 방식 |
| 08 | [MCP 글로벌 셋업](/blog-repo/aios-core-guide-08-mcp-global-setup/) | `~/.aios/` 기반 MCP 서버 공유 구성 |
| 09 | [LLM 라우팅/비용 최적화](/blog-repo/aios-core-guide-09-llm-routing/) | `claude-max`/`claude-free`, DeepSeek 백엔드 |
| 10 | [아키텍처/보안/운영](/blog-repo/aios-core-guide-10-ops-security-troubleshooting/) | 코어 구성, 하드닝 포인트, 트러블슈팅 |

---

## 주요 특징

- **CLI First**: 모든 기능은 CLI에서 먼저 완성되고, UI는 후순위로 관측/관리 목적에 가깝습니다.
- **역할 분리된 에이전트**: Dev/QA/PM/SM/Architect 등 역할별 에이전트를 분리해 책임을 명확히 합니다.
- **Task-First**: “무엇을 할지”를 태스크로 정의하고, 에이전트는 태스크를 실행하는 방식으로 표준화합니다.
- **워크플로우 오케스트레이션**: Greenfield/Brownfield 등 시나리오에 맞게 단계별 실행 흐름을 정의합니다.
- **반복 가능한 산출물**: 템플릿 기반으로 PRD/아키텍처/스토리 등을 일관된 포맷으로 생성합니다.

---

## 빠른 시작

```bash
# 새 프로젝트(그린필드)
npx @synkra/aios-core init my-project
cd my-project

# 기존 프로젝트(브라운필드)
cd existing-project
npx @synkra/aios-core install

# 설치 상태 점검
npx @synkra/aios-core doctor
```

IDE/Claude Code에서 에이전트를 활성화해 시작합니다.

```
@aios-master
*help
```

---

## 설치 후 디렉토리 감각 잡기

AIOS는 프로젝트 내부에 `.aios-core/`를 “두뇌(프레임워크 본체)”로 설치하고, IDE/CLI에서 에이전트/태스크/워크플로우를 사용합니다.

```
your-project/
├── .aios-core/
│   ├── core/               # 레지스트리/헬스체크/오케스트레이션 등
│   ├── development/        # agents/tasks/workflows
│   ├── product/            # 템플릿/체크리스트
│   └── infrastructure/     # 스크립트/통합/템플릿
├── .claude/                # Claude Code 설정/룰/커맨드(환경에 따라)
└── src/                    # 여러분의 코드
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| Node.js 18+ | CLI 실행 환경 |
| npm 9+ | 설치/배포 |
| TypeScript | 타입/구조화 |
| Jest/Mocha | 테스트 |

---

## 관련 링크

- GitHub: https://github.com/SynkraAI/aios-core
- Quick Start: `docs/installation/v2.1-quick-start.md`
- User Guide: `docs/guides/user-guide.md`
- Workflows: `docs/guides/workflows-guide.md`
- Discord(README에 기재): https://discord.gg/gk8jAdXWmj
