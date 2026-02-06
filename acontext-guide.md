---
layout: page
title: Acontext 가이드
permalink: /acontext-guide/
icon: fas fa-database
---

# Acontext 완벽 가이드

> **클라우드 네이티브 AI 에이전트를 위한 Context Data Platform**

Acontext는 에이전트 컨텍스트를 저장/편집/관찰하기 위한 플랫폼입니다. 이 가이드는 공식 문서(`docs/docs.json`)의 정보 구조를 기준으로, 실무에서 바로 적용할 수 있게 챕터별로 정리했습니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 구조](/acontext-guide-01-intro/) | Acontext의 문제 정의, 핵심 구성요소, 아키텍처 |
| 02 | [빠른 시작](/acontext-guide-02-quickstart/) | Hosted/Self-hosted 시작, SDK 초기 연결 |
| 03 | [세션 & 메시지 저장](/acontext-guide-03-session-messages/) | 멀티 프로바이더/멀티모달/메타데이터/세션 설정 |
| 04 | [스토리지 계층](/acontext-guide-04-storage/) | Disk, Skill, Sandbox, 사용자 단위 격리 |
| 05 | [에이전트 툴](/acontext-guide-05-agent-tools/) | Sandbox/Disk/Skill Tool 호출 패턴 |
| 06 | [컨텍스트 엔지니어링](/acontext-guide-06-context-engineering/) | Session Summary, Context Editing, Cache 안정화 |
| 07 | [관측성(Observability)](/acontext-guide-07-observability/) | Task 추출, Buffer, Dashboard, Tracing |
| 08 | [설정 & 런타임](/acontext-guide-08-settings-runtime/) | 로컬 배포, 코어 환경변수, 런타임 튜닝 |
| 09 | [프레임워크 연동](/acontext-guide-09-integrations/) | Agno, OpenAI SDK, OpenAI Agents, Vercel AI SDK |
| 10 | [API 운영 체크리스트](/acontext-guide-10-api-ops/) | REST API 사용, 인증, 운영 시 주의사항 |

---

## 핵심 요약

- **Context Storage**: 세션 메시지, 파일(Disk), 스킬, 샌드박스까지 한곳에 통합
- **Context Engineering**: 저장 원본을 보존하면서 조회 시점에 컨텍스트 편집
- **Observability**: 에이전트 대화에서 작업(Task)을 추출해 대시보드로 추적
- **Integrations**: OpenAI/Agno/Vercel AI SDK 등 기존 스택에 쉽게 연결

---

## 공식 링크

- [GitHub 저장소](https://github.com/memodb-io/Acontext)
- [공식 문서](https://docs.acontext.io)
- [대시보드](https://dash.acontext.io)
