---
layout: post
title: "Cline 완벽 가이드 (06) - MCP 연동: 외부 도구를 에이전트의 ‘툴킷’으로 붙이기"
date: 2026-02-15
permalink: /cline-guide-06-mcp-integration/
author: Cline Bot Inc.
categories: [AI 코딩 에이전트, Cline]
tags: [Cline, MCP, Tools, Resources, Security, Integrations]
original_url: "https://docs.cline.bot/mcp/mcp-overview"
excerpt: "MCP의 개념(도구/리소스/호스트), Cline에서의 설치/구성 흐름, 보안 고려사항을 정리합니다."
---

## MCP를 한 문장으로

문서는 MCP(Model Context Protocol)를 “AI 앱을 위한 USB-C 포트”에 비유합니다.

핵심은:

- LLM이 외부 시스템과 상호작용할 때
- 도구(tools), 리소스(resources), 프롬프트(prompts)를
- 표준 인터페이스로 연결하는 프로토콜이라는 점입니다.

---

## MCP 서버가 제공하는 것

문서 관점에서 MCP 서버는 “LLM이 호출할 수 있는 API 집합”입니다.

대표 구성:

- **Tools**: LLM이 실행할 수 있는 함수(예: GitHub 이슈 조회, DB 쿼리 등)
- **Resources**: 읽기 중심의 데이터 접근(파일 경로, 쿼리 결과 등)
- **Security**: 민감 정보/자격증명 격리와 승인 기반 실행

---

## Cline에서 MCP를 어떻게 쓰나

문서가 강조하는 흐름은 “Cline이 MCP 서버를 직접 만들어주거나, GitHub에서 가져와 구성할 수 있다”는 점입니다.

실전 사용 패턴:

1. 커뮤니티 서버를 설치해 연결한다
2. 필요한 기능이 없으면 Cline에게 “나만의 MCP 서버”를 만들게 한다
3. 여러 서버를 조합해 워크플로우를 구성한다

예:

- GitHub + Notion을 묶어 “PR 요약을 문서로 자동 저장”
- Jira + PagerDuty를 묶어 “장애 이슈 맥락을 자동 수집”

---

## 보안 관점 체크리스트(최소)

MCP는 강력한 만큼, “무엇을 연결하느냐”가 곧 권한입니다.

- 서버가 어떤 자격증명(토큰/키)을 갖는지
- 어떤 도구가 쓰기(변경)를 수행하는지
- Auto Approve/YOLO 모드와 함께 쓸 때 위험도가 어떻게 바뀌는지

를 먼저 점검하는 게 안전합니다.

---

*다음 글에서는 Cline Rules와 Workflows로 “프로젝트 정책”과 “반복 작업 자동화”를 어떻게 구조화하는지 정리합니다.*

