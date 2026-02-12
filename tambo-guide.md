---
layout: page
title: Tambo 가이드
permalink: /tambo-guide/
icon: fas fa-robot
---

# Tambo 완벽 가이드

> **React 앱에 \"UI로 말하는\" 에이전트를 추가하는 오픈소스 생성형 UI 툴킷**

**Tambo**는 React 컴포넌트를 \"에이전트가 선택해서 렌더링\"할 수 있게 만드는 생성형 UI 툴킷입니다.
컴포넌트의 프롭스를 Zod 스키마로 등록하면, LLM이 상황에 맞는 컴포넌트를 선택하고 프롭스를 스트리밍으로 생성해 UI를 구성합니다.

- React SDK: `@tambo-ai/react`
- 백엔드: Tambo Cloud(호스팅) 또는 Docker 기반 자체 호스팅
- 통합: MCP(Model Context Protocol), 로컬 도구 실행, 추가 컨텍스트/인증/추천 프롬프트

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/tambo-guide-01-intro/) | Tambo란? 어떤 문제를 푸는가, 핵심 특징 |
| 02 | [설치 및 빠른 시작](/blog-repo/tambo-guide-02-quickstart/) | `npm create tambo-app`, 개발 서버 실행 |
| 03 | [작동 원리](/blog-repo/tambo-guide-03-how-it-works/) | 컴포넌트 등록, 선택, 프롭스 스트리밍 흐름 |
| 04 | [생성형 컴포넌트](/blog-repo/tambo-guide-04-generative-components/) | 1회성 렌더링 컴포넌트 패턴 |
| 05 | [상호작용 컴포넌트](/blog-repo/tambo-guide-05-interactable-components/) | 지속/업데이트되는 상태형 UI 패턴 |
| 06 | [프로바이더와 훅, 인증/컨텍스트](/blog-repo/tambo-guide-06-provider-hooks-auth/) | `TamboProvider`, 주요 훅, userKey/userToken |
| 07 | [로컬 도구와 MCP 통합](/blog-repo/tambo-guide-07-tools-mcp/) | 브라우저 도구 호출, MCP 서버 연결 |
| 08 | [모노레포 아키텍처](/blog-repo/tambo-guide-08-architecture/) | Turborepo 구성, 패키지/앱 구조 |
| 09 | [자체 호스팅 및 운영 가이드](/blog-repo/tambo-guide-09-self-host-ops/) | Docker 배포, 환경변수, 운영 체크 |

---

## 빠른 시작

```bash
npm create tambo-app my-tambo-app
cd my-tambo-app
npm run dev
```

---

## 관련 링크

- GitHub: https://github.com/tambo-ai/tambo
- 문서: https://docs.tambo.co
