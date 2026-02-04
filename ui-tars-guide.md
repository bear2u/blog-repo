---
layout: page
title: "UI-TARS 완벽 가이드"
permalink: /ui-tars-guide/
---

# UI-TARS 완벽 가이드

ByteDance의 멀티모달 AI 에이전트 스택 **UI-TARS**를 상세하게 분석한 가이드입니다.

---

## 목차

### 1. [소개 및 개요](/ui-tars-guide-01-intro/)
- UI-TARS란?
- Agent TARS vs UI-TARS Desktop
- 프로젝트 구조
- 기술 스택

### 2. [전체 아키텍처](/ui-tars-guide-02-architecture/)
- 계층화된 모듈식 아키텍처
- 핵심 모듈 관계
- 데이터 흐름
- 이벤트 스트림 시스템

### 3. [Desktop 앱 분석](/ui-tars-guide-03-desktop-app/)
- Electron 메인/렌더러 프로세스
- IPC 라우팅 시스템
- Zustand 상태 관리
- 윈도우 및 권한 관리

### 4. [Agent TARS Core](/ui-tars-guide-04-agent-tars/)
- CLI 구현
- AgentTARS 클래스
- 이벤트 스트림
- 브라우저 제어 전략

### 5. [GUI Agent SDK](/ui-tars-guide-05-gui-agent/)
- Action Parser
- GUIAgent 클래스
- 시스템 프롬프트
- SoM 시각화

### 6. [Operators](/ui-tars-guide-06-operators/)
- Browser Operator (Puppeteer)
- NutJS Operator (데스크톱)
- ADB Operator (모바일)
- Remote Operators

### 7. [Tarko 프레임워크](/ui-tars-guide-07-tarko/)
- Agent 기본 클래스
- EventStream
- LLM Client
- MCP Agent

### 8. [MCP 인프라](/ui-tars-guide-08-mcp/)
- MCP 서버 아키텍처
- Browser MCP Server
- Filesystem MCP Server
- Commands MCP Server

### 9. [Context Engineering](/ui-tars-guide-09-context/)
- 컨텍스트 전략
- 메모리 시스템
- 토큰 관리
- 프롬프트 최적화

### 10. [활용 가이드 및 결론](/ui-tars-guide-10-conclusion/)
- 실제 활용 예제
- 커스텀 개발
- 보안 고려사항
- 결론

---

## 빠른 시작

```bash
# Agent TARS 실행
npx @agent-tars/cli@latest

# 또는 전역 설치
npm install @agent-tars/cli@latest -g
agent-tars --provider openai --model gpt-4o
```

---

## 주요 특징

| 특징 | 설명 |
|------|------|
| **Vision-Language Model** | 스크린샷 기반 UI 인식 |
| **MCP 통합** | 표준화된 도구 프로토콜 |
| **멀티 플랫폼** | 브라우저, 데스크톱, 모바일 |
| **이벤트 스트림** | 실시간 상태 추적 |

---

## 관련 링크

- [GitHub Repository](https://github.com/bytedance/UI-TARS-desktop)
- [MCP 공식 문서](https://modelcontextprotocol.io)

---

<p style="text-align: center; color: #666; margin-top: 2rem;">
  원본: ByteDance UI-TARS Desktop
</p>
