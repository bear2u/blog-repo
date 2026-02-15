---
layout: post
title: "Cline 완벽 가이드 (10) - 트러블슈팅: 터미널/네트워크/히스토리 복구"
date: 2026-02-15
permalink: /cline-guide-10-troubleshooting/
author: Cline Bot Inc.
categories: [AI 코딩 에이전트, Cline]
tags: [Cline, Troubleshooting, Terminal, Proxy, Networking, Recovery]
original_url: "https://docs.cline.bot/troubleshooting/terminal-quick-fixes"
excerpt: "가장 흔한 문제인 터미널 통합 이슈, 프록시/방화벽 환경, 작업 히스토리 복구 방법을 정리합니다."
---

## 1) 터미널 통합이 불안하면: Background Execution Mode부터

문서의 “빠른 해결책”은 명확합니다.

- VS Code 터미널 통합이 자주 실패한다면
- **Background Execution Mode**로 전환해 우회하는 게 가장 빠른 해결책일 수 있습니다.

설정 경로(개념):

1. Cline Settings
2. Terminal Settings
3. Terminal Execution Mode → Background Exec

---

## 2) 자주 쓰는 터미널 퀵픽스(문서 요약)

- 기본 셸을 bash로 바꾸기
- shell integration timeout을 늘리기
- aggressive terminal reuse 비활성화
- WSL/PowerShell/Oh-My-Zsh 같은 환경별 설정 점검

이 이슈는 “Cline 자체”보다 IDE의 터미널/셸 통합 환경 영향이 커서,
재현 조건(OS/셸/IDE 버전)을 함께 확인하는 게 중요합니다.

---

## 3) 프록시/방화벽 환경: IDE는 IDE 설정, CLI는 env var

문서 기준으로 프록시 설정은 플랫폼마다 결이 다릅니다.

- VS Code 확장: VS Code의 프록시 설정을 사용
- JetBrains: IDE 프록시 설정을 사용
- Cline CLI: `http_proxy/https_proxy/no_proxy` 같은 환경변수 사용

CLI 예(개념):

```bash
export https_proxy=http://proxy.company.com:8080
export http_proxy=http://proxy.company.com:8080
export no_proxy=localhost,127.0.0.1
```

프록시에 커스텀 CA가 있다면 `NODE_EXTRA_CA_CERTS` 같은 설정도 고려해야 합니다(문서 참조).

---

## 4) 작업 히스토리가 사라진 것처럼 보일 때: 재구성 명령

문서에서 소개하는 대표 복구 방법은:

- 커맨드 팔레트에서 **"Cline: Reconstruct Task History"** 실행

Cline은 작업 데이터를 저장해두고, 히스토리 인덱스가 깨진 경우 인덱스를 다시 만드는 방식으로 복구합니다.

또한 백업 파일(`taskHistory.backup.*.json`)이 남아 있을 수 있어, 수동 복구 경로도 문서에 안내돼 있습니다.

---

## 정리

트러블슈팅의 1순위는 “Cline 기능 문제”보다 “통합 계층(IDE/셸/네트워크)”인 경우가 많습니다.

- 터미널: 통합 우회(Background Exec)로 먼저 해결
- 네트워크: IDE/CLI별 프록시 설정 분리
- 히스토리: 인덱스 재구성으로 복구

---

*이 시리즈는 여기서 마무리합니다. 다음으로는 Cline 문서의 각 기능(Worktrees, Hooks 샘플, MCP 서버 구성)을 실제 프로젝트에 맞게 묶어 ‘자주 쓰는 워크플로우 템플릿’으로 만드는 것도 추천합니다.*

