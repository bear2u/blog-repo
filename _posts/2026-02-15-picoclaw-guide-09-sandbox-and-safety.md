---
layout: post
title: "PicoClaw 가이드 (09) - 샌드박스/보안: restrict_to_workspace와 exec 가드레일"
date: 2026-02-15
permalink: /picoclaw-guide-09-sandbox-and-safety/
author: Sipeed
categories: [AI 에이전트, 개발 도구]
tags: ["PicoClaw", "Security", "Sandbox", "Workspace Restriction", "Guardrails"]
original_url: "https://github.com/sipeed/picoclaw#-security-sandbox"
excerpt: "PicoClaw는 기본적으로 워크스페이스 내부로 파일/명령 접근을 제한합니다. restrict_to_workspace 옵션과 위험 명령 차단 등 README의 가드레일 설계를 정리합니다."
---

## 기본값: 워크스페이스 내부로 제한

README는 기본 보안 모델로 `restrict_to_workspace: true`를 소개합니다.

의미:

- 에이전트가 파일을 읽거나 쓰거나
- 디렉토리를 나열하거나
- 명령을 실행하는

행동이 “설정된 워크스페이스 내부”로 제한됩니다.

이 모델의 장점은 단순합니다.

- 실수로 홈 디렉토리/시스템 파일을 건드릴 확률을 크게 줄임
- 운영 환경에서 위험도를 낮춤

---

## Protected tools와 exec 추가 보호

README는 `restrict_to_workspace: true`일 때 샌드박싱되는 도구들을 테이블로 나열합니다.

또한 `restrict_to_workspace`를 꺼도, `exec` 도구는:

- 대량 삭제
- 디스크/포맷/이미징
- 시스템 종료
- 포크 봄

같은 위험 패턴을 차단한다고 설명합니다.

---

## 제한 해제는 마지막 선택

README는 제한을 해제하는 방법을 두 가지로 안내합니다.

- config.json에서 `restrict_to_workspace: false`
- 환경 변수로 토글

하지만 이는 곧 “에이전트가 시스템 전체 경로에 접근”할 수 있다는 의미이므로,

- 로컬 실험
- 격리된 컨테이너
- 권한이 제한된 사용자 계정

같은 통제된 환경에서만 하는 편이 좋습니다.

다음 장에서는 스케줄(cron/heartbeat)과 운영 관점 트러블슈팅을 정리합니다.

