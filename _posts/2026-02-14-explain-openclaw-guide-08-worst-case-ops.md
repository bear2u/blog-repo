---
layout: post
title: "Explain OpenClaw 완벽 가이드 (08) - 최악의 시나리오와 운영 최적화"
date: 2026-02-14
permalink: /explain-openclaw-guide-08-worst-case-ops/
author: centminmod
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, Prompt Injection, Supply Chain, Incident Response, Cost, Resource Usage]
original_url: "https://github.com/centminmod/explain-openclaw"
excerpt: "프롬프트 인젝션/공급망 리스크/인시던트 대응 플로우, 그리고 비용/리소스 최적화 포인트를 운영자 관점에서 정리합니다."
---
## 1) 프롬프트 인젝션은 "도구"와 결합될 때 현실이 된다

Explain OpenClaw의 prompt-injection 문서는 다양한 패턴을 나열하지만,
운영자에게 남는 결론은 단순합니다.

- 외부 입력(메시지/웹/파일)은 항상 비신뢰 데이터다.
- 도구 표면이 넓을수록, 인젝션 성공 시 피해가 커진다.
- DM/그룹 정책(pairing/allowlist/mention gating)이 1차 방어선이다.

---

## 2) 공급망: 스킬/플러그인은 사실상 코드 설치

Explain OpenClaw는 플러그인/스킬을 "in-process"로 다루는 관점이 강합니다.

실전 방어는 문서가 제안하는 것처럼:
- 신뢰할 수 있는 것만 설치
- 사회공학(문서의 "사전 명령" 실행 유도)을 특히 경계
- 감사 루틴과 권한 최소화를 함께

---

## 3) 인시던트 대응: 먼저 막고, 그 다음 조사

플레이북은 단계가 중요합니다.

1. Contain: Gateway 중지
2. Assess: 로그/페어링/세션/크리덴셜 변화 확인
3. Rotate: 토큰/키 전면 교체
4. Recover: `security audit --fix` + 재감사 + 모니터링

---

## 4) 비용/리소스: "항상 켜짐"은 자동으로 최적화되지 않는다

운영에서 자주 터지는 문제:
- 호출량이 생각보다 많아 비용이 급증
- 세션이 길어져 토큰이 증가
- 브라우저/이미지/로컬 임베딩이 CPU/RAM/디스크를 크게 사용

문서가 권장하는 실전 조치(요지):
- 모델 라우팅/대체 전략(OpenRouter 등) 검토
- 브라우저 도구는 필요할 때만
- 디스크에서 무한 성장 가능한 자원(JSONL 트랜스크립트, 브라우저 프로필)을 주기적으로 점검

---

## 추가 읽기(원문)

이 시리즈는 요약/가이드 역할이고, 상세는 원문이 가장 빠릅니다.

- Worst-case overview: 05-worst-case-security/README.md
- Prompt injection: 05-worst-case-security/prompt-injection-attacks.md
- Incident response: 05-worst-case-security/incident-response.md
- Resource usage: 06-optimizations/resource-usage.md
- Security audit reference: 08-security-analysis/security-audit-command-reference.md
