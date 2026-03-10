---
layout: post
title: "ai-hedge-fund 완벽 가이드 (05) - 운영/확장/베스트 프랙티스"
date: 2026-03-10
permalink: /ai-hedge-fund-guide-05-best-practices/
author: virattt
categories: [개발 도구, ai-hedge-fund]
tags: [Trending, GitHub, ai-hedge-fund]
original_url: "https://github.com/virattt/ai-hedge-fund"
excerpt: "키 관리/재현성/에이전트 확장 관점 체크리스트"
---

## 체크리스트

- `.env`는 로컬 전용(커밋 금지). CI를 쓰면 Secret으로 분리
- 동일 조건 재현을 위해: 티커, 기간, 모델/온도 등 옵션을 로그로 남기기
- 에이전트 확장 시: `src/agents/`에 새 에이전트를 추가하고, 포트폴리오 매니저의 의사결정 흐름과 연결

---

## 함정

- API Rate Limit/에러로 결과가 흔들릴 수 있습니다: 재시도/캐시 전략 필요
- 교육용 PoC이므로 “투자 의사결정 시스템”으로 바로 쓰기에는 위험합니다(README의 Disclaimer 참고)

---

## TODO / 확인 필요

- Web App 경로(`app/`)는 별도 런북으로 정리(로컬/도커 포함)
- 테스트/평가 기준을 문서화해 에이전트 결과의 품질을 비교 가능하게 만들기

