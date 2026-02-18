---
layout: post
title: "LLM Reader 가이드 (10) - 제한사항/트러블슈팅: 운영 안정화를 위한 체크리스트"
date: 2026-02-18
permalink: /llm-reader-guide-10-limitations-and-troubleshooting/
author: m92vyas
categories: ['개발 도구', '웹 스크래핑']
tags: [Troubleshooting, Reliability, llm-reader, Playwright, Production]
original_url: "https://github.com/m92vyas/llm-reader"
excerpt: "llm-reader를 운영 파이프라인에 넣을 때 자주 맞닥뜨리는 실패 유형과 개선 포인트를 점검합니다."
---

## 현재 구현의 제한사항

- 브라우저 인스턴스 재사용/풀링 로직이 기본 제공되지 않음
- 복잡한 lazy-load 이미지(`data-src`, `srcset`) 처리 미흡
- 고난도 테이블(`colspan`/중첩 구조)에서 정밀 변환 한계
- 예외 처리 시 빈 문자열 반환 중심이라 원인 추적 정보가 적음

---

## 장애 대응 체크리스트

1. HTML 수집 실패율(HTTP/브라우저 에러) 모니터링
2. 처리 후 텍스트 길이 급감/급증 알람
3. 추출 결과 스키마 검증(JSON schema)
4. 실패 URL 재처리 큐와 재시도 정책 분리

---

## 코드 레벨 개선 아이디어

- `goto` 타임아웃/리트라이 옵션 외부 노출
- 표 변환 유닛 테스트 보강(실제 상거래 샘플)
- 로깅 구조화(JSON logs)
- 링크/이미지 처리 전략을 preset 클래스로 분리

---

## 운영 아키텍처 권장안

```text
Fetcher Layer (Playwright/API/Proxy)
 -> Normalizer Layer (llm-reader)
 -> Validator Layer (schema checks)
 -> LLM Extractor
 -> Post Processor (dedupe/enrichment)
```

llm-reader는 이 중 Normalizer Layer에 집중해 두는 것이 가장 유지보수성이 높습니다.

---

10개 챕터를 모두 마쳤습니다. 이후에는 실제 대상 사이트(쇼핑/문서/뉴스)별 프리셋을 만들어 정확도를 고도화하는 단계로 넘어가면 됩니다.
