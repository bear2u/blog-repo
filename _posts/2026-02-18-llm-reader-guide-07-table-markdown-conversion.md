---
layout: post
title: "LLM Reader 가이드 (07) - 표 변환 로직: HTML Table을 Markdown으로 보존하기"
date: 2026-02-18
permalink: /llm-reader-guide-07-table-markdown-conversion/
author: m92vyas
categories: ['개발 도구', '웹 스크래핑']
tags: [HTML Table, Markdown, Data Extraction, llm-reader, Parsing]
original_url: "https://github.com/m92vyas/llm-reader"
excerpt: "표 데이터를 LLM이 읽기 쉬운 Markdown 표로 치환하는 로직과 현재 구현의 경계 조건을 정리합니다."
---

## 왜 표 변환이 중요한가

가격표/스펙표/비교표는 일반 텍스트로 평탄화되면 컬럼 정보가 쉽게 깨집니다.

LLM Reader는 테이블을 Markdown 형태로 바꿔 구조를 유지하려고 시도합니다.

---

## 변환 흐름

1. `soup.find_all('table')` 순회
2. 헤더(`th`) 기반으로 Markdown 헤더 생성
3. 각 `tr`를 `| col | col |` 포맷으로 변환
4. 원본 테이블을 치환 문자열로 대체

치환 시 내부 마커(`ttaabbllee ssttaarrtt`, `ttabbllee eenndd`)를 사용해 이후 텍스트 변환 과정에서 경계를 유지합니다.

---

## 예외 처리 특징

기본 변환 실패 시 fallback 경로를 한 번 더 시도합니다.

- 1차: 헤더/rowspan을 고려한 변환
- 2차: 단순 헤더+행 재구성

완벽한 HTML 표 재현보다는 "최소한 의미 있는 표 텍스트" 확보가 목표인 구현입니다.

---

## 운영 시 체크할 부분

- 병합 셀(`colspan`, 복잡한 `rowspan`)이 많은 표
- 표 안에 중첩 HTML/링크가 많은 경우
- 캡션/단위 정보 누락 여부

복잡한 재무/리서치 표를 다룬다면 후처리 정합성 검증 단계를 추가하는 편이 안전합니다.

다음 장에서는 래퍼 함수와 동시성 패턴을 다룹니다.
