---
layout: post
title: "flutter_blueprint 가이드 (07) - API 설정과 보안 기본값"
date: 2026-02-20
permalink: /flutter-blueprint-guide-07-api-config-and-security/
author: chirag640
categories: ['개발 도구', 'Flutter']
tags: [flutter_blueprint, ApiConfig, Security, Dio, OWASP]
original_url: "https://github.com/chirag640/flutter_blueprint-Package"
excerpt: "ApiConfig 프리셋과 보안 인터셉터 기본값을 중심으로 생성 코드가 어떤 안전장치를 제공하는지 설명합니다."
---

## `ApiConfig`로 백엔드 차이를 흡수

`lib/src/config/api_config.dart`는 응답 파싱과 토큰 처리를 설정 객체로 분리합니다.

- 성공 판정 키/값
- 데이터 경로(`data`, `obj`, nested path)
- 에러 메시지/코드 경로
- 토큰 추출/전송 방식

프리셋으로 `modern`, `legacyDotNet`, `laravel`, `django`를 제공합니다.

---

## 생성 API 클라이언트의 보안 기본값

템플릿 공통 코드에는 보안 헤더와 인터셉터가 들어갑니다.

- `X-Content-Type-Options`, `X-Frame-Options`
- HSTS, no-cache 계열 헤더
- 인증 토큰 삽입/리프레시 처리
- 레이트 리밋, 에러 sanitize, SSRF 방어

기본값을 제공한다는 점이 핵심입니다. 팀은 "보안 적용"이 아니라 "보안 조정"부터 시작할 수 있습니다.

---

## 실무 적용 시 확인할 것

1. 서버 규약과 `ApiConfig` 경로 일치 여부
2. 토큰 리프레시 정책(만료/재시도/로그아웃) 정합성
3. 로깅에서 민감정보 마스킹 여부

생성 코드는 출발점이지 완성본이 아니므로, 프로젝트 정책에 맞게 빠르게 재구성해야 합니다.

다음 글에서는 품질 관련 커맨드(`analyze/optimize/refactor`)를 봅니다.

