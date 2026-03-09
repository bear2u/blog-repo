---
layout: post
title: "flutter_blueprint 가이드 (04) - 생성 결과 구조 해부"
date: 2026-02-20
permalink: /flutter-blueprint-guide-04-generated-project-structure/
author: chirag640
categories: ['개발 도구', 'Flutter']
tags: [flutter_blueprint, Project Structure, Clean Architecture]
original_url: "https://github.com/chirag640/flutter_blueprint-Package"
excerpt: "생성된 Flutter 프로젝트에서 core/features/test 계층을 어떻게 이해하고 확장할지 정리합니다."
---

## 기본 뼈대

README의 예시 구조는 다음처럼 `core`와 `features`를 분리합니다.

```text
lib/
  app/
  core/
    api/
    config/
    errors/
    routing/
    storage/
    theme/
    utils/
  features/
test/
```

---

## `core`는 공통 기반 계층

`core`는 기능별 화면보다 먼저 안정화해야 하는 인프라 성격의 코드가 위치합니다.

- API 클라이언트/인터셉터
- 환경/설정 로딩
- 공통 에러/실패 모델
- 재사용 위젯/유틸리티

여기가 흔들리면 기능 모듈이 연쇄적으로 깨지므로, 초기 규칙을 명확히 두는 것이 좋습니다.

---

## `features`는 도메인 단위

기능은 "화면"이 아니라 도메인(예: auth, profile, home) 중심으로 나눠야 확장 시 충돌이 줄어듭니다.

- presentation
- domain
- data

이 분리는 테스트 범위를 명확히 나누는 데도 유리합니다.

---

## 구조를 오래 유지하는 법

1. `core`에 기능 의존 코드 넣지 않기
2. feature 간 직접 참조보다 계약(인터페이스)로 연결
3. 공통 로직이 생기면 즉시 `core`로 승격

다음 글에서는 이 생성물을 만드는 CLI v2 내부 구조를 코드 레벨에서 보겠습니다.

