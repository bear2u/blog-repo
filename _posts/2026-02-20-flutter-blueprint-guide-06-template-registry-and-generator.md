---
layout: post
title: "flutter_blueprint 가이드 (06) - 템플릿 레지스트리와 생성 파이프라인"
date: 2026-02-20
permalink: /flutter-blueprint-guide-06-template-registry-and-generator/
author: chirag640
categories: ['개발 도구', 'Flutter']
tags: [flutter_blueprint, TemplateRegistry, ProjectGenerator, Code Generation]
original_url: "https://github.com/chirag640/flutter_blueprint-Package"
excerpt: "TemplateRegistry 기반 템플릿 선택과 ProjectGenerator의 파일 생성 흐름을 단계별로 정리합니다."
---

## 템플릿 추상화 핵심

`lib/src/templates/core/template_interface.dart`에서 `ITemplateRenderer`와 `TemplateRegistry`를 정의합니다.

이 구조의 장점은 단순합니다.

- 템플릿 선택 로직을 한곳에 모음
- 플랫폼/상태관리 조합 확장 용이
- 렌더러 단위 테스트 가능

---

## 선택 규칙

`TemplateRegistry.selectFor`는 다음 순서로 렌더러를 찾습니다.

1. 멀티플랫폼이면 `universal`
2. 단일 플랫폼이면 `platform_state` 키 조합
3. 없으면 플랫폼 기본 렌더러

즉, 옵션 조합이 늘어도 switch-case를 비대하게 만들지 않아도 됩니다.

---

## `ProjectGenerator.generate` 단계

코드 기준 생성 순서는 아래와 같습니다.

1. 앱 이름/경로 검증
2. 렌더러 선택
3. `render(context)`로 파일 목록 확보
4. 병렬 파일 쓰기
5. CI 설정 파일 생성(옵션)
6. `blueprint.yaml` 저장
7. `flutter create` + `flutter pub get`

---

## 캐시 레이어

`CachedTemplateRenderer`는 같은 설정 해시에 대해 렌더 결과를 메모리 캐시해 반복 생성 비용을 줄입니다.

장기적으로는 대규모 템플릿 조합에서 생성 시간을 안정화하는 데 의미가 큽니다.

다음 글에서는 API 프리셋과 보안 인터셉터 설계를 다룹니다.

