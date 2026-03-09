---
layout: post
title: "flutter_blueprint 가이드 (02) - 설치와 첫 실행"
date: 2026-02-20
permalink: /flutter-blueprint-guide-02-installation-and-first-run/
author: chirag640
categories: ['개발 도구', 'Flutter']
tags: [flutter_blueprint, Dart, Installation, Quick Start]
original_url: "https://github.com/chirag640/flutter_blueprint-Package"
excerpt: "설치부터 첫 프로젝트 생성, 생성 결과 검증까지 가장 짧은 시작 경로를 정리합니다."
---

## 설치

README 기준 설치는 Dart global activate 방식입니다.

```bash
dart pub global activate flutter_blueprint
```

설치 후 `flutter_blueprint` 실행 파일이 PATH에서 호출 가능해야 합니다.

---

## 첫 프로젝트 생성

가장 빠른 경로는 인터랙티브 모드입니다.

```bash
flutter_blueprint init
```

명령 인자를 함께 주면 비대화형으로도 생성할 수 있습니다.

```bash
flutter_blueprint init my_app --state provider --api --tests
```

---

## 생성 직후 확인 항목

생성 후 최소한 다음을 확인하면 초기 장애를 줄일 수 있습니다.

1. `my_app/blueprint.yaml` 생성 여부
2. `flutter pub get` 성공 여부
3. `flutter run` 혹은 `flutter test` 기본 동작 여부

CLI 내부 `ProjectGenerator`는 생성 후 `flutter create`와 의존성 설치 단계까지 이어서 수행하도록 설계되어 있습니다.

---

## 설치 단계에서 자주 하는 실수

- Dart/Flutter 버전 미스매치
- PATH 미설정으로 실행 파일 미인식
- 권한 문제로 global activate 실패

실패 시에는 `dart pub global activate flutter_blueprint`를 먼저 재실행하고 환경 변수를 점검하는 것이 가장 빠릅니다.

다음 글에서는 `init` 명령의 옵션 설계를 실제 플래그 중심으로 해설합니다.

