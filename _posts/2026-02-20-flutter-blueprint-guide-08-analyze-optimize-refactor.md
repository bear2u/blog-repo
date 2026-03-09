---
layout: post
title: "flutter_blueprint 가이드 (08) - analyze/optimize/refactor 실무 패턴"
date: 2026-02-20
permalink: /flutter-blueprint-guide-08-analyze-optimize-refactor/
author: chirag640
categories: ['개발 도구', 'Flutter']
tags: [flutter_blueprint, Analyze, Optimize, Refactor, QA]
original_url: "https://github.com/chirag640/flutter_blueprint-Package"
excerpt: "코드 품질 점검, 번들 최적화, 자동 리팩토링 명령을 실제 운영 루틴에서 어떻게 조합할지 정리합니다."
---

## `analyze`: 품질 진단

```bash
flutter_blueprint analyze .
flutter_blueprint analyze . --all --strict --accessibility
```

코드 품질, 번들, 성능 진단을 목적별로 분리 실행할 수 있습니다.

---

## `optimize`: 절감 가능성 탐색

```bash
flutter_blueprint optimize --all --dry-run
flutter_blueprint optimize --assets
```

`--dry-run`으로 먼저 잠재 절감량을 보고, 실제 제거/최적화는 팀 리뷰 후 반영하는 흐름이 안전합니다.

---

## `refactor`: 명시적 자동 변경

```bash
flutter_blueprint refactor --add-caching --dry-run
flutter_blueprint refactor --migrate-to-riverpod
```

리팩토링 타입을 명시적으로 고르게 한 점이 좋습니다. 다만 자동 변경 도구는 항상 사전 백업과 테스트가 필수입니다.

---

## CI에 붙이는 최소 루틴

1. PR 생성 시 `analyze --all`
2. 주간 점검으로 `optimize --dry-run`
3. 구조 전환 시 `refactor --dry-run` 결과 리뷰 후 적용

다음 글에서는 `share` 명령으로 팀 공통 설정을 다룹니다.

