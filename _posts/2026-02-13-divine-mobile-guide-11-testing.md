---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (11) - 테스트 전략"
date: 2026-02-13
permalink: /divine-mobile-guide-11-testing/
author: divinevideo
categories: [모바일, Flutter]
tags: [Testing, Flutter Test, Integration Test, Riverpod, diVine]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "테스트 디렉토리 구조와 실행 루프, 그리고 문서에 정리된 테스트 인프라 복구 전략의 핵심을 요약합니다."
---

## 테스트는 어디에 있나

기본적으로 Flutter 앱 테스트는 `mobile/` 아래에서 돌아갑니다.

- `mobile/test/`: 유닛/위젯 테스트가 주로 위치
- `mobile/integration_test/`: 통합 테스트

또한 `docs/`에는 테스트 관련 문서가 꽤 많습니다.

- `docs/TEST_FIX_STRATEGY.md`
- `docs/*TEST*.md`

---

## 기본 실행 루프(최소)

```bash
cd mobile
flutter test
flutter analyze
```

이 두 개가 “최소한의 신호”입니다.

---

## Riverpod 테스트 관점

이 레포는 provider가 많기 때문에, 테스트에서 중요한 것은 “의존성 오버라이드”입니다.
문서(`docs/TEST_FIX_STRATEGY.md`)도 다음 방향을 제시합니다.

- 테스트 빌더/헬퍼를 만들어 모델 생성 비용을 줄이고
- 최소 구현(테스트 더블)을 통해 서비스 의존성을 단순화하고
- mockito 기반으로 mocks를 재생성(build_runner)

즉, “모델/서비스의 생성자 시그니처가 바뀌면 테스트가 연쇄적으로 깨질 수 있다”는 전제를 깔고,
테스트 인프라를 다시 정비하는 전략이 필요합니다.

---

## 통합 테스트(개념)

통합 테스트는 UI 레벨에서 실제 흐름을 검증할 때 유용합니다.
예를 들어:

- 업로드 플로우(녹화 -> 업로드 -> 퍼블리시)
- 피드 페이지네이션(스크롤 -> 더 로드 -> 중복 방지)

같은 “연결된 플로우”는 단위 테스트보다 통합 테스트가 더 가치가 큽니다.

---

## 실패할 때 보는 순서(실무 팁)

테스트가 깨졌을 때는 보통 아래 중 하나입니다.

1. 모델 생성자 변경(필수 필드 추가/타입 변경)
2. provider/서비스 의존성 변경(주입 파라미터 추가)
3. mock 파일/헬퍼 삭제 또는 경로 변경

그래서 “하나씩 고치기”보다, 문서가 말하는 것처럼 “핵심 모델 -> 서비스 초기화 패턴 -> mock 재생성” 순으로 정리하는 편이 빠를 때가 많습니다.

---

*다음 글에서는 실제 빌드/릴리스/배포 스크립트(`mobile/*.sh`, `codemagic.yaml`)를 중심으로 배포 흐름을 정리합니다.*

