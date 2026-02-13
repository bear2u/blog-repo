---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (10) - 모더레이션과 신고"
date: 2026-02-13
permalink: /divine-mobile-guide-10-moderation-reporting/
author: divinevideo
categories: [모바일, Nostr]
tags: [Moderation, Reporting, NIP-51, NIP-32, NIP-56, diVine]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "신고(kind 1984), 뮤트 리스트(kind 10000), 라벨(kind 1985)을 어떻게 조합해 필터링을 구성하는지 문서/코드 기준으로 정리합니다."
---

## 이 레포의 모더레이션은 “하이브리드”다

`docs/MODERATION_STATUS.md`는 모더레이션 시스템을 크게 두 트랙으로 정리합니다.

1. **중앙화 트랙(백엔드)**: 신고 집계/임계치(auto-hide) 같은 “집계성 로직”
2. **탈중앙 트랙(Nostr)**: 뮤트 리스트/라벨 같은 “클라이언트 측 필터링 신호”

즉, “전부 Nostr만으로 끝내기”가 아니라, 실무적으로 필요한 집계는 백엔드에 두고,
클라이언트에서도 분산 신호를 조합해 사용자 경험을 구성하려는 접근입니다.

---

## 핵심 NIP/이벤트(kind)

문서 기준으로 자주 등장하는 세 가지입니다.

- **NIP-56 신고**: kind `1984` (사용자가 콘텐츠를 신고)
- **NIP-51 뮤트 리스트**: kind `10000` (개인/외부 뮤트 리스트 구독 및 필터링)
- **NIP-32 라벨**: kind `1985` (라벨러(labeler)들의 판단을 구독/집계)

---

## 코드에서의 시작 지점

문서에서 직접 파일을 가리키고 있어서, 따라가기 좋습니다.

### 1) 신고 생성

- `mobile/lib/services/content_reporting_service.dart`

여기는 “신고 이벤트(kind 1984)를 만들고 relay로 브로드캐스트”하는 쪽에 가깝습니다.

### 2) 뮤트 리스트 기반 필터링

- `mobile/lib/services/content_moderation_service.dart`

여기는 NIP-51 뮤트 리스트를 구독하고, 이벤트/저자/키워드/해시태그 등을 기준으로 필터링 결정을 내리는 역할입니다.

### 3) 라벨(레이블러) 구독

- `mobile/lib/services/moderation_label_service.dart`

여기는 NIP-32 라벨러들의 라벨 이벤트를 구독하고, 네임스페이스별 라벨을 집계하는 구조를 갖습니다.

---

## 문서가 강조하는 “조합 레이어”

`docs/MODERATION_STATUS.md`는 “각 신호를 하나로 합쳐 최종 결정을 내리는 coordinator(또는 feed service)가 필요하다”는 방향을 제시합니다.

요지는:

- 뮤트 리스트만으로는 부족할 수 있고
- 라벨과 백엔드 집계 상태까지 포함해
- “허용/경고/블러/숨김” 같은 최종 정책을 내리는 곳이 필요하다

블로그 독자 관점에서는, 지금 코드가 “어디까지 구현되어 있고” “어디가 TODO인지”를 구분해 읽는 게 중요합니다.

---

## 실전 디버깅 포인트

모더레이션은 종종 “데이터가 없어서 안 먹는지” vs “정책이 달라서 안 먹는지”가 섞여 보입니다.
따라서 다음 순서로 점검하면 빠릅니다.

1. 내가 기대하는 필터링은 어떤 신호(NIP-51/32/56)인가?
2. 해당 서비스가 실제로 구독/캐시를 하고 있는가?
3. 최종 렌더링(UI)에서 그 결정을 반영하는가?

---

*다음 글에서는 이 레포의 테스트 구조(유닛/위젯/통합)와, 문서에 정리된 테스트 인프라 복구 전략을 정리합니다.*

