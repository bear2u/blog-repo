---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (03) - 레포 구조 한눈에 보기"
date: 2026-02-13
permalink: /divine-mobile-guide-03-repo-structure/
author: divinevideo
categories: [모바일, Flutter]
tags: [diVine, OpenVine, Flutter, Architecture, Docs]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "문서/앱/웹이 어떻게 분리되어 있고, 코드 탐색은 어디서 시작해야 하는지 정리합니다."
---

## top-level: 무엇이 어디에 있나

레포 루트에서 눈에 띄는 디렉토리들은 대략 아래처럼 역할이 갈립니다.

```text
divine-mobile/
  docs/         # 아키텍처/운영/테스트/트러블슈팅 문서
  mobile/       # Flutter 앱(핵심)
  website/      # 웹 관련(별도 구성)
  crawler/      # 보조 컴포넌트(데이터 수집 등)
  geo-blocker/  # 지역 차단 관련(별도 컴포넌트)
```

이 시리즈는 “앱 개발자” 관점이므로 `mobile/`을 중심으로 봅니다.

---

## docs/: 문서가 제공하는 것

`docs/`는 프로젝트의 설계 의도와 운영 경험을 많이 담고 있습니다.
예를 들면:

- `docs/NOSTR_EVENT_TYPES.md`: 앱에서 필요한 kind 목록과 기대 동작
- `docs/nip-vine.md`: kind `32222` 스펙 초안(주소가능 이벤트)
- `docs/nostr_pagination_docs.md`: 피드 페이징 동작 설명
- `docs/TROUBLESHOOTING_GUIDE.md`: 트러블슈팅 모음
- `docs/RELEASE_CHECKLIST.md`: 릴리스/배포 체크(일부는 SDK 관점)

처음엔 “이 프로젝트는 어떤 NIP(kind)를 무엇에 쓰는가”부터 읽는 것이 비용 대비 효과가 큽니다.

---

## mobile/: Flutter 앱 내부 구조

문서가 제시하는 큰 디렉토리 구분은 다음과 같습니다.

- `mobile/lib/screens/`: 화면 단위 UI
- `mobile/lib/widgets/`: 재사용 UI 컴포넌트
- `mobile/lib/services/`: 도메인/비즈니스 로직(업로드/피드/인증/모더레이션 등)
- `mobile/lib/providers/`: Riverpod provider(의존성 조립)
- `mobile/lib/models/`: 데이터 모델
- `mobile/lib/utils/`: 공용 유틸
- `mobile/lib/config/`: 설정 값

실제 프로젝트 규모가 커지면, “UI”보다 “서비스 레이어”를 먼저 잡는 게 이해 속도가 빠릅니다.

---

## providers/: 의존성 주입(조립) 지점

이 레포는 Riverpod을 사용합니다.
“어떤 서비스가 어떤 서비스를 참조하는지”를 한눈에 보려면, 다음 파일들이 큰 도움이 됩니다.

- `mobile/lib/providers/app_providers.dart`
- `mobile/lib/providers/nostr_client_provider.dart`

보통 “서비스 생성자 주입”이 provider에서 이뤄지므로, 시스템을 이해할 때는:

1. provider에서 객체가 어떻게 만들어지는지 본다
2. 해당 서비스 파일로 내려가서 실제 로직을 본다

의 순서가 효율적입니다.

---

## 추천 탐색 순서

처음 보는 사람 기준으로, 아래 순서로 읽으면 “대강 어떻게 돌아가는지”를 빠르게 잡을 수 있습니다.

1. `mobile/lib/main.dart`: 앱 시작점
2. `mobile/lib/providers/nostr_client_provider.dart`: Nostr 클라이언트 초기화/릴레이 구성
3. `docs/NOSTR_EVENT_TYPES.md`: 이벤트(kind) 요구사항
4. `mobile/lib/services/upload_manager.dart`: 업로드 라이프사이클
5. `mobile/lib/services/video_event_publisher.dart`: 업로드 이후 퍼블리시

---

## 팁: 문서와 코드는 “둘 다” 봐야 한다

이 레포는 문서가 풍부하지만, 일부는 현재 코드 상태와 어긋날 수 있습니다(예: 업로드 토큰 주입 경로).
가이드를 읽을 때도, 항상 다음 기준으로 확인하는 습관이 좋습니다.

- 문서가 말하는 흐름: “의도/설계”
- 코드가 하는 일: “현재 구현”

둘이 다르면 “문서가 틀렸다”가 아니라, 보통 “구현이 진화하는 중”이거나 “문서 갱신이 늦었다”는 신호입니다.

---

*다음 글에서는 Nostr 클라이언트가 Riverpod에서 어떻게 초기화되고, 릴레이가 어떤 방식으로 구성되는지 살펴봅니다.*

