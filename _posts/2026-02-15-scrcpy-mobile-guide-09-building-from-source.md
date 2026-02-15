---
layout: post
title: "scrcpy-mobile(Scrcpy Remote) 가이드 (09) - 소스 빌드: libs, scrcpy-server, Xcode 실행"
date: 2026-02-15
permalink: /scrcpy-mobile-guide-09-building-from-source/
author: wsvn53
categories: [개발 도구, 모바일]
tags: [scrcpy-mobile, Build, Xcode, make, CMake, scrcpy-server, iOS]
original_url: "https://github.com/wsvn53/scrcpy-mobile"
excerpt: "README의 빌드 섹션을 기준으로, 의존성 빌드(make libs), scrcpy-server 빌드, Xcode 프로젝트 실행까지 최소 경로를 정리합니다."
---

## 0) 서브모듈부터 확인

레포에 `.gitmodules`가 있으므로, 소스 빌드라면 서브모듈 초기화가 필요할 수 있습니다.

```bash
git submodule update --init --recursive
```

---

## 1) 의존성 빌드: `make libs` (README)

README의 첫 단계:

```bash
make libs
```

이 단계는 포팅에 필요한 라이브러리/산출물을 준비하는 역할입니다.

---

## 2) scrcpy-server 빌드(README)

README는 다음 명령을 제시합니다.

```bash
make -C porting scrcpy-server
```

`porting/Makefile`을 보면 내부적으로:

- scrcpy 특정 버전을 내려받아
- server 쪽만 빌드하도록 구성하는 흐름이 들어있습니다.

---

## 3) Xcode로 실행(README)

마지막으로 README는 아래 프로젝트를 열어 빌드/런하라고 합니다.

- `scrcpy-ios/scrcpy-ios.xcodeproj`

실전 팁:

- iOS 실제 디바이스에서 실행하려면 서명/프로비저닝 설정이 필요할 수 있습니다.
- 네트워크 권한(로컬 네트워크 접근) 관련으로 iOS 권한 프롬프트가 뜰 수 있습니다.

다음 장에서는 실제 사용에서 가장 자주 만나는 연결/성능 문제를 트러블슈팅 형태로 정리합니다.

