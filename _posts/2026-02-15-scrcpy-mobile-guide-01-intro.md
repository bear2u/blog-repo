---
layout: post
title: "scrcpy-mobile(Scrcpy Remote) 가이드 (01) - 소개: 모바일에서 Android 원격 제어하기"
date: 2026-02-15
permalink: /scrcpy-mobile-guide-01-intro/
author: wsvn53
categories: [개발 도구, 모바일]
tags: [scrcpy-mobile, scrcpy, Remote Control, iOS, Android, ADB, Wireless Debugging]
original_url: "https://github.com/wsvn53/scrcpy-mobile"
excerpt: "scrcpy-mobile은 scrcpy를 모바일로 포팅해 iPhone에서 Android 기기를 원격 제어할 수 있게 합니다. 지원 범위, 동작 원리, 어떤 상황에 유용한지 정리합니다."
---

## scrcpy-mobile이 해결하는 문제

`scrcpy`는 원래 “PC에서 Android를 제어”하는 데 널리 쓰이는 도구입니다.  
**scrcpy-mobile**은 이 아이디어를 모바일로 가져와, **iPhone(또는 모바일 기기)에서 Android 화면을 보고 조작**할 수 있게 합니다.

예를 들면:

- 다른 방에 있는 Android 기기를 손쉽게 조작(테스트/설정)
- 개발자 옵션/무선 디버깅을 활용해 케이블 없이 제어
- 불안정한 네트워크에서 제스처 경험을 최적화한 형태로 원격 조작

---

## 현재 지원 범위(README 기준)

README 기준으로 현재는:

- **iOS에서 Android 기기 제어**가 중심
- **Android에서 Android 제어**는 “추후 지원 예정”

따라서 이 시리즈도 iOS 기준으로 설명합니다(개념/네트워크는 Android에서도 유사할 수 있습니다).

---

## 핵심 기능 요약

README에 언급된 핵심 기능은 다음과 같습니다.

- **ADB over Wi-Fi**로 scrcpy 연결 지원
- **하드웨어 디코딩**(전력/CPU 절감)
- **제스처 UX 최적화**
- `scrcpy2://` **URL Scheme** 지원(빠른 접속)
- Android 10+ **페어링 코드**로 무선 디버깅 페어링
- Android 내비게이션 버튼(Back/Home/App switch)
- iOS 키보드 토글 및 키 전송
- **클립보드 동기화**
- **adbkey 관리**(생성/수정/가져오기/내보내기)
- Android 11+ **오디오 포워딩**

---

## 동작 원리(큰 그림)

정확한 내부 구현은 케이스별로 다르지만, 사용 관점에서는 이렇게 이해하면 됩니다.

1. Android 기기에서 **무선 디버깅/ADB 연결을 허용**
2. iOS 앱이 Android의 ADB 엔드포인트로 연결(필요 시 페어링)
3. scrcpy 프로토콜로 화면 스트림을 받고 입력을 전달

이때 “연결”은 크게 두 갈래로 나뉩니다.

- **ADB 모드**: 성능/경험이 더 중요한 쪽(무선 디버깅/5555)
- **VNC 모드**: noVNC 기반(README에 성능 한계 안내가 있음)

다음 장부터는 설치와 준비(특히 Android 개발자 옵션/무선 디버깅)를 먼저 정리합니다.

