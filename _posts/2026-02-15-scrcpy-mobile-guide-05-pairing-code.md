---
layout: post
title: "scrcpy-mobile(Scrcpy Remote) 가이드 (05) - 페어링 코드: Android 10+ 무선 디버깅을 안전하게 붙이기"
date: 2026-02-15
permalink: /scrcpy-mobile-guide-05-pairing-code/
author: wsvn53
categories: [개발 도구, 모바일]
tags: [scrcpy-mobile, Android 10, Pairing Code, Wireless Debugging, ADB]
original_url: "https://github.com/wsvn53/scrcpy-mobile"
excerpt: "Android 10+는 '페어링 코드'로 무선 디버깅을 페어링할 수 있습니다. scrcpy-mobile에서의 메뉴 경로와 Android 설정 경로를 따라 연결하는 방법을 정리합니다."
---

## 언제 필요한가

README는 “페어링 코드는 Android 10+에서만 동작”한다고 명시합니다.

- Android 10+이고
- 무선 디버깅을 좀 더 명시적으로(코드 기반) 페어링하고 싶다면

이 방식이 가장 깔끔합니다.

---

## iOS 앱에서 페어링 시작(README)

README 기준 iOS 앱에서의 흐름:

1. Scrcpy Remote 메인 화면 좌측 상단의 `...` 메뉴
2. **Pair With Pairing Code** 선택

---

## Android에서 페어링 코드 화면 열기(README 경로)

README가 안내하는 Android 경로는 다음과 같습니다.

1. `Settings` → `System` → `Developer Options`
2. `Enable Wireless Debugging`
3. `Pair device with pairing code`

기기/안드로이드 버전에 따라 메뉴 명칭이 조금 다를 수 있지만, 핵심은 “Wireless debugging의 pairing code 화면”입니다.

---

## 페어링이 되면 달라지는 점

페어링이 성립하면, 이후 ADB Wi-Fi 연결(원격 제어)이 더 매끄러워지고:

- 승인(Authorize) 팝업 빈도가 줄어들고
- adbkey 관리도 일관되게 가져갈 수 있습니다.

다음 장에서는 **URL Scheme**으로 접속 설정을 통째로 “링크”로 만들고, iOS 단축어로 빠르게 연결하는 방법을 정리합니다.

