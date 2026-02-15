---
layout: post
title: "scrcpy-mobile(Scrcpy Remote) 가이드 (02) - 설치와 준비: 앱 설치, 개발자 옵션, 네트워크"
date: 2026-02-15
permalink: /scrcpy-mobile-guide-02-install-and-prereqs/
author: wsvn53
categories: [개발 도구, 모바일]
tags: [scrcpy-mobile, iOS, Android, Setup, ADB, Wireless Debugging, Network]
original_url: "https://github.com/wsvn53/scrcpy-mobile"
excerpt: "App Store 설치부터 Android 측 준비(개발자 옵션, 무선 디버깅/adb tcpip), 같은 네트워크 조건까지 연결 전에 필요한 체크리스트를 정리합니다."
---

## 1) 앱 설치

README에 따르면 앱은 App Store에서 받을 수 있습니다.

- App Store: https://apps.apple.com/us/app/scrcpy-remote/id1629352527

설치 후 기본 모드는 **VNC**로 시작할 수 있고, 필요하면 **ADB Wi-Fi 모드**로 전환합니다(다음 장에서 설명).

---

## 2) Android 기기 준비: 개발자 옵션/무선 디버깅

ADB Wi-Fi(무선 디버깅)를 쓰려면 Android에서 아래를 준비해야 합니다.

- **개발자 옵션 활성화**
- **USB 디버깅**(초기 설정에 필요할 수 있음)
- **무선 디버깅(Wireless debugging)** 활성화(기기/버전에 따라 메뉴명이 조금 다름)

### `adb tcpip 5555`가 필요한 경우

README의 ADB 모드 안내에는 다음 명령이 등장합니다.

```bash
adb tcpip 5555
```

이 단계는 보통 “ADB를 TCP(5555)로 받도록” Android 쪽 상태를 바꾸는 역할을 합니다.  
환경에 따라 **초기 1회는 PC에서 `adb`를 통해 실행**해야 할 수 있습니다(USB 연결이나 동일 네트워크에서 ADB 연결 필요).

---

## 3) 네트워크 체크리스트

원격 제어 앱은 결국 네트워크 품질의 영향을 크게 받습니다.

- iPhone과 Android가 **동일 Wi-Fi**(또는 라우팅 가능한 동일 LAN)
- VPN/프록시가 끼어 있으면 연결/지연이 크게 나빠질 수 있으니 가능하면 끄고 테스트
- 공유기/방화벽이 **5555(ADB)** 또는 페어링 포트를 막으면 연결 실패 가능

---

## 4) “권한/보안” 관점의 기본 원칙

무선 디버깅은 편하지만, 원격 제어가 가능한 만큼 안전도 중요합니다.

- 공용 Wi-Fi에서는 무선 디버깅 사용을 피하기
- 연결이 끝나면 Android에서 무선 디버깅을 꺼 두기
- adbkey를 다른 기기와 공유하거나 외부로 내보낼 때는 “그 키가 곧 접근 권한”이라는 점을 인지하기

다음 장에서는 **ADB 모드/VNC 모드**를 어떻게 전환하고 무엇이 다른지부터 잡고 갑니다.

