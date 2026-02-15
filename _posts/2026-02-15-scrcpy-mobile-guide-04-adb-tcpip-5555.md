---
layout: post
title: "scrcpy-mobile(Scrcpy Remote) 가이드 (04) - ADB Wi-Fi 시작: tcpip 5555, 인증, 연결 흐름"
date: 2026-02-15
permalink: /scrcpy-mobile-guide-04-adb-tcpip-5555/
author: wsvn53
categories: [개발 도구, 모바일]
tags: [scrcpy-mobile, ADB, tcpip, port-5555, Wireless Debugging]
original_url: "https://github.com/wsvn53/scrcpy-mobile"
excerpt: "ADB 모드에서 가장 중요한 준비는 Android가 tcpip 5555로 ADB를 받도록 하는 것입니다. adb tcpip, 권한 승인, 연결 지속을 어떻게 이해할지 정리합니다."
---

## 1) `adb tcpip 5555`의 의미

README의 ADB 모드 섹션은 Android에서 `adb tcpip 5555`를 활성화하라고 안내합니다.

```bash
adb tcpip 5555
```

이 설정이 켜져 있어야 iOS 앱이 네트워크를 통해 ADB로 붙을 수 있습니다.

현실적으로는 다음 중 하나가 필요할 수 있습니다.

- PC에서 `adb` 실행(초기 1회)
- Android 버전/제조사에 따라 “무선 디버깅” UI에서 동일 효과를 내는 경우

---

## 2) Android에서 승인(Authorize) 흐름

README에는 “Android 디바이스에서 authorized 되면 scrcpy가 계속 연결을 시도한다”는 문장이 있습니다.

즉, ADB는 보통 아래 흐름입니다.

1. 클라이언트(iOS 앱)가 ADB로 연결 요청
2. Android가 **디버깅 키**(adbkey) 기반으로 승인/거부를 요구
3. 승인되면 이후 세션에서 연결이 원활해짐

다음 장의 “페어링 코드(Android 10+)”는 이 과정을 더 안전하고 관리하기 쉽게 해줍니다.

---

## 3) 연결 안정성을 높이는 팁

ADB Wi-Fi는 네트워크 품질에 민감합니다.

- iPhone/Android를 가능한 **같은 AP(2.4/5GHz 혼용 주의)**에 붙이기
- 공유기에서 **클라이언트 격리(AP isolation)** 기능이 켜져 있으면 끄기
- 영상이 끊기면 해상도/비트레이트를 낮추기(뒤 장의 URL Scheme에서 예시)

다음 장에서는 Android 10+에서 지원하는 **페어링 코드 기반 연결**을 자세히 다룹니다.
