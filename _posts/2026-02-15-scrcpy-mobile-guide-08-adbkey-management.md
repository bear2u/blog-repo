---
layout: post
title: "scrcpy-mobile(Scrcpy Remote) 가이드 (08) - adbkey 관리: 생성/가져오기/내보내기와 보안"
date: 2026-02-15
permalink: /scrcpy-mobile-guide-08-adbkey-management/
author: wsvn53
categories: [개발 도구, 모바일]
tags: [scrcpy-mobile, adbkey, Security, Wireless Debugging, ADB]
original_url: "https://github.com/wsvn53/scrcpy-mobile"
excerpt: "scrcpy-mobile은 adbkey를 생성/수정/가져오기/내보내기 할 수 있습니다. adbkey의 의미와 팀/기기간 이동 시 주의할 점(권한, 회전, 폐기)을 정리합니다."
---

## adbkey는 “접근 권한”이다

ADB 연결은 일반적으로 “키 기반 승인”을 전제로 합니다.  
따라서 adbkey는 단순 설정 파일이 아니라, 사실상 **원격 디버깅 권한**을 나타냅니다.

README에는 adbkey 관리 기능이 명시돼 있습니다.

- generate, modify, import, export

---

## 언제 adbkey를 관리해야 하나

- 새 iPhone/새 설치로 인해 승인(Authorize)을 다시 받고 싶지 않을 때
- 여러 환경(테스트폰/사내 단말)에서 동일한 “연결 주체”로 보이게 하고 싶을 때
- 키를 분실했거나 노출 가능성이 생겨 **키 회전**이 필요할 때

---

## 보안 관점 체크리스트

- adbkey를 외부로 내보내기(export) 했다면: 보관 위치(클라우드/메신저)를 점검
- 테스트가 끝났다면: Android의 무선 디버깅 페어링/승인 목록을 정리(가능한 경우)
- 공용 네트워크에서는: 무선 디버깅을 켜 둔 채로 방치하지 않기

다음 장에서는 개발자 관점에서 **소스 빌드** 흐름(`make libs`, `scrcpy-server`, Xcode 실행)을 정리합니다.

