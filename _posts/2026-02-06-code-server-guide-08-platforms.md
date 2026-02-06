---
layout: post
title: "code-server 완벽 가이드 (08) - 플랫폼별 가이드"
date: 2026-02-06
permalink: /code-server-guide-08-platforms/
author: Coder
categories: [웹 개발, 원격 개발]
tags: [code-server, iPad, Android, iOS, Termux, 모바일 개발]
original_url: "https://github.com/coder/code-server"
excerpt: "iPad, Android, iOS, Termux 등 다양한 플랫폼에서 code-server 사용하기."
---

## 플랫폼별 지원 현황

| 플랫폼 | 지원 | 방법 | 권장도 |
|--------|------|------|--------|
| **iPad** | ✅ | Safari/Chrome | ⭐⭐⭐⭐⭐ |
| **iPhone** | ✅ | Safari/Chrome | ⭐⭐⭐ |
| **Android 태블릿** | ✅ | Chrome | ⭐⭐⭐⭐ |
| **Android 폰** | ✅ | Chrome | ⭐⭐ |
| **Termux (Android)** | ✅ | 로컬 설치 | ⭐⭐⭐⭐ |

---

## iPad에서 code-server 사용하기

iPad는 code-server의 **최고의 클라이언트**입니다!

### 요구사항

- **OS**: iPadOS 14 이상
- **브라우저**: Safari 또는 Chrome
- **키보드**: 외장 키보드 (Magic Keyboard, Bluetooth 키보드 등)
- **네트워크**: 안정적인 Wi-Fi

### 접속 방법

#### 1. Safari/Chrome에서 접속

```
https://code.yourdomain.com
```

#### 2. 홈 화면에 추가 (PWA)

Safari:
1. `code.yourdomain.com` 접속
2. 공유 버튼 → "홈 화면에 추가"
3. 이름 설정: "Dev Environment"
4. 추가

이제 앱처럼 실행 가능!

### 키보드 단축키

대부분의 VS Code 단축키가 작동합니다:

| 단축키 | 기능 |
|--------|------|
| `Cmd+P` | 파일 찾기 |
| `Cmd+Shift+P` | 명령 팔레트 |
| `Cmd+B` | 사이드바 토글 |
| `Cmd+J` | 패널 토글 |
| `Cmd+\`` | 터미널 토글 |
| `Cmd+/` | 주석 토글 |

### 터미널 사용

iPad의 터미널은 완벽하게 작동합니다:

```bash
# Git 사용
git clone https://github.com/user/repo
cd repo
git add .
git commit -m "iPad에서 커밋!"
git push

# npm 사용
npm install
npm start

# 도커 (서버에서)
docker ps
docker-compose up
```

### 파일 업로드/다운로드

- **업로드**: 파일 탐색기 → 우클릭 → Upload
- **다운로드**: 파일 우클릭 → Download

### 알려진 제한사항

#### 자체 서명 인증서 미지원

**문제**: Safari는 자체 서명 인증서를 신뢰하지 않음

**해결책**:
- Let's Encrypt 사용 (Caddy/NGINX)
- SSH 포트 포워딩 (Blink Shell 앱 사용)

#### Cmd+W로 탭 닫힘

**문제**: `Cmd+W`가 에디터 탭 대신 브라우저 탭을 닫음

**해결책**:
- `Cmd+Option+W` 사용 (에디터 탭 닫기)
- 키보드 설정 재매핑

### 추천 앱

- **Blink Shell**: SSH 클라이언트 (포트 포워딩 지원)
- **Working Copy**: Git 클라이언트

---

## iPhone에서 code-server 사용하기

iPhone에서도 가능하지만, 화면이 작아 비추천.

### 사용 시나리오

- 긴급 버그 수정
- 짧은 코드 수정
- 로그 확인
- 서버 관리

### 팁

- 가로 모드 사용
- 글꼴 크기 조정: Settings → Text Editor → Font Size
- 파일 트리 숨기기 (`Cmd+B`)

---

## Android에서 code-server 사용하기

### Chrome 브라우저 사용

Android Chrome은 iPad Safari만큼 좋지는 않지만 사용 가능합니다.

#### 접속

```
https://code.yourdomain.com
```

#### 홈 화면에 추가

1. Chrome 메뉴 → "홈 화면에 추가"
2. 이름 설정
3. 추가

#### 키보드 단축키

Bluetooth 키보드 연결 시 대부분 작동.

### Samsung DeX 모드

**최고의 Android 경험!**

- DeX 모드 활성화 (삼성 태블릿/폰)
- 마우스 + 키보드 연결
- 거의 데스크톱 수준

---

## Termux에서 code-server 설치

**Termux**: Android에서 Linux 환경 실행

### 설치

```bash
# Termux 앱 설치 (F-Droid 권장)
# https://f-droid.org/en/packages/com.termux/

# 패키지 업데이트
pkg update && pkg upgrade

# TUR (Termux User Repository) 추가
pkg install tur-repo

# code-server 설치
pkg install code-server

# 실행
code-server

# 접속: http://localhost:8080
# 비밀번호: ~/.config/code-server/config.yaml
```

### 설정

```yaml
# ~/.config/code-server/config.yaml
bind-addr: 127.0.0.1:8080
auth: password
password: mypassword
cert: false
```

### 외부 접속

#### SSH 터널 (권장)

```bash
# Termux에서 SSH 서버 시작
pkg install openssh
sshd

# PC에서 포트 포워딩
ssh -p 8022 -L 8080:localhost:8080 user@android-ip
```

#### 직접 노출 (Wi-Fi에서만)

```yaml
# config.yaml
bind-addr: 0.0.0.0:8080
```

```bash
# Android IP 확인
ip addr | grep inet

# PC에서 접속
http://192.168.1.xxx:8080
```

### Termux 팁

#### 1. 백그라운드 실행

```bash
# Termux:Boot 앱 설치
# https://f-droid.org/en/packages/com.termux.boot/

# 시작 스크립트
mkdir -p ~/.termux/boot
nano ~/.termux/boot/start-code-server.sh
```

```bash
#!/data/data/com.termux/files/usr/bin/sh
termux-wake-lock
code-server
```

```bash
chmod +x ~/.termux/boot/start-code-server.sh
```

#### 2. 배터리 최적화 해제

Settings → Apps → Termux → Battery → Unrestricted

#### 3. 저장소 접근 권한

```bash
termux-setup-storage
```

이제 `~/storage/shared`에서 내부 저장소 접근 가능.

---

## iOS (iPhone)

iPad와 동일하나 화면이 작습니다.

### 사용 시나리오

- 긴급 수정
- 로그 확인
- 간단한 편집

### 권장사항

- Safari 사용 (최적화 더 좋음)
- 홈 화면에 추가 (PWA)
- 가로 모드

---

## Chrome OS

Chromebook에서도 완벽하게 작동합니다.

### Linux 앱으로 설치

```bash
# Linux (Beta) 활성화
# Settings → Advanced → Developers → Linux development environment

# 터미널에서
curl -fsSL https://code-server.dev/install.sh | sh
code-server
```

### 브라우저로 접속

```
http://localhost:8080
```

---

## 플랫폼별 베스트 프랙티스

### iPad

1. **Magic Keyboard 사용**: 최고의 경험
2. **Blink Shell**: SSH 포트 포워딩
3. **Split View**: 문서 + code-server
4. **Safari 사용**: Chrome보다 최적화 좋음

### Android

1. **DeX 모드 활용** (삼성)
2. **Bluetooth 키보드 + 마우스**
3. **Termux**: 로컬 개발 환경
4. **Chrome Flags 활성화**: `chrome://flags` → Desktop Mode

### Termux

1. **SSH 포트 포워딩 사용** (보안)
2. **Wake Lock 유지** (백그라운드 실행)
3. **배터리 최적화 해제**
4. **외부 키보드 연결**

---

## 모바일 개발 시나리오

### 시나리오 1: 카페에서 iPad로 개발

```
1. iPad + Magic Keyboard
2. Safari → code.yourdomain.com
3. Git pull → 코드 수정 → Git push
4. 서버에서 자동 배포
```

### 시나리오 2: 출퇴근 중 버그 수정

```
1. iPhone
2. GitHub 이슈 확인
3. code-server 접속
4. 긴급 수정 + 커밋
5. PR 생성 (GitHub 모바일 앱)
```

### 시나리오 3: Android 태블릿으로 풀스택 개발

```
1. Termux에 code-server 설치
2. Node.js, Python, PostgreSQL 설치
3. 로컬에서 전체 스택 개발
4. Git으로 서버에 배포
```

---

## 성능 최적화

### 1. 확장 최소화

모바일은 리소스가 제한적이므로 필수 확장만 설치.

### 2. 테마 간소화

밝은 테마가 어두운 테마보다 빠름.

### 3. 파일 Watcher 제한

```json
// settings.json
{
  "files.watcherExclude": {
    "**/node_modules/**": true,
    "**/.git/**": true
  }
}
```

### 4. 자동 저장 활성화

네트워크 끊김 대비:

```json
{
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000
}
```

---

## 문제 해결

### iPad: 키보드 단축키 작동 안 함

**해결**: Safari 설정 → 고급 → JavaScript → 허용

### Android: 화면 꺼짐

**해결**: 개발자 옵션 → "화면 끄지 않기" 활성화

### Termux: 비밀번호 입력 안됨

**해결**:
```bash
cat ~/.config/code-server/config.yaml
# 비밀번호 확인 후 복사/붙여넣기
```

---

*다음 글에서는 Docker, Kubernetes, 클라우드 배포 등 운영 관련 내용을 다룹니다.*
