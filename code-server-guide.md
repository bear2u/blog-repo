---
layout: page
title: code-server 가이드
permalink: /code-server-guide/
icon: fas fa-terminal
---

# code-server 완벽 가이드

> **브라우저에서 실행하는 VS Code - 어디서나 일관된 개발 환경**

**code-server**는 원격 서버에서 VS Code를 실행하고 브라우저에서 접근할 수 있게 해주는 오픈소스 프로젝트입니다. iPad, 노트북, 데스크톱 어디서든 동일한 강력한 개발 환경을 제공합니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/code-server-guide-01-intro/) | code-server란? 주요 특징, 왜 사용하는가 |
| 02 | [설치 및 시작](/blog-repo/code-server-guide-02-installation/) | 다양한 설치 방법 (install.sh, npm, Docker, Helm) |
| 03 | [아키텍처 분석](/blog-repo/code-server-guide-03-architecture/) | 프로젝트 구조, VS Code 통합, 데이터 흐름 |
| 04 | [설정 및 구성](/blog-repo/code-server-guide-04-configuration/) | config.yaml, CLI 옵션, 환경 변수 |
| 05 | [보안 및 인증](/blog-repo/code-server-guide-05-security/) | HTTPS, 비밀번호 인증, OAuth, 외부 인증 |
| 06 | [네트워크 설정](/blog-repo/code-server-guide-06-network/) | SSH 포트 포워딩, Caddy, NGINX 리버스 프록시 |
| 07 | [프록시 시스템](/blog-repo/code-server-guide-07-proxy/) | 웹 서비스 프록시, React/Vue/Angular 설정 |
| 08 | [플랫폼별 가이드](/blog-repo/code-server-guide-08-platforms/) | iPad, Android, iOS, Termux 사용법 |
| 09 | [배포 및 운영](/blog-repo/code-server-guide-09-deployment/) | Docker, Kubernetes, 클라우드 배포 |
| 10 | [확장 및 커스터마이징](/blog-repo/code-server-guide-10-customization/) | 국제화, 테마, 베스트 프랙티스 |

---

## 주요 특징

- **어디서나 코딩** - 브라우저만 있으면 OK. 태블릿, 노트북, 데스크톱 모두 가능
- **클라우드 파워** - 서버의 강력한 CPU/메모리로 빠른 빌드, 테스트, 컴파일
- **배터리 절약** - 모든 무거운 작업은 서버에서 실행
- **VS Code 완벽 호환** - 확장 프로그램, 테마, 설정 모두 호환
- **iPad 지원** - Magic Keyboard와 함께 완벽한 개발 환경
- **보안** - HTTPS, 비밀번호 인증, OAuth 지원

---

## 빠른 시작

```bash
# 자동 설치 (Linux, macOS, FreeBSD)
curl -fsSL https://code-server.dev/install.sh | sh

# 시작
code-server

# 접속: http://127.0.0.1:8080
# 비밀번호: ~/.config/code-server/config.yaml
```

Docker:
```bash
docker run -it -p 127.0.0.1:8080:8080 \
  -v "$PWD:/home/coder/project" \
  codercom/code-server:latest
```

---

## 아키텍처 개요

```
┌──────────────────────────────────────────────────────────┐
│                     클라이언트 (브라우저)                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │         VS Code UI (HTML/JS/CSS)                 │   │
│  │  Extensions │ Editor │ Terminal │ Debug           │   │
│  └─────────────────────┬────────────────────────────┘   │
└────────────────────────┼─────────────────────────────────┘
                         │
                    WebSocket / HTTP
                         │
┌────────────────────────┼─────────────────────────────────┐
│                   code-server                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │          Node.js HTTP Server                     │   │
│  │  (Express.js + WebSocket)                        │   │
│  ├──────────────────────────────────────────────────┤   │
│  │          VS Code Backend                         │   │
│  │  (File System, Terminal, Extensions, LSP)        │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **기반** | VS Code (Microsoft) |
| **백엔드** | Node.js, TypeScript, Express.js |
| **프록시** | http-proxy |
| **인증** | 비밀번호, OAuth 지원 |
| **배포** | Docker, Helm, Systemd |
| **지원 플랫폼** | Linux, macOS, Windows, Docker, Kubernetes |

---

## VS Code vs code-server

| 기능 | VS Code (Desktop) | code-server |
|------|-------------------|-------------|
| **실행 환경** | 로컬 머신 | 원격 서버 |
| **접근 방법** | 데스크톱 앱 | 브라우저 |
| **확장 프로그램** | ✅ | ✅ |
| **터미널** | 로컬 | 서버 |
| **성능** | 로컬 CPU/RAM | 서버 CPU/RAM |
| **iPad 지원** | ❌ | ✅ |
| **모바일 지원** | ❌ | ✅ |

---

## 사용 사례

### 개인 개발자
- iPad로 카페에서 개발
- 여행 중에도 가벼운 디바이스로 작업
- 집/회사/카페 어디서든 동일한 환경

### 팀/기업
- 신규 팀원 온보딩 간소화 (URL만 전달)
- 일관된 개발 환경 제공
- 중앙 집중식 관리

### 교육 기관
- 학생용 표준 개발 환경 제공
- 인프라 비용 절감

---

## 관련 링크

- [GitHub 저장소](https://github.com/coder/code-server)
- [공식 문서](https://coder.com/docs/code-server)
- [Discord 커뮤니티](https://discord.gg/coder)
- [Slack 커뮤니티](https://coder.com/community)
- [GitHub Discussions](https://github.com/coder/code-server/discussions)

---

## 엔터프라이즈 솔루션

개인/소규모 팀용 **code-server**와 별개로, Coder 팀은 중대형 팀/기업을 위한 [Coder](https://github.com/coder/coder) 제품도 제공합니다:

- 멀티 테넌트 지원
- 중앙 집중식 관리
- 리소스 할당 관리
- 템플릿 기반 환경 프로비저닝
- 감사 로그

---

## 라이선스

MIT License
