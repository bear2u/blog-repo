---
layout: post
title: "Ghostty 완벽 가이드 (01) - 소개 및 개요"
date: 2026-02-07
permalink: /ghostty-guide-01-intro/
author: Mitchell Hashimoto
categories: [개발 도구, CLI]
tags: [Ghostty, Terminal, Zig, GPU Rendering, Terminal Emulator]
original_url: "https://github.com/ghostty-org/ghostty"
excerpt: "빠르고 네이티브하며 기능이 풍부한 차세대 터미널 에뮬레이터"
---

## Ghostty란?

**Ghostty**는 빠르고, 네이티브하며, 기능이 풍부한 차세대 터미널 에뮬레이터입니다. Zig 언어로 작성되었으며, GPU 가속 렌더링(Metal/OpenGL)을 통해 최고의 성능을 제공합니다.

---

## 핵심 특징

### 1. **세 가지 강점의 균형**

대부분의 터미널 에뮬레이터는 다음 중 하나를 선택해야 합니다:
- **속도** (Alacritty)
- **기능** (iTerm2, Kitty)
- **네이티브 UI** (Terminal.app)

Ghostty는 **세 가지 모두**를 제공합니다.

### 2. **압도적인 성능**

```
평문 텍스트 읽기 벤치마크 (상대 속도):
────────────────────────────────
Ghostty     ████████████ 1.0x (기준)
Alacritty   █████████████ 1.1x
Kitty       ████████████████ 1.5x
iTerm2      ████████████████████████ 2.5x
Terminal.app ████████████████ 1.8x
```

- **Metal 렌더러** (macOS): 리거처 지원 60fps 유지
- **OpenGL 렌더러** (Linux): 고성능 GPU 가속
- **전용 IO 스레드**: 낮은 지터, 빠른 응답

### 3. **표준 준수**

- **xterm 호환성**: 포괄적인 xterm 감사 완료
- **ECMA-48 표준** 구현
- 기존 셸 및 CLI 도구와 완벽 호환

### 4. **네이티브 플랫폼 경험**

#### macOS
- **SwiftUI** 기반 앱
- 네이티브 메뉴바, 윈도우 관리
- Mac 설정 패널
- **Metal 렌더러** + CoreText 폰트

#### Linux
- **GTK** 기반 앱
- GNOME/KDE 통합
- **OpenGL 렌더러**

---

## Ghostty vs 다른 터미널

| 터미널 | 속도 | 기능 | 네이티브 | 렌더러 | 언어 |
|--------|------|------|---------|---------|------|
| **Ghostty** | ⚡️⚡️⚡️ | ✅✅✅ | ✅✅✅ | Metal/OpenGL | Zig |
| Alacritty | ⚡️⚡️⚡️ | ⚠️ | ❌ | OpenGL | Rust |
| Kitty | ⚡️⚡️ | ✅✅✅ | ❌ | OpenGL | C/Python |
| iTerm2 | ⚡️ | ✅✅✅ | ✅✅ | CPU/Metal* | Objective-C |
| Terminal.app | ⚡️⚡️ | ⚠️ | ✅✅✅ | CPU | Objective-C |

*iTerm2는 리거처 활성화 시 CPU 렌더러로 전환

---

## 주요 기능

### 윈도우 관리
- ✅ 멀티 윈도우
- ✅ 탭
- ✅ 수평/수직 스플릿

### 렌더링
- ✅ GPU 가속 (Metal/OpenGL)
- ✅ 리거처 지원 (Metal에서도 60fps)
- ✅ 이모지 및 유니코드
- ✅ True Color (24비트)

### 터미널 기능
- ✅ xterm 호환성
- ✅ 256색 및 True Color
- ✅ iTerm2 이미지 프로토콜
- ✅ Kitty 이미지 프로토콜
- ✅ OSC 52 (클립보드)

### 커스터마이징
- ✅ 폰트 패밀리 및 크기
- ✅ 색상 테마
- ✅ 키바인딩
- ✅ 설정 파일 (TOML/JSON)

---

## 로드맵

| # | 항목 | 상태 |
|---|------|------|
| 1 | 표준 준수 터미널 에뮬레이션 | ✅ 완료 |
| 2 | 경쟁력 있는 성능 | ✅ 완료 |
| 3 | 기본 커스터마이징 | ✅ 완료 |
| 4 | 윈도우/탭/스플릿 | ✅ 완료 |
| 5 | 네이티브 플랫폼 경험 | ⚠️ 진행 중 |
| 6 | libghostty (임베딩 가능) | ⚠️ 진행 중 |
| 7 | Windows 지원 | ❌ 계획됨 |
| N | 고급 기능 | ❌ 미정 |

---

## libghostty - 임베딩 가능한 터미널

Ghostty는 독립 실행형 앱뿐만 아니라 **C 호환 라이브러리**로도 제공됩니다.

```
libghostty
├── libghostty-vt    # VT 시퀀스 파싱, 터미널 상태 관리
├── libghostty-ui    # UI 렌더링 (개발 중)
└── libghostty       # 완전한 임베딩 가능 터미널
```

**사용 사례:**
- 텍스트 에디터에 터미널 임베딩 (VSCode, Neovim)
- IDE 통합 터미널
- 게임 콘솔
- 웹 브라우저 터미널

---

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| **언어** | Zig (0.13.0+) |
| **macOS UI** | SwiftUI |
| **Linux UI** | GTK 4 |
| **렌더러 (macOS)** | Metal + CoreText |
| **렌더러 (Linux)** | OpenGL + Fontconfig |
| **빌드** | Zig Build System |
| **테스트** | Zig Test + XCTest |

---

## 왜 Zig인가?

- **성능**: C/C++과 동등한 속도
- **안전성**: 컴파일 타임 안전성
- **간결함**: 숨겨진 제어 흐름 없음
- **C 상호 운용성**: 쉬운 C 라이브러리 통합
- **크로스 컴파일**: 단일 명령으로 모든 플랫폼 빌드

---

## 시작하기

```bash
# macOS
brew install ghostty

# Linux (Flatpak)
flatpak install flathub org.ghostty.ghostty

# 소스 빌드
git clone https://github.com/ghostty-org/ghostty
cd ghostty
zig build -Doptimize=ReleaseFast
```

---

## 커뮤니티 및 리소스

- **공식 웹사이트**: https://ghostty.org
- **문서**: https://ghostty.org/docs
- **GitHub**: https://github.com/ghostty-org/ghostty
- **다운로드**: https://ghostty.org/download

---

*다음 글에서는 Ghostty 설치 방법과 첫 설정을 살펴봅니다.*
