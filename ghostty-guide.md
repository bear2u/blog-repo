---
layout: page
title: Ghostty 가이드
permalink: /ghostty-guide/
icon: fas fa-terminal
---

# Ghostty 완벽 가이드

> **빠르고 네이티브하며 기능이 풍부한 차세대 터미널 에뮬레이터**

**Ghostty**는 Mitchell Hashimoto(HashiCorp 창업자)가 Zig 언어로 작성한 차세대 터미널 에뮬레이터입니다. GPU 가속 렌더링(Metal/OpenGL)으로 최고의 성능을 제공하며, 플랫폼 네이티브 UI와 풍부한 기능을 모두 갖추고 있습니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요]({{ site.baseurl }}/ghostty-guide-01-intro/) | Ghostty란? 특징, 다른 터미널과 비교 |
| 02 | [설치 및 빠른 시작]({{ site.baseurl }}/ghostty-guide-02-installation/) | 설치 방법, 첫 실행, 기본 설정 |
| 03 | [아키텍처 분석]({{ site.baseurl }}/ghostty-guide-03-architecture/) | Zig 기반, 멀티 렌더러, 플랫폼별 구현 |
| 04 | [설정 및 커스터마이징]({{ site.baseurl }}/ghostty-guide-04-configuration/) | 설정 파일, 테마, 폰트, 키바인딩 |
| 05 | [렌더링 시스템]({{ site.baseurl }}/ghostty-guide-05-rendering/) | Metal/OpenGL 렌더러, 성능 최적화 |
| 06 | [터미널 에뮬레이션]({{ site.baseurl }}/ghostty-guide-06-terminal-emulation/) | VT 시퀀스, xterm 호환성 |
| 07 | [플랫폼 통합]({{ site.baseurl }}/ghostty-guide-07-platform-integration/) | macOS/Linux 네이티브 기능 |
| 08 | [libghostty]({{ site.baseurl }}/ghostty-guide-08-libghostty/) | 임베딩 가능한 라이브러리 |
| 09 | [고급 기능]({{ site.baseurl }}/ghostty-guide-09-advanced-features/) | 멀티윈도우, 탭, 스플릿 |
| 10 | [실전 활용 및 팁]({{ site.baseurl }}/ghostty-guide-10-best-practices/) | 워크플로우, 베스트 프랙티스 |

---

## 주요 특징

- **압도적인 성능**: Metal/OpenGL GPU 가속, 60fps 유지
- **네이티브 UI**: SwiftUI (macOS), GTK (Linux)
- **표준 준수**: xterm 호환, ECMA-48 표준
- **리거처 지원**: Metal 렌더러에서 60fps 유지
- **임베딩 가능**: libghostty C API

---

## Ghostty vs 다른 터미널

| 터미널 | 속도 | 기능 | 네이티브 | 렌더러 |
|--------|------|------|---------|---------|
| **Ghostty** | ⚡️⚡️⚡️ | ✅✅✅ | ✅✅✅ | Metal/OpenGL |
| Alacritty | ⚡️⚡️⚡️ | ⚠️ | ❌ | OpenGL |
| Kitty | ⚡️⚡️ | ✅✅✅ | ❌ | OpenGL |
| iTerm2 | ⚡️ | ✅✅✅ | ✅✅ | CPU/Metal* |
| Terminal.app | ⚡️⚡️ | ⚠️ | ✅✅✅ | CPU |

---

## 빠른 시작

### 설치

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

### 기본 설정

```toml
# ~/.config/ghostty/config

font-family = "JetBrains Mono"
font-size = 14
theme = "catppuccin-mocha"
window-padding-x = 10
window-padding-y = 10
```

---

## 아키텍처 개요

```
┌──────────────────────────────────────────────────┐
│            Ghostty Application                    │
├──────────────────────────────────────────────────┤
│  macOS (SwiftUI)  │  Linux (GTK)  │  libghostty  │
└─────────┬──────────┴───────┬───────┴──────┬──────┘
          │                  │              │
┌─────────┴──────────────────┴──────────────┴──────┐
│          Ghostty Core (Zig)                      │
│  • Terminal Emulator                             │
│  • VT Parser                                     │
│  • Screen State                                  │
└─────────┬──────────────────────────────┬─────────┘
          │                              │
┌─────────┴────────┐          ┌──────────┴─────────┐
│  Metal Renderer  │          │  OpenGL Renderer   │
│     (macOS)      │          │      (Linux)       │
└──────────────────┘          └────────────────────┘
```

---

## 성능 벤치마크

### 평문 텍스트 읽기 (상대 속도)

```
Ghostty     ████████████ 1.0x (기준)
Alacritty   █████████████ 1.1x
Terminal.app ████████████████ 1.8x
Kitty       ████████████████ 1.5x
iTerm2      ████████████████████████ 2.5x
```

### 리거처 렌더링 (fps)

```
Ghostty (Metal)   60 fps ████████████
Alacritty         60 fps ████████████
Kitty             60 fps ████████████
iTerm2 (CPU)*     30 fps ██████
```

*iTerm2는 리거처 활성화 시 CPU 렌더러로 전환

---

## 핵심 기능

### 윈도우 관리
- ✅ 멀티 윈도우
- ✅ 탭
- ✅ 수평/수직 스플릿
- ✅ 빠른 터미널 (Quake 스타일)

### 렌더링
- ✅ Metal 렌더러 (macOS)
- ✅ OpenGL 렌더러 (Linux)
- ✅ 리거처 지원 (60fps)
- ✅ True Color (24비트)
- ✅ 이모지 및 유니코드

### 터미널 기능
- ✅ xterm 호환성
- ✅ iTerm2 이미지 프로토콜
- ✅ Kitty 이미지 프로토콜
- ✅ OSC 52 (클립보드)
- ✅ 쉘 통합 (Fish, Zsh)

---

## 기술 스택

| 컴포넌트 | 기술 |
|---------|------|
| **코어** | Zig 0.13.0+ |
| **macOS UI** | SwiftUI |
| **Linux UI** | GTK 4 |
| **macOS 렌더러** | Metal + CoreText |
| **Linux 렌더러** | OpenGL + Fontconfig |
| **빌드** | Zig Build System |

---

## libghostty - 임베딩 가능한 터미널

Ghostty는 C 호환 라이브러리로도 제공되어 모든 애플리케이션에 터미널을 임베딩할 수 있습니다.

```c
#include <ghostty/ghostty.h>

ghostty_t *term = ghostty_new();
ghostty_write(term, "Hello, World!\n", 14);
ghostty_render(term, render_callback, user_data);
ghostty_free(term);
```

**사용 사례:**
- 텍스트 에디터 (VSCode, Neovim)
- IDE 통합 터미널
- 게임 콘솔
- 웹 브라우저 터미널 (WASM)

---

## 로드맵

| # | 항목 | 상태 |
|---|------|------|
| 1 | 표준 준수 터미널 에뮬레이션 | ✅ 완료 |
| 2 | 경쟁력 있는 성능 | ✅ 완료 |
| 3 | 기본 커스터마이징 | ✅ 완료 |
| 4 | 윈도우/탭/스플릿 | ✅ 완료 |
| 5 | 네이티브 플랫폼 경험 | ⚠️ 진행 중 |
| 6 | libghostty | ⚠️ 진행 중 |
| 7 | Windows 지원 | ❌ 계획됨 |

---

## 학습 자료

- **공식 웹사이트**: [https://ghostty.org](https://ghostty.org)
- **문서**: [https://ghostty.org/docs](https://ghostty.org/docs)
- **GitHub**: [https://github.com/ghostty-org/ghostty](https://github.com/ghostty-org/ghostty)
- **다운로드**: [https://ghostty.org/download](https://ghostty.org/download)

---

## 시작하기

Ghostty의 강력함을 직접 경험해보세요! [01. 소개 및 개요]({{ site.baseurl }}/ghostty-guide-01-intro/)부터 시작하여 단계별로 학습할 수 있습니다.

---

*Zig의 성능과 안전성, GPU 가속 렌더링, 플랫폼 네이티브 UI를 결합한 Ghostty로 차세대 터미널 경험을 만나보세요!*
