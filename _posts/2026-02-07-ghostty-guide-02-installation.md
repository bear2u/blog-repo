---
layout: post
title: "Ghostty 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-02-07
permalink: /ghostty-guide-02-installation/
author: Mitchell Hashimoto
categories: [개발 도구, CLI]
tags: [Ghostty, Installation, Setup, Configuration]
original_url: "https://github.com/ghostty-org/ghostty"
excerpt: "Ghostty 설치 방법 및 첫 설정 가이드"
---

## 설치 방법

### macOS

#### Homebrew (권장)

```bash
brew install ghostty
```

#### 직접 다운로드

1. https://ghostty.org/download 방문
2. macOS용 `.dmg` 파일 다운로드
3. 애플리케이션 폴더로 드래그

### Linux

#### Flatpak (권장)

```bash
flatpak install flathub org.ghostty.ghostty
```

#### Ubuntu/Debian

```bash
# 빌드 의존성
sudo apt install zig libgtk-4-dev

# 소스 빌드
git clone https://github.com/ghostty-org/ghostty
cd ghostty
zig build -Doptimize=ReleaseFast
```

#### Fedora

```bash
# 빌드 의존성
sudo dnf install zig gtk4-devel

# 소스 빌드
git clone https://github.com/ghostty-org/ghostty
cd ghostty
zig build -Doptimize=ReleaseFast
```

### 소스에서 빌드

#### 요구사항

- **Zig**: 0.13.0 이상
- **macOS**: Xcode 15.0+
- **Linux**: GTK 4, OpenGL

```bash
# 클론
git clone https://github.com/ghostty-org/ghostty
cd ghostty

# 빌드
zig build -Doptimize=ReleaseFast

# 실행
./zig-out/bin/ghostty
```

---

## 첫 실행

### 기본 실행

```bash
ghostty
```

### 명령 실행

```bash
ghostty -e vim myfile.txt
ghostty -e "ssh user@server"
```

---

## 설정 파일

### 설정 파일 위치

```
macOS: ~/.config/ghostty/config
Linux: ~/.config/ghostty/config
```

### 기본 설정 예제

```toml
# ~/.config/ghostty/config

# 폰트
font-family = "JetBrains Mono"
font-size = 14

# 색상 테마
theme = "dracula"

# 윈도우
window-padding-x = 10
window-padding-y = 10

# 커서
cursor-style = "block"
cursor-style-blink = false

# 키바인딩
keybind = ctrl+shift+c=copy_to_clipboard
keybind = ctrl+shift+v=paste_from_clipboard
```

---

## 색상 테마

### 내장 테마

```toml
theme = "dracula"
# 또는
theme = "gruvbox-dark"
theme = "nord"
theme = "solarized-dark"
```

### 커스텀 색상

```toml
background = "#1e1e2e"
foreground = "#cdd6f4"

# 일반 색상
palette = 0=#45475a
palette = 1=#f38ba8
palette = 2=#a6e3a1
palette = 3=#f9e2af
palette = 4=#89b4fa
palette = 5=#f5c2e7
palette = 6=#94e2d5
palette = 7=#bac2de
```

---

## 폰트 설정

```toml
font-family = "Fira Code"
font-size = 13

# 리거처 활성화
font-feature = "calt"
font-feature = "liga"

# 볼드/이탤릭
font-family-bold = "Fira Code Bold"
font-family-italic = "Fira Code Italic"
```

---

## 키바인딩

```toml
# 클립보드
keybind = ctrl+shift+c=copy_to_clipboard
keybind = ctrl+shift+v=paste_from_clipboard

# 탭 관리
keybind = ctrl+shift+t=new_tab
keybind = ctrl+shift+w=close_tab
keybind = ctrl+tab=next_tab
keybind = ctrl+shift+tab=previous_tab

# 스플릿
keybind = ctrl+shift+d=new_split:right
keybind = ctrl+shift+shift+d=new_split:down

# 폰트 크기
keybind = ctrl+plus=increase_font_size:1
keybind = ctrl+minus=decrease_font_size:1
keybind = ctrl+0=reset_font_size
```

---

## 빠른 시작 가이드

### 1. 터미널 열기

```bash
ghostty
```

### 2. 새 탭 열기

```
macOS: Cmd+T
Linux: Ctrl+Shift+T
```

### 3. 수평/수직 스플릿

```
macOS: Cmd+D (수평), Cmd+Shift+D (수직)
Linux: Ctrl+Shift+D (수평), Ctrl+Shift+Shift+D (수직)
```

### 4. 탭 간 이동

```
macOS: Cmd+1, Cmd+2, ...
Linux: Alt+1, Alt+2, ...
```

---

*다음 글에서는 Ghostty의 아키텍처를 자세히 살펴봅니다.*
