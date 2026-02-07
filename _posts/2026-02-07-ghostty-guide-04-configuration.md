---
layout: post
title: "Ghostty 완벽 가이드 (04) - 설정 및 커스터마이징"
date: 2026-02-07
permalink: /ghostty-guide-04-configuration/
author: Mitchell Hashimoto
categories: [개발 도구, CLI]
tags: [Ghostty, Configuration, Customization, Themes]
original_url: "https://github.com/ghostty-org/ghostty"
excerpt: "Ghostty 설정 파일 완벽 가이드"
---

## 설정 파일 구조

```toml
# ~/.config/ghostty/config

# 폰트
font-family = "JetBrains Mono"
font-size = 14
font-feature = "calt"
font-feature = "liga"

# 테마
theme = "catppuccin-mocha"

# 윈도우
window-padding-x = 10
window-padding-y = 10
window-decoration = true

# 커서
cursor-style = "block"
cursor-color = "#f5e0dc"

# 쉘
shell-integration = "fish"
shell-integration-features = "cursor,sudo,title"

# 마우스
mouse-hide-while-typing = true
copy-on-select = true

# 벨
audible-bell = false
visual-bell = true
```

---

## 색상 테마

### Catppuccin Mocha

```toml
theme = "catppuccin-mocha"
```

### Nord

```toml
theme = "nord"
```

### 커스텀 테마

```toml
background = "#1e1e2e"
foreground = "#cdd6f4"

# 검은색
palette = 0=#45475a
palette = 8=#585b70

# 빨간색
palette = 1=#f38ba8
palette = 9=#f38ba8

# 초록색
palette = 2=#a6e3a1
palette = 10=#a6e3a1
```

---

## 폰트 설정

```toml
font-family = "Fira Code"
font-size = 13

# 리거처
font-feature = "calt"
font-feature = "liga"
font-feature = "dlig"

# 변형
font-family-bold = "Fira Code Bold"
font-family-italic = "Fira Code Italic"
font-family-bold-italic = "Fira Code Bold Italic"

# 대체 폰트
font-family-fallback = "Apple Color Emoji"
font-family-fallback = "Noto Color Emoji"
```

---

## 키바인딩

```toml
# 클립보드
keybind = ctrl+shift+c=copy_to_clipboard
keybind = ctrl+shift+v=paste_from_clipboard

# 탭
keybind = ctrl+shift+t=new_tab
keybind = ctrl+shift+w=close_tab
keybind = ctrl+tab=next_tab

# 스플릿
keybind = ctrl+shift+d=new_split:right
keybind = ctrl+shift+shift+d=new_split:down
keybind = ctrl+shift+[=goto_split:left
keybind = ctrl+shift+]=goto_split:right

# 검색
keybind = ctrl+shift+f=toggle_quick_terminal
```

---

## 윈도우 설정

```toml
# 크기
window-width = 100
window-height = 30

# 패딩
window-padding-x = 10
window-padding-y = 10
window-padding-balance = true

# 장식
window-decoration = true
window-title-font-family = "SF Pro Display"

# 투명도
background-opacity = 0.95
background-blur-radius = 20
```

---

*다음 글에서는 렌더링 시스템을 살펴봅니다.*
