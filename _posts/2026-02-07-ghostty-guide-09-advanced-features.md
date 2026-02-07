---
layout: post
title: "Ghostty 완벽 가이드 (09) - 고급 기능"
date: 2026-02-07
permalink: /ghostty-guide-09-advanced-features/
author: Mitchell Hashimoto
categories: [개발 도구, CLI]
tags: [Ghostty, Advanced Features, Tabs, Splits]
original_url: "https://github.com/ghostty-org/ghostty"
excerpt: "멀티윈도우, 탭, 스플릿 활용"
---

## 멀티윈도우

### 새 윈도우 열기
```bash
# macOS
Cmd+N

# Linux
Ctrl+Shift+N
```

### 설정
```toml
window-vsync = true
window-inherit-working-directory = true
window-inherit-font-size = true
```

---

## 탭 관리

### 탭 생성 및 이동
```
새 탭: Cmd+T (macOS), Ctrl+Shift+T (Linux)
탭 닫기: Cmd+W (macOS), Ctrl+Shift+W (Linux)
탭 이동: Cmd+1-9 (macOS), Alt+1-9 (Linux)
```

### 설정
```toml
tab-bar-style = "native"
tab-width = 200
```

---

## 스플릿

### 수평/수직 스플릿
```
수평: Cmd+D (macOS), Ctrl+Shift+D (Linux)
수직: Cmd+Shift+D (macOS), Ctrl+Shift+Shift+D (Linux)
```

### 스플릿 간 이동
```
왼쪽: Cmd+[ (macOS), Ctrl+Shift+[ (Linux)
오른쪽: Cmd+] (macOS), Ctrl+Shift+] (Linux)
위: Cmd+Option+Up
아래: Cmd+Option+Down
```

---

## 검색

```
검색 열기: Cmd+F (macOS), Ctrl+Shift+F (Linux)
다음 찾기: Enter
이전 찾기: Shift+Enter
```

---

## 쉘 통합

```toml
shell-integration = "fish"
shell-integration-features = "cursor,sudo,title"
```

### 지원 기능
- **커서**: 프롬프트 표시
- **sudo**: sudo 비밀번호 입력 시 보안 표시
- **title**: 자동 윈도우 제목

---

*다음 글에서는 실전 활용 팁을 살펴봅니다.*
