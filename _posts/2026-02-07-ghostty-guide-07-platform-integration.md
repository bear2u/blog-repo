---
layout: post
title: "Ghostty 완벽 가이드 (07) - 플랫폼 통합"
date: 2026-02-07
permalink: /ghostty-guide-07-platform-integration/
author: Mitchell Hashimoto
categories: [개발 도구, CLI]
tags: [Ghostty, macOS, Linux, GTK, SwiftUI]
original_url: "https://github.com/ghostty-org/ghostty"
excerpt: "플랫폼별 네이티브 기능 활용"
---

## macOS 통합

### SwiftUI 앱
```swift
struct GhosttyApp: App {
    var body: some Scene {
        WindowGroup {
            TerminalView()
        }
        .commands {
            GhosttyCommands()
        }
    }
}
```

### 네이티브 기능
- ✅ 메뉴바
- ✅ 윈도우 관리
- ✅ 설정 패널
- ✅ 시스템 색상 스킴
- ✅ Touch Bar

---

## Linux/GTK 통합

### GTK 4 앱
```c
GtkApplication *app = gtk_application_new(
    "org.ghostty.ghostty",
    G_APPLICATION_DEFAULT_FLAGS
);
```

### 네이티브 기능
- ✅ GNOME 통합
- ✅ KDE 통합
- ✅ DBus 지원
- ✅ 시스템 테마

---

*다음 글에서는 libghostty를 살펴봅니다.*
