---
layout: post
title: "Ghostty 완벽 가이드 (03) - 아키텍처 분석"
date: 2026-02-07
permalink: /ghostty-guide-03-architecture/
author: Mitchell Hashimoto
categories: [개발 도구, CLI]
tags: [Ghostty, Architecture, Zig, Metal, OpenGL]
original_url: "https://github.com/ghostty-org/ghostty"
excerpt: "Ghostty의 내부 아키텍처와 멀티 렌더러 시스템"
---

## 전체 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                   Ghostty Application                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌────────────┐  ┌─────────────┐  ┌────────────────┐   │
│  │  macOS UI  │  │  Linux UI   │  │   Embedded     │   │
│  │  (SwiftUI) │  │    (GTK)    │  │  (libghostty)  │   │
│  └─────┬──────┘  └──────┬──────┘  └────────┬───────┘   │
│        └────────────┬────────────────────┘              │
├──────────────────────┼──────────────────────────────────┤
│              Ghostty Core (Zig)                          │
│  ┌───────────────────┴────────────────────┐             │
│  │         Terminal Emulator Core          │             │
│  │  - VT Sequence Parsing                  │             │
│  │  - Screen State Management              │             │
│  │  - PTY Management                       │             │
│  └──────────────┬──────────────────────────┘             │
├─────────────────┼──────────────────────────────────────┤
│       Rendering Layer                                    │
│  ┌──────────┬──┴────────┬───────────┐                  │
│  │  Metal   │  OpenGL   │  Software │                  │
│  │ (macOS)  │  (Linux)  │  (Fallback)│                 │
│  └─────┬────┴────┬──────┴─────┬─────┘                  │
├────────┼─────────┼────────────┼────────────────────────┤
│        ▼         ▼            ▼                          │
│     CoreText  Fontconfig  Platform Fonts                │
└──────────────────────────────────────────────────────────┘
```

---

## 핵심 컴포넌트

### 1. Terminal Core (Zig)

```zig
// src/terminal/Terminal.zig
pub const Terminal = struct {
    screen: Screen,
    parser: Parser,
    pty: Pty,

    pub fn write(self: *Terminal, data: []const u8) !void {
        try self.parser.feed(data);
    }
};
```

### 2. VT Parser

VT 시퀀스 파싱 및 처리:
- CSI (Control Sequence Introducer)
- OSC (Operating System Command)
- DCS (Device Control String)

### 3. Screen State

- **Grid**: 문자 그리드 (행 x 열)
- **Cursor**: 현재 커서 위치
- **Attributes**: 색상, 스타일 (볼드, 이탤릭 등)
- **Scrollback**: 스크롤백 버퍼

---

## 멀티 렌더러 아키텍처

### Metal 렌더러 (macOS)

```swift
// Metal Shader
vertex VertexOut vertex_main(uint vertex_id [[vertex_id]]) {
    // 정점 셰이더
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    // 프래그먼트 셰이더 - 글리프 렌더링
}
```

**장점:**
- 60fps 유지 (리거처 포함)
- 저전력 소비
- macOS 네이티브 가속

### OpenGL 렌더러 (Linux)

```c
// OpenGL 렌더링 파이프라인
glUseProgram(shader_program);
glBindVertexArray(vao);
glDrawArrays(GL_TRIANGLES, 0, vertex_count);
```

---

## PTY (Pseudo-Terminal)

```zig
// src/pty/Pty.zig
pub fn spawn(self: *Pty, argv: []const []const u8) !void {
    const pid = try std.os.fork();
    if (pid == 0) {
        // 자식 프로세스 - 셸 실행
        try std.os.execvpe(argv[0], argv, environ);
    }
}
```

---

*다음 글에서는 설정 시스템을 살펴봅니다.*
