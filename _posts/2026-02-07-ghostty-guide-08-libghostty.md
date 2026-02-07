---
layout: post
title: "Ghostty 완벽 가이드 (08) - libghostty"
date: 2026-02-07
permalink: /ghostty-guide-08-libghostty/
author: Mitchell Hashimoto
categories: [개발 도구, CLI]
tags: [Ghostty, libghostty, Embedding, C API]
original_url: "https://github.com/ghostty-org/ghostty"
excerpt: "임베딩 가능한 터미널 라이브러리"
---

## libghostty 개요

libghostty는 C 호환 라이브러리로, 모든 애플리케이션에 터미널을 임베딩할 수 있습니다.

---

## 라이브러리 구조

```
libghostty/
├── libghostty-vt    # VT 파서 + 터미널 상태
├── libghostty-ui    # UI 렌더링 (개발 중)
└── libghostty       # 완전한 터미널
```

---

## C API 예제

```c
#include <ghostty/ghostty.h>

// 터미널 생성
ghostty_t *term = ghostty_new();

// 데이터 쓰기
const char *data = "Hello, World!\n";
ghostty_write(term, data, strlen(data));

// 화면 렌더링
ghostty_render(term, render_callback, user_data);

// 정리
ghostty_free(term);
```

---

## Zig API 예제

```zig
const ghostty = @import("ghostty");

var term = try ghostty.Terminal.init(allocator);
defer term.deinit();

try term.write("Hello, World!\n");
```

---

## 사용 사례

- **텍스트 에디터**: VSCode, Neovim
- **IDE**: 통합 터미널
- **게임**: 인게임 콘솔
- **웹**: WASM 터미널

---

*다음 글에서는 고급 기능을 살펴봅니다.*
