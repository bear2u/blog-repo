---
layout: post
title: "Ghostty 완벽 가이드 (05) - 렌더링 시스템"
date: 2026-02-07
permalink: /ghostty-guide-05-rendering/
author: Mitchell Hashimoto
categories: [개발 도구, CLI]
tags: [Ghostty, Rendering, Metal, OpenGL, Performance]
original_url: "https://github.com/ghostty-org/ghostty"
excerpt: "GPU 가속 렌더링과 성능 최적화"
---

## 렌더링 파이프라인

```
텍스트 입력
    ↓
VT 파서
    ↓
스크린 상태 업데이트
    ↓
글리프 래스터화
    ↓
GPU 텍스처 업로드
    ↓
Metal/OpenGL 렌더링
    ↓
화면 출력 (60fps)
```

---

## Metal 렌더러 (macOS)

### 특징
- **리거처 지원**: 60fps 유지
- **저전력**: 배터리 효율적
- **CoreText 통합**: 네이티브 폰트 렌더링

### 성능
```
리거처 활성화:
Ghostty (Metal)   60 fps ████████████
iTerm2 (CPU)      30 fps ██████
```

---

## OpenGL 렌더러 (Linux)

### 특징
- **Fontconfig**: 폰트 발견
- **FreeType**: 글리프 래스터화
- **OpenGL 3.3+**: 셰이더 기반 렌더링

---

## 성능 최적화

### 글리프 캐싱
```zig
const GlyphCache = struct {
    atlas: Texture,
    glyphs: HashMap(GlyphKey, GlyphInfo),
};
```

### 더티 영역 추적
- 변경된 영역만 재렌더링
- 커서 깜빡임 최소화

---

*다음 글에서는 터미널 에뮬레이션을 살펴봅니다.*
