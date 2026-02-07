---
layout: post
title: "Ghostty 완벽 가이드 (06) - 터미널 에뮬레이션"
date: 2026-02-07
permalink: /ghostty-guide-06-terminal-emulation/
author: Mitchell Hashimoto
categories: [개발 도구, CLI]
tags: [Ghostty, VT, xterm, Terminal Emulation]
original_url: "https://github.com/ghostty-org/ghostty"
excerpt: "VT 시퀀스와 xterm 호환성"
---

## VT 시퀀스 지원

### CSI (Control Sequence Introducer)
```
ESC [ <params> <command>
```

예:
- `ESC[2J` - 화면 지우기
- `ESC[31m` - 빨간색 텍스트
- `ESC[H` - 커서를 홈으로 이동

### OSC (Operating System Command)
```
ESC ] <command> ; <params> BEL
```

예:
- `ESC]0;Title\007` - 윈도우 제목 설정
- `ESC]52;c;<base64>\007` - 클립보드 복사

---

## xterm 호환성

Ghostty는 포괄적인 xterm 감사를 통과했습니다:
- ✅ 커서 이동
- ✅ 색상 (256색, True Color)
- ✅ 대체 스크린
- ✅ 마우스 이벤트
- ✅ 브래킷 페이스트

---

## 이미지 프로토콜

### iTerm2 이미지 프로토콜
```bash
printf '\033]1337;File=inline=1:'
cat image.png | base64
printf '\007'
```

### Kitty 이미지 프로토콜
```bash
kitty +kitten icat image.png
```

---

*다음 글에서는 플랫폼 통합을 살펴봅니다.*
