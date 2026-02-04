# 포스트 템플릿

## Front Matter (필수)

```yaml
---
layout: post
title: "{프로젝트명} 완벽 가이드 ({번호}) - {챕터 제목}"
date: YYYY-MM-DD
permalink: /{series-slug}-{part번호}-{slug}/
author: {원저자}
category: {카테고리}
tags: [{태그1}, {태그2}, ...]
series: {series-slug}
part: {번호}
original_url: "{원본 URL}"
excerpt: "{한 줄 요약}"
---
```

## 예시

```yaml
---
layout: post
title: "UI-TARS 완벽 가이드 (3) - Desktop 앱 분석"
date: 2025-02-04
permalink: /ui-tars-guide-03-desktop-app/
author: ByteDance
category: AI
tags: [UI-TARS, Electron, Desktop App, IPC, React]
series: ui-tars-guide
part: 3
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "UI-TARS Desktop Electron 앱의 구조를 분석합니다."
---
```

## 본문 구조

```markdown
## {섹션 1 제목}

{설명 텍스트}

```{언어}
// 코드 예제
```

---

## {섹션 2 제목}

### {서브섹션}

{내용}

---

*다음 글에서는 {다음 주제}를 살펴봅니다.*
```

## 권장 요소

### 코드 블록
- 언어 명시: ```typescript, ```python 등
- 주석으로 설명 추가
- 너무 긴 코드는 핵심만 발췌

### 다이어그램
- ASCII art 사용
- 디렉토리 구조: tree 형식
- 흐름도: 박스와 화살표

### 표
| 항목 | 설명 |
|-----|------|
| 값1 | 설명1 |

### 섹션 구분
- `---` 로 주요 섹션 구분
- 마지막에 다음 글 안내
