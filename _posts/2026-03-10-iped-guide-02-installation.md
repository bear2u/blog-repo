---
layout: post
title: "IPED 완벽 가이드 (02) - 설치/빌드 및 빠른 시작"
date: 2026-03-10
permalink: /iped-guide-02-installation/
author: sepinf-inc
categories: [개발 도구, iped]
tags: [Trending, GitHub, IPED]
original_url: "https://github.com/sepinf-inc/IPED"
excerpt: "Maven/JDK11 기반 빌드 흐름을 정리합니다."
---

## 요구사항(README 기준)

- Git
- Maven
- Java JDK 11 + JavaFX(예: Liberica OpenJDK 11 Full)

---

## 소스 빌드

```bash
git clone https://github.com/sepinf-inc/IPED.git
cd IPED
mvn clean install
```

빌드 결과는 `target/release`에 생성된다고 README에 안내되어 있습니다.

---

## 주의사항(README 기반)

- 기본 `master` 브랜치는 개발 브랜치로 **불안정**할 수 있으니, 안정 버전은 릴리즈 태그 체크아웃을 고려합니다.
- Linux의 경우 Sleuthkit 및 추가 의존성이 필요할 수 있으니 Wiki(Linux 섹션)를 확인합니다.

---

*다음 글에서는 핵심 개념과 아키텍처를 정리합니다.*

