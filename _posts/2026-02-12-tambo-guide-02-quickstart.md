---
layout: post
title: "Tambo 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-02-12
permalink: /tambo-guide-02-quickstart/
author: Tambo AI
categories: [AI 에이전트, 웹 개발]
tags: [Tambo, React, 빠른 시작, npm, create-tambo-app]
original_url: "https://github.com/tambo-ai/tambo"
excerpt: "5분 만에 Tambo 앱 시작하기"
---

## 전제 조건

Tambo 모노레포의 개발 전제(레포 내 AGENTS.md 기준):
- Node.js >= 22
- npm >= 11

---

## create-tambo-app로 시작하기

README에서 안내하는 빠른 시작은 아래 한 줄입니다.

```bash
npm create tambo-app my-tambo-app  # auto-initializes git + tambo setup
cd my-tambo-app
npm run dev
```

이 흐름은 새로운 프로젝트를 초기화하고, 개발 서버를 실행하는 데 필요한 기본 설정을 자동으로 구성합니다.

---

## 템플릿과 컴포넌트 라이브러리

Tambo는 예제/템플릿을 제공합니다.

- 사전 구축 컴포넌트 라이브러리: https://ui.tambo.co
- 템플릿(README):
  - 생성형 UI가 포함된 AI 채팅
  - AI 분석 대시보드

---

## 클라우드 vs 자체 호스팅

시작할 때는 보통 다음 두 가지 중 하나를 선택합니다.

- **Tambo Cloud**: 호스팅된 백엔드 사용
- **자체 호스팅**: Docker로 동일한 백엔드를 직접 운영

자체 호스팅은 시리즈 후반(운영 편)에서 다룹니다.

*다음 글에서는 Tambo가 컴포넌트를 어떻게 "도구"로 만들고 UI로 연결하는지 흐름을 설명합니다.*
