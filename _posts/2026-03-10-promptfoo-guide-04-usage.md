---
layout: post
title: "promptfoo 완벽 가이드 (04) - 실전 사용 패턴"
date: 2026-03-10
permalink: /promptfoo-guide-04-usage/
author: "Ian Webster"
categories: [개발 도구, promptfoo]
tags: [Trending, GitHub, promptfoo]
original_url: "https://github.com/promptfoo/promptfoo"
excerpt: "init→eval→view 중심으로 사용 패턴을 잡습니다."
---

## 1) 샘플로 시작해서 성공 경험 만들기

```bash
promptfoo init --example getting-started
cd getting-started
promptfoo eval
promptfoo view
```

---

## 2) 설정 파일 기반으로 확장

- 다양한 커맨드/툴 구현에서 기본 설정 파일로 `promptfooconfig.yaml`를 가정하는 흐름이 보입니다.
- 팀에서는 “샘플 → 설정 파일 고정 → CI에 eval” 순서가 가장 안전합니다.

---

## 3) 운영/협업 관점

- CI에서 반복 가능한 eval을 돌려 **모델/프롬프트 변경에 대한 회귀**를 막습니다.
- 레드팀/취약점 스캔은 “정책/가이드”와 같이 관리하면 결과를 공유하기 쉽습니다.

---

*다음 글에서는 운영/확장/베스트 프랙티스를 정리합니다.*

