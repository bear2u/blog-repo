---
layout: post
title: "Prompt Engineering Guide 가이드 (03) - 프롬프트 기본기와 LLM 설정"
date: 2026-02-20
permalink: /prompt-engineering-guide-03-prompt-basics-and-settings/
author: Elvis Saravia (DAIR.AI)
categories: ['LLM 학습', '프롬프트 엔지니어링']
tags: [Prompt Design, Temperature, Top-p, Instruction]
original_url: "https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-intro.md"
excerpt: "Instruction/Context/Input/Output 요소와 temperature, top-p 설정을 어떻게 결합할지 기본 원칙을 정리합니다."
---

## 프롬프트의 4요소

`prompts-intro.md`에서 반복되는 핵심은 아래 4요소입니다.

- Instruction: 무엇을 하라고 지시하는가
- Context: 어떤 배경을 주는가
- Input data: 실제 질의/원문은 무엇인가
- Output indicator: 어떤 형식으로 내게 할 것인가

모든 요소가 항상 필요하지는 않지만, 복잡도가 올라갈수록 구조화가 중요해집니다.

---

## LLM 설정 기본

가이드에서 강조하는 기본 설정 포인트는 다음입니다.

- `temperature`: 낮을수록 결정적, 높을수록 다양성
- `top_p`: 샘플링 후보 범위 제어
- 권장: 둘 다 동시에 크게 조정하기보다 하나씩 튜닝

사실성 QA는 낮게, 창의 생성은 높게 두는 식으로 업무별 프로파일을 두는 방식이 실무에서 안정적입니다.

---

## 프롬프트 작성 원칙

`prompts-intro.md` 기준으로 실무에서 특히 유효한 원칙은 다음입니다.

1. 처음엔 단순하게 시작
2. 모호한 표현 대신 구체적 요구
3. 해야 할 행동을 명시(하지 말라보다 더 효과적)
4. 구분자와 형식을 먼저 고정

---

## 예시: 형식 강제

```text
### Instruction ###
다음 문장을 한국어로 번역하라.

Text: "hello"
```

이처럼 역할과 입력을 분리하면 모델이 추론해야 할 여지가 줄어듭니다.

다음 장에서 Zero/Few-shot, CoT 같은 고급 패턴으로 넘어갑니다.
