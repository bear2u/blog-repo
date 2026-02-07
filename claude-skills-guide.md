---
layout: page
title: Claude Skills 완벽 가이드
permalink: /claude-skills-guide/
---

# Claude Skills 완벽 가이드

> Anthropic의 공식 Claude Skills 가이드를 한국어로 완벽 번역한 종합 가이드입니다.

Claude Skills는 AI 에이전트가 특정 작업이나 워크플로우를 수행하는 방법을 가르치는 강력한 시스템입니다. 이 가이드는 스킬의 기초부터 고급 패턴, 배포 전략까지 모든 것을 다룹니다.

---

## 📚 가이드 목차

### [01. 소개 및 개요](/claude-skills-guide-01-intro/)
- Claude Skills란 무엇인가
- 왜 Skills가 필요한가
- MCP vs Skills 비교
- 가이드 구조 소개

### [02. 기초 개념](/claude-skills-guide-02-fundamentals/)
- 스킬의 정의와 구조
- Progressive Disclosure (점진적 공개)
- Composability (구성 가능성)
- Portability (이식성)
- MCP 개발자를 위한 Skills

### [03. 계획 및 설계](/claude-skills-guide-03-planning/)
- 스킬 만들기 전 질문들
- 유스케이스 정의하기
- 스킬 카테고리 선택
  - Document Creation Skills
  - Workflow Automation Skills
  - MCP Enhancement Skills
- 설계 체크리스트

### [04. 스킬 구조 및 YAML](/claude-skills-guide-04-structure/)
- 파일 구조 및 네이밍 규칙
- SKILL.md 작성법
- YAML Frontmatter 완벽 가이드
- 필수 및 선택 필드
- 보안 제약사항

### [05. 효과적인 명령어 작성](/claude-skills-guide-05-writing/)
- Description 필드 작성법
- 메인 명령어 구조
- 베스트 프랙티스
- Progressive Disclosure 활용
- 실전 예시

### [06. 테스트 및 반복 개선](/claude-skills-guide-06-testing/)
- 테스트 수준 (수동/스크립트/API)
- 트리거 테스트
- 기능 테스트
- 성능 비교
- skill-creator 스킬 활용
- 피드백 기반 반복

### [07. 배포 및 공유](/claude-skills-guide-07-distribution/)
- 현재 배포 모델
- 조직 레벨 배포
- API를 통한 스킬 사용
- GitHub 호스팅 전략
- 스킬 포지셔닝
- 릴리스 체크리스트

### [08. 실전 패턴](/claude-skills-guide-08-patterns/)
- Pattern 1: Sequential Workflow Orchestration
- Pattern 2: Multi-MCP Coordination
- Pattern 3: Iterative Refinement
- Pattern 4: Context-Aware Tool Selection
- Pattern 5: Domain-Specific Intelligence
- 패턴 선택 가이드

### [09. 트러블슈팅 및 참고 자료](/claude-skills-guide-09-troubleshooting/)
- 일반적인 문제와 해결책
  - 업로드 실패
  - 트리거 문제
  - MCP 연결 문제
  - 명령어 무시
  - 성능 문제
- 빠른 참고 자료
- 개발 체크리스트
- YAML 스펙
- 유용한 리소스

---

## 🎯 누구를 위한 가이드인가요?

### MCP 개발자
이미 MCP 서버를 만들었다면, Skills는 다음 단계입니다. MCP가 **연결성**을 제공한다면, Skills는 **지식**을 제공합니다.

### 제품 개발자
Claude를 제품에 통합하고 있다면, Skills를 통해 일관되고 신뢰할 수 있는 사용자 경험을 제공할 수 있습니다.

### AI 에이전트 빌더
복잡한 워크플로우를 자동화하는 AI 에이전트를 만들고 있다면, Skills는 베스트 프랙티스를 코드화하는 방법입니다.

### 엔터프라이즈 IT 관리자
조직 전체에 AI 워크플로우를 배포하고 관리해야 한다면, Skills는 중앙 집중식 관리를 제공합니다.

---

## 🚀 빠른 시작

### 1. 기초 이해하기
[01. 소개](/claude-skills-guide-01-intro/)와 [02. 기초 개념](/claude-skills-guide-02-fundamentals/)부터 시작하세요.

### 2. 첫 스킬 계획하기
[03. 계획 및 설계](/claude-skills-guide-03-planning/)를 읽고 유스케이스를 정의하세요.

### 3. 스킬 만들기
[04. 구조](/claude-skills-guide-04-structure/)와 [05. 명령어 작성](/claude-skills-guide-05-writing/)을 따라 스킬을 만드세요.

### 4. 테스트 및 개선
[06. 테스트](/claude-skills-guide-06-testing/)로 스킬을 검증하고 개선하세요.

### 5. 배포하기
[07. 배포](/claude-skills-guide-07-distribution/)를 참고해 사용자에게 전달하세요.

---

## 💡 주요 개념

### Progressive Disclosure (점진적 공개)
스킬은 3단계 시스템을 사용합니다:
1. **YAML Frontmatter** - 항상 로드 (최소 정보)
2. **SKILL.md 본문** - 관련 시 로드 (핵심 지침)
3. **참조 파일** - 필요시 로드 (상세 문서)

### MCP + Skills = 완벽한 통합
- **MCP**: Claude를 서비스에 연결 (할 수 있는 것)
- **Skills**: 서비스 사용 방법 가르침 (어떻게 해야 하는 것)

### 실전에서 검증된 패턴
5가지 실전 패턴으로 대부분의 유스케이스를 커버할 수 있습니다.

---

## 📖 원본 출처

이 가이드는 Anthropic의 공식 Claude Skills 가이드를 기반으로 합니다:
- **GitHub**: [corca-ai/claude-plugins](https://github.com/corca-ai/claude-plugins/tree/main/references/anthropic-skills-guide)
- **원저자**: Anthropic
- **번역 및 편집**: AI Agent Guide Series

---

## 🔗 관련 리소스

### 공식 문서
- [Claude Skills 문서](https://docs.anthropic.com/skills)
- [Claude API 문서](https://docs.anthropic.com/api)
- [MCP 프로토콜](https://modelcontextprotocol.io)

### 커뮤니티
- [Claude Developers Discord](https://discord.gg/claude)
- [GitHub - anthropics/skills](https://github.com/anthropics/skills)
- [Skills 버그 리포트](https://github.com/anthropics/skills/issues)

### 예시 스킬
- [Document Skills](https://github.com/anthropics/skills/tree/main/document-skills)
- [Partner Skills](https://github.com/anthropics/skills/tree/main/partners)
- [Example Skills](https://github.com/anthropics/skills/tree/main/examples)

---

## 📝 가이드 사용 팁

### 순차적 학습
처음부터 순서대로 읽으면 가장 효과적입니다. 각 챕터는 이전 챕터의 개념을 기반으로 합니다.

### 레퍼런스로 활용
특정 문제를 해결할 때 [09. 트러블슈팅](/claude-skills-guide-09-troubleshooting/)을 빠른 레퍼런스로 사용하세요.

### 실습하며 학습
각 챕터의 예시를 직접 따라해 보세요. 실제로 스킬을 만들어보는 것이 가장 빠른 학습 방법입니다.

---

## 🤝 기여하기

가이드에 오류를 발견하거나 개선 사항이 있다면:
- GitHub Issues에 리포트해 주세요
- Pull Request를 제출해 주세요
- 피드백을 이메일로 보내주세요

---

## 📅 업데이트 로그

- **2026-02-07**: 초판 발행 (전체 9개 챕터)
- 기반: Anthropic Skills Guide (2025 버전)

---

**시작할 준비가 되셨나요?** [01. 소개 및 개요](/claude-skills-guide-01-intro/)로 시작하세요!

---

<div style="text-align: center; margin: 40px 0; padding: 20px; background: #f5f5f5; border-radius: 8px;">
  <p style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">
    🎓 Complete Guide Series
  </p>
  <p style="margin: 0;">
    9개 챕터 • 종합 가이드 • 실전 예시 • 트러블슈팅
  </p>
</div>
