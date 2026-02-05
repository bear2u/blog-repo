---
layout: page
title: Ralph 가이드
permalink: /ralph-guide/
icon: fas fa-sync
---

# 🔄 Ralph for Claude Code 완벽 가이드

> **자율 AI 개발 루프 시스템**

**Ralph**는 Geoffrey Huntley의 "Ralph 기법"을 구현한 오픈소스 도구로, Claude Code를 활용한 자율적인 연속 개발 사이클을 가능하게 합니다.

---

## 📚 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/ralph-guide-01-intro/) | Ralph란? Geoffrey Huntley의 기법 |
| 02 | [설치 및 시작](/blog-repo/ralph-guide-02-installation/) | 글로벌 설치, 프로젝트 초기화 |
| 03 | [파일 구조](/blog-repo/ralph-guide-03-files/) | .ralph/, PROMPT.md, fix_plan.md |
| 04 | [핵심 개념](/blog-repo/ralph-guide-04-concepts/) | 자율 루프, 종료 감지, EXIT_SIGNAL |
| 05 | [CLI 명령어](/blog-repo/ralph-guide-05-commands/) | ralph, ralph-enable, ralph-import |
| 06 | [구성 및 설정](/blog-repo/ralph-guide-06-configuration/) | .ralphrc, 속도 제한, 타임아웃 |
| 07 | [서킷 브레이커](/blog-repo/ralph-guide-07-circuit-breaker/) | 에러 감지, 상태 전환, 자동 복구 |
| 08 | [세션 관리](/blog-repo/ralph-guide-08-session/) | 세션 연속성, 만료, 리셋 |
| 09 | [모니터링](/blog-repo/ralph-guide-09-monitoring/) | tmux, 라이브 대시보드, 로그 |
| 10 | [베스트 프랙티스](/blog-repo/ralph-guide-10-best-practices/) | 효과적인 프롬프트, 문제 해결 |

---

## ✨ 주요 특징

- **🔄 자율 개발 루프** - Claude Code가 자동으로 반복하며 프로젝트 완성
- **🧠 지능형 종료 감지** - Dual-condition 체크로 조기 종료 방지
- **⚡ 서킷 브레이커** - 무한 루프와 에러 상황 자동 감지 및 복구
- **📊 라이브 모니터링** - tmux 기반 실시간 진행 상황 대시보드
- **🔒 속도 제한** - 시간당 API 호출 제한으로 비용 관리
- **📋 PRD 가져오기** - 기존 요구사항 문서를 Ralph 형식으로 변환

---

## 🚀 빠른 시작

```bash
# 설치 (한 번만)
git clone https://github.com/frankbria/ralph-claude-code.git
cd ralph-claude-code && ./install.sh

# 프로젝트에서 활성화
cd my-project
ralph-enable
ralph --monitor
```

---

## 🔗 관련 링크

- [GitHub 저장소](https://github.com/frankbria/ralph-claude-code)
- [Ralph 기법 원문](https://ghuntley.com/ralph/)
- [Claude Code](https://claude.ai/code)
