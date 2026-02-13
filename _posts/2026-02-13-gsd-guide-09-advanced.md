---
layout: post
title: "GSD (Get Shit Done) 완벽 가이드 (09) - 심화 활용법"
date: 2026-02-13
permalink: /gsd-guide-09-advanced/
author: TÂCHES
categories: [AI 코딩, 개발 도구]
tags: [Claude Code, Advanced, Best Practices, Tips]
original_url: "https://github.com/gsd-build/get-shit-done"
excerpt: "GSD 심화 활용법과 모범 사례"
---

## TDD (테스트 주도 개발) 통합

GSD는 TDD를 위한 **전용 계획 타입**을 제공합니다.

### TDD 감지

**테스트:** `fn`을 작성하기 전에 `expect(fn(input)).toBe(output)`를 작성할 수 있는가?

- **예** → 전용 TDD 계획 생성 (type: tdd)
- **아니오** → 표준 태스크

### TDD 후보

| TDD 계획 | 표준 태스크 |
|----------|------------|
| 비즈니스 로직 | UI 레이아웃/스타일 |
| 정의된 I/O가 있는 API 엔드포인트 | 설정 |
| 데이터 변환 | 글루 코드 |
| 검증 규칙 | 일회성 스크립트 |
| 알고리즘 | 단순 CRUD |
| 상태 머신 | |

### TDD가 별도 계획인 이유

TDD는 **RED→GREEN→REFACTOR** 사이클이 필요하며 컨텍스트의 40-50%를 소비합니다. 다중 태스크 계획에 포함하면 품질이 저하됩니다.

---

## Discovery Levels (발견 수준)

Discovery는 현재 컨텍스트가 존재함을 증명할 수 없는 한 **필수**입니다.

### Level 0 - 건너뛰기

**조건:** 모든 작업이 확립된 코드베이스 패턴을 따름

**예시:**
- 삭제 버튼 추가
- 모델에 필드 추가
- CRUD 엔드포인트 생성

### Level 1 - 빠른 검증 (2-5분)

**조건:** 단일 알려진 라이브러리, 구문/버전 확인

**동작:** Context7 resolve-library-id + query-docs, DISCOVERY.md 불필요

### Level 2 - 표준 리서치 (15-30분)

**조건:** 2-3개 옵션 중 선택, 새로운 외부 통합

**동작:** Discovery 워크플로우로 라우팅, DISCOVERY.md 생성

### Level 3 - 심층 조사 (1시간+)

**조건:** 장기 영향이 있는 아키텍처 결정, 새로운 문제

**동작:** DISCOVERY.md로 전체 리서치

### 수준 표시기

- **Level 2+:** package.json에 없는 새 라이브러리, 외부 API, 설명에 "choose/select/evaluate"
- **Level 3:** "architecture/design/system", 여러 외부 서비스, 데이터 모델링, 인증 설계

---

## 사용자 설정 감지

태스크에 사용자 설정이 필요한 경우 GSD가 자동으로 감지합니다.

### 설정 유형

| 유형 | 예시 |
|------|------|
| API 키 | OpenAI, Anthropic, Stripe |
| 환경 변수 | DATABASE_URL, NODE_ENV |
| 설정 파일 | config.json, .env |
| CLI 로그인 | vercel, aws, gcloud |

### 프로토콜

1. 태스크에 설정이 필요함을 감지
2. 설정 단계를 태스크에 포함
3. 체크포인트로 사용자 설정 확인

---

## 체크포인트 타입

| 타입 | 용도 | 동작 |
|------|------|------|
| `auto` | Claude가 독립적으로 수행할 수 있는 모든 것 | 완전 자율 |
| `checkpoint:human-verify` | 시각적/기능적 검증 | 사용자 대기 |
| `checkpoint:decision` | 구현 선택 | 사용자 대기 |
| `checkpoint:human-action` | 진정으로 불가피한 수동 단계 (드묾) | 사용자 대기 |

### 자동화 우선 규칙

Claude가 CLI/API로 할 수 있다면 Claude가 **반드시** 해야 합니다. 체크포인트는 자동화를 대체하는 것이 아니라 자동화 **후에** 검증합니다.

---

## 태스크 크기 가이드라인

각 태스크: **15-60분** Claude 실행 시간

| 시간 | 동작 |
|------|------|
| < 15분 | 너무 작음 — 관련 태스크와 결합 |
| 15-60분 | 적절한 크기 |
| > 60분 | 너무 큼 — 분할 |

### 너무 큰 신호

- 3-5개 이상의 파일을 건드림
- 여러 개별 청크
- action 섹션이 1단락 이상

### 결합 신호

- 한 태스크가 다음 태스크를 준비
- 별도의 태스크가 같은 파일을 건드림
- 어느 것도 단독으로 의미 없음

---

## 구체성 예시

| 너무 모호함 | 적절함 |
|------------|--------|
| "인증 추가" | "jose 라이브러리로 JWT 인증 추가, httpOnly 쿠키에 저장, 15분 액세스 / 7일 리프레시" |
| "API 만들기" | "POST /api/projects 엔드포인트 생성, {name, description} 수락, 이름 길이 3-50자 검증, 201과 프로젝트 객체 반환" |
| "대시보드 스타일링" | "Dashboard.tsx에 Tailwind 클래스 추가: 그리드 레이아웃 (lg에서 3열, 모바일에서 1열), 카드 그림자, 액션 버튼에 호버 상태" |
| "에러 처리" | "API 호출을 try/catch로 래핑, 4xx/5xx에서 {error: string} 반환, 클라이언트에서 sonner로 토스트 표시" |
| "데이터베이스 설정" | "schema.prisma에 User와 Project 모델 추가, UUID id, email unique 제약조건, createdAt/updatedAt 타임스탬프, prisma db push 실행" |

**테스트:** 다른 Claude 인스턴스가 명확한 질문 없이 실행할 수 있는가? 아니라면 구체성을 추가하세요.

---

## 마일스톤 워크플로우

### 완료

```
/gsd:complete-milestone
```

1. 마일스톤 아카이브
2. 릴리스 태그 생성
3. `.planning/archive/{milestone}/`로 이동

### 새 마일스톤

```
/gsd:new-milestone
```

1. 다음 버전 설명
2. 도메인 리서치
3. 요구사항 범위 지정
4. 새 로드맵 생성

각 마일스톤은 **정의 → 구축 → 출시**의 깨끗한 주기입니다.

---

## 모범 사례 요약

1. **컨텍스트 관리** — 50% 컨텍스트 한계 유지
2. **원자적 커밋** — 각 태스크에 자체 커밋
3. **구체적 지침** — 모호함 피하기
4. **사용자 결정 존중** — CONTEXT.md 충실도 유지
5. **자동화 우선** — 가능하면 자동으로, 체크포인트는 검증용
6. **반복 개선** — discuss → plan → execute → verify 루프

---

## 커뮤니티

- **Discord:** https://discord.gg/5JJgD5svVS
- **X (Twitter):** [@gsd_foundation](https://x.com/gsd_foundation)
- **GitHub:** https://github.com/gsd-build/get-shit-done

---

## 마치며

GSD는 Claude Code의 힘을 **신뢰할 수 있게** 만듭니다.

- **Context Engineering**으로 품질 저하 방지
- **멀티 에이전트**로 병렬 처리
- **원자적 커밋**으로 추적 가능성
- **검증**으로 품질 보장

복잡성은 시스템 안에, 워크플로우는 단순하게.

**Claude Code is powerful. GSD makes it reliable.**
