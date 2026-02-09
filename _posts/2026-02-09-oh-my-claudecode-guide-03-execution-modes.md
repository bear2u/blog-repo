---
layout: post
title: "oh-my-claudecode 완벽 가이드 (03) - 실행 모드 상세"
date: 2026-02-09
permalink: /oh-my-claudecode-guide-03-execution-modes/
author: Yeachan Heo
categories: [AI 코딩, CLI]
tags: [Claude Code, Multi-Agent, Orchestration, AI, CLI, Autopilot, Ultrawork]
original_url: "https://github.com/Yeachan-Heo/oh-my-claudecode"
excerpt: "Autopilot, Ultrawork, Ralph 등 oh-my-claudecode의 7가지 실행 모드를 상세히 알아보고 각 모드의 최적 사용 시나리오를 학습합니다."
---

## 실행 모드 개요

oh-my-claudecode는 7가지 실행 모드를 제공하며, 각 모드는 특정 작업 유형에 최적화되어 있습니다. 상황에 맞는 모드를 선택하면 효율성, 속도, 비용을 크게 개선할 수 있습니다.

### 모드 선택 가이드

```
빠른 작업 완료가 필요한가?        → Autopilot
병렬 처리가 가능한 작업인가?      → Ultrawork
작업이 반드시 완료되어야 하는가?  → Ralph
대규모 시스템을 구축하는가?       → Ultrapilot
비용을 절감하고 싶은가?          → Ecomode
조율된 병렬 작업이 필요한가?     → Swarm
순차적 단계가 명확한가?          → Pipeline
```

## 1. Autopilot: 빠른 자율 워크플로우

Autopilot은 가장 일반적이고 빠른 실행 모드입니다. 단일 작업을 빠르게 완료하는 데 최적화되어 있습니다.

### 특징

- **속도 최적화**: 최소한의 오버헤드로 빠른 실행
- **자율 실행**: 사용자 개입 없이 작업 완료
- **스마트 에이전트 선택**: 작업에 가장 적합한 에이전트 자동 선택
- **적응형 전략**: 작업 진행에 따라 전략 조정

### Magic Keywords

```bash
> autopilot: [task description]
> auto: [task description]
> go: [task description]
```

### 사용 시나리오

#### 시나리오 1: 새 기능 추가

```bash
> autopilot: add user profile page with avatar upload and bio editing
```

**실행 과정:**
```
1. [Architect] 프로필 페이지 구조 설계
2. [Frontend] React 컴포넌트 생성
3. [Backend] 프로필 API 엔드포인트 구현
4. [Database] 프로필 스키마 확장
5. [Testing] 테스트 케이스 작성
```

**완료 시간:** 약 3-5분

#### 시나리오 2: 버그 수정

```bash
> autopilot: fix the bug where login fails with OAuth providers
```

**실행 과정:**
```
1. [Debugger] 버그 원인 파악
2. [Security] OAuth 흐름 분석
3. [Backend] 인증 로직 수정
4. [Testing] OAuth 테스트 추가
5. [Validator] 수정 검증
```

**완료 시간:** 약 2-3분

#### 시나리오 3: 문서 생성

```bash
> autopilot: generate comprehensive API documentation with examples
```

**실행 과정:**
```
1. [Analyst] API 엔드포인트 분석
2. [Documentation] Markdown 문서 생성
3. [Generator] 코드 예제 생성
4. [Reviewer] 문서 품질 검토
```

**완료 시간:** 약 2-4분

### 성능 특성

```
속도:     ★★★★★ (매우 빠름)
병렬화:   ★★☆☆☆ (제한적)
완성도:   ★★★★☆ (높음)
비용:     ★★★☆☆ (보통)
```

### 제한사항

- 복잡한 멀티컴포넌트 작업에는 부적합
- 병렬 처리가 제한적
- 작업 실패 시 자동 재시도 없음

## 2. Ultrawork: 최대 병렬 처리

Ultrawork는 병렬 처리를 극대화하여 독립적인 작업을 동시에 실행합니다.

### 특징

- **최대 병렬화**: 가능한 모든 작업을 동시 실행
- **의존성 분석**: 작업 간 의존성 자동 파악
- **동적 스케줄링**: 실시간으로 작업 우선순위 조정
- **리소스 최적화**: CPU와 메모리 효율적 사용

### Magic Keywords

```bash
> ultrawork: [task description]
> ulw: [task description]
> parallel: [task description]
```

### 사용 시나리오

#### 시나리오 1: 다중 모델 생성

```bash
> ulw: create User, Product, Order, and Payment models with relationships
```

**병렬 실행:**
```
시간 0초:
├─ [Agent 1] User 모델 생성
├─ [Agent 2] Product 모델 생성
├─ [Agent 3] Order 모델 생성
└─ [Agent 4] Payment 모델 생성

시간 60초:
└─ [Agent 5] 모델 관계 정의 및 통합
```

**단일 에이전트:** 4분 (순차 실행)
**Ultrawork:** 1.2분 (병렬 실행)
**속도 향상:** 3.3배

#### 시나리오 2: 마이크로서비스 개발

```bash
> ulw: build auth service, user service, and notification service
```

**병렬 실행:**
```
동시 진행:
├─ [Team 1] Auth Service
│  ├─ [Agent 1] 서버 설정
│  ├─ [Agent 2] JWT 구현
│  └─ [Agent 3] 테스트
│
├─ [Team 2] User Service
│  ├─ [Agent 4] CRUD API
│  ├─ [Agent 5] 데이터베이스
│  └─ [Agent 6] 테스트
│
└─ [Team 3] Notification Service
   ├─ [Agent 7] 이메일 전송
   ├─ [Agent 8] 푸시 알림
   └─ [Agent 9] 테스트
```

**완료 시간:** 5-7분 (순차 실행: 15-20분)

#### 시나리오 3: 다국어 지원

```bash
> ulw: add i18n support for English, Korean, Japanese, Spanish, and French
```

**병렬 실행:**
```
동시 작업:
├─ [Agent 1] i18n 프레임워크 설정
├─ [Agent 2] 영어 번역
├─ [Agent 3] 한국어 번역
├─ [Agent 4] 일본어 번역
├─ [Agent 5] 스페인어 번역
└─ [Agent 6] 프랑스어 번역
```

**완료 시간:** 3-4분 (순차 실행: 10-15분)

### 성능 특성

```
속도:     ★★★★★ (매우 빠름 - 병렬 작업 시)
병렬화:   ★★★★★ (최대)
완성도:   ★★★★☆ (높음)
비용:     ★★★★☆ (높음 - 다중 에이전트)
```

### 최적 사용 케이스

- 독립적인 여러 컴포넌트 생성
- 여러 서비스 동시 개발
- 대량의 반복 작업
- 시간이 중요한 프로젝트

## 3. Ralph: 끈질긴 완성 모드

Ralph는 작업이 100% 완료될 때까지 절대 포기하지 않는 실행 모드입니다.

### 특징

- **자동 재시도**: 오류 발생 시 무제한 재시도
- **자가 수정**: 문제를 스스로 진단하고 해결
- **의존성 해결**: 누락된 의존성 자동 설치
- **완전한 완성**: 테스트 통과까지 보장

### Magic Keywords

```bash
> ralph: [task description]
> persistent: [task description]
> complete: [task description]
```

### Ralph의 특별한 점

Ralph 모드는 자동으로 Ultrawork를 포함합니다:

```bash
> ralph: build entire e-commerce platform
```

실제로는 다음과 같이 실행됩니다:
```
ralph: (
  ultrawork: build entire e-commerce platform
) + persistence + auto-retry + self-healing
```

### 사용 시나리오

#### 시나리오 1: 복잡한 마이그레이션

```bash
> ralph: migrate the entire codebase from JavaScript to TypeScript
```

**Ralph의 끈질김:**
```
시도 1: 파일 변환 시작
  → Error: Type conflicts in utils/helpers.js
  → Ralph: 타입 정의 추가 및 재시도

시도 2: 변환 계속
  → Error: Missing type declarations for dependencies
  → Ralph: @types 패키지 자동 설치

시도 3: 변환 완료
  → Error: 23 type errors
  → Ralph: 각 오류 분석 및 수정

시도 4: 전체 빌드
  → Success: 모든 파일 변환 완료, 타입 검사 통과
  → Ralph: 테스트 실행

시도 5: 테스트
  → 3 tests failing
  → Ralph: 테스트 코드 업데이트

최종: ✓ 100% 완성 - 모든 파일 변환, 빌드 성공, 테스트 통과
```

**일반 모드:** 중간에 멈춤, 사용자 개입 필요
**Ralph 모드:** 완전 자동 완성

#### 시나리오 2: 프로덕션 준비

```bash
> ralph: make this app production-ready with all best practices
```

**Ralph의 체크리스트:**
```
✓ 환경 변수 설정
✓ 보안 헤더 추가
✓ Rate limiting 구현
✓ 에러 핸들링 강화
✓ 로깅 시스템 설정
✓ 모니터링 도구 통합
✓ CI/CD 파이프라인 구축
✓ Docker 컨테이너화
✓ Kubernetes 매니페스트 작성
✓ 성능 최적화
✓ 보안 감사 통과
✓ 부하 테스트 통과
✓ 문서 완성

작업 시간: 30-45분
재시도 횟수: 평균 12회
최종 결과: 프로덕션 배포 가능한 완성된 앱
```

#### 시나리오 3: 레거시 리팩토링

```bash
> ralph: refactor this legacy codebase to modern standards
```

**Ralph의 집요함:**
```
Phase 1: 분석
  ✓ 코드베이스 구조 파악
  ✓ 기술 부채 식별
  ✓ 리팩토링 계획 수립

Phase 2: 리팩토링
  ✓ 오래된 패턴 현대화
  ✓ 중복 코드 제거
  ✓ 모듈화 개선
  → 여러 번의 재시도를 통해 모든 이슈 해결

Phase 3: 테스트
  ✓ 기존 테스트 모두 통과
  ✓ 새로운 테스트 추가
  ✓ 커버리지 80% 이상 달성

Phase 4: 검증
  ✓ 성능 저하 없음 확인
  ✓ 모든 기능 정상 작동
  ✓ 코드 품질 메트릭 개선

최종: ✓ 완전히 현대화된 코드베이스
```

### 성능 특성

```
속도:     ★★★☆☆ (느림 - 재시도 포함)
병렬화:   ★★★★★ (Ultrawork 포함)
완성도:   ★★★★★ (완벽 - 100% 보장)
비용:     ★★★★☆ (높음 - 재시도로 인해)
```

### 주의사항

- 작업 시간이 길어질 수 있음
- API 비용이 높을 수 있음
- 간단한 작업에는 과도함

## 4. Ultrapilot: 멀티컴포넌트 시스템

Ultrapilot은 대규모 멀티컴포넌트 시스템을 구축할 때 사용하는 모드입니다.

### 특징

- **시스템 수준 설계**: 전체 아키텍처 자동 설계
- **컴포넌트 조율**: 여러 컴포넌트 간 통합 자동화
- **3-5배 속도 향상**: Autopilot 대비 대폭 빠름
- **통합 테스트**: 전체 시스템 테스트 자동 실행

### Magic Keywords

```bash
> ultrapilot: [system description]
> bigproject: [system description]
> fullstack: [system description]
```

### 사용 시나리오

#### 시나리오 1: 전체 스택 앱 구축

```bash
> ultrapilot: build a social media platform with:
> - React frontend with posts, comments, likes, follows
> - Node.js backend with GraphQL API
> - PostgreSQL database
> - Redis caching
> - S3 image storage
> - Real-time notifications
```

**Ultrapilot의 오케스트레이션:**
```
Phase 1: 설계 (2분)
└─ [Architect] 전체 시스템 아키텍처 설계

Phase 2: 병렬 개발 (10분)
├─ Team Frontend (4 agents)
│  ├─ 컴포넌트 라이브러리
│  ├─ 상태 관리
│  ├─ GraphQL 클라이언트
│  └─ UI/UX
│
├─ Team Backend (4 agents)
│  ├─ GraphQL 스키마
│  ├─ Resolvers
│  ├─ 인증/인가
│  └─ API 최적화
│
└─ Team Infrastructure (3 agents)
   ├─ 데이터베이스 스키마
   ├─ Redis 설정
   └─ S3 통합

Phase 3: 통합 (3분)
├─ [Integration] 프론트엔드-백엔드 연결
├─ [Integration] 실시간 기능 구현
└─ [Integration] 스토리지 연결

Phase 4: 테스트 및 최적화 (5분)
├─ [Testing] E2E 테스트
├─ [Performance] 성능 최적화
└─ [Security] 보안 감사

총 시간: 약 20분
```

**비교:**
- Autopilot: 60-90분
- Ultrapilot: 20분
- **속도 향상: 3-4.5배**

#### 시나리오 2: 마이크로서비스 아키텍처

```bash
> ultrapilot: create a microservices architecture with:
> - API Gateway
> - 5 microservices (auth, users, products, orders, payments)
> - Service mesh
> - Distributed tracing
> - Centralized logging
```

**실행 구조:**
```
Layer 1: 인프라 (병렬)
├─ API Gateway
├─ Service Mesh (Istio)
├─ Logging (ELK)
└─ Tracing (Jaeger)

Layer 2: 서비스들 (병렬)
├─ Auth Service
├─ User Service
├─ Product Service
├─ Order Service
└─ Payment Service

Layer 3: 통합
├─ Service Discovery
├─ Circuit Breakers
├─ Load Balancing
└─ API Composition

Layer 4: 운영
├─ Monitoring
├─ Alerting
├─ Auto-scaling
└─ CI/CD
```

**완료 시간:** 25-30분

### 성능 특성

```
속도:     ★★★★★ (매우 빠름 - 대규모 프로젝트)
병렬화:   ★★★★★ (최대)
완성도:   ★★★★★ (시스템 레벨 완성)
비용:     ★★★★★ (매우 높음)
```

## 5. Ecomode: 예산 최적화 모드

Ecomode는 API 비용을 30-50% 절감하면서 품질을 유지하는 모드입니다.

### 특징

- **비용 절감**: 30-50% API 비용 감소
- **스마트 캐싱**: 이전 결과 재사용
- **경량 모델 활용**: 간단한 작업에 작은 모델 사용
- **배치 처리**: 요청 최소화

### Magic Keywords

```bash
> eco: [task description]
> ecomode: [task description]
> budget: [task description]
```

### 비용 절감 전략

#### 전략 1: 모델 계층화

```
간단한 작업        → Claude Haiku (저비용)
중간 복잡도        → Claude Sonnet (중비용)
복잡한 작업        → Claude Opus (고비용)
```

#### 전략 2: 컨텍스트 최소화

```bash
> eco: add error handling to auth.js
```

**일반 모드:**
```
로드된 파일: 전체 프로젝트 (50+ 파일)
토큰 사용: ~50,000 tokens
비용: $0.25
```

**Ecomode:**
```
로드된 파일: auth.js와 직접 관련 파일만 (3 파일)
토큰 사용: ~5,000 tokens
비용: $0.025
절감: 90%
```

#### 전략 3: 결과 캐싱

```bash
> eco: add similar components for ProductCard, UserCard, OrderCard
```

**Ecomode의 스마트함:**
```
1. ProductCard 생성 (API 호출)
2. 패턴 캐싱
3. UserCard 생성 (캐시 사용, API 호출 최소)
4. OrderCard 생성 (캐시 사용, API 호출 최소)

API 비용: 1회분 + α (대신 3회분)
절감: ~60%
```

### 사용 시나리오

#### 시나리오 1: 대량 리팩토링

```bash
> eco: refactor all 50+ components to use hooks instead of class components
```

**비용 비교:**
```
일반 모드:
  - 각 컴포넌트마다 전체 컨텍스트 로드
  - API 호출: 50회
  - 비용: ~$12.50

Ecomode:
  - 첫 컴포넌트로 패턴 학습
  - 나머지는 패턴 재사용
  - API 호출: 5-10회
  - 비용: ~$5-6

절감: 52-60%
```

#### 시나리오 2: 문서화

```bash
> eco: generate documentation for all API endpoints
```

**Ecomode 최적화:**
```
1. 첫 엔드포인트 문서 생성 (템플릿 학습)
2. 템플릿 재사용하여 나머지 문서 생성
3. 최소한의 API 호출

비용 절감: ~40%
```

### 성능 특성

```
속도:     ★★★☆☆ (보통 - 최적화로 인한 지연)
병렬화:   ★★☆☆☆ (제한적)
완성도:   ★★★★☆ (높음)
비용:     ★☆☆☆☆ (매우 저렴)
```

## 6. Swarm: 조율된 병렬 작업

Swarm은 여러 에이전트가 하나의 목표를 향해 협력하는 모드입니다.

### 특징

- **협력적 실행**: 에이전트 간 실시간 통신
- **동적 작업 분배**: 작업 상황에 따라 재분배
- **공유 컨텍스트**: 모든 에이전트가 진행 상황 공유
- **적응형 전략**: 팀 전략 실시간 조정

### Magic Keywords

```bash
> swarm: [task description]
> team: [task description]
```

### 사용 시나리오

#### 시나리오: 대규모 테스트 작성

```bash
> swarm: write comprehensive tests for the entire application
```

**Swarm의 협력:**
```
[Queen Agent] 테스트 전략 수립
    ↓
[Worker Agents] 작업 분담
├─ Agent 1: Unit tests for models
├─ Agent 2: Unit tests for controllers
├─ Agent 3: Integration tests
├─ Agent 4: E2E tests
└─ Agent 5: Performance tests
    ↓
[실시간 조율]
- Agent 1이 모델 구조 발견 → 다른 에이전트에게 공유
- Agent 3이 API 변경 감지 → 관련 테스트 자동 조정
- Agent 4가 버그 발견 → Agent 2에게 알림
    ↓
[Queen Agent] 모든 테스트 통합 및 검증
```

### 성능 특성

```
속도:     ★★★★☆ (빠름)
병렬화:   ★★★★☆ (높음 - 조율 필요)
완성도:   ★★★★★ (매우 높음 - 협력으로 품질 향상)
비용:     ★★★★☆ (높음)
```

## 7. Pipeline: 순차 다단계 처리

Pipeline은 명확한 단계가 있는 작업을 순차적으로 처리합니다.

### 특징

- **단계별 실행**: 각 단계가 완료된 후 다음 진행
- **검증 게이트**: 각 단계 후 검증
- **롤백 지원**: 실패 시 이전 단계로 복귀
- **진행 추적**: 단계별 진행 상황 명확히 표시

### Magic Keywords

```bash
> pipeline: [task with stages]
> stages: [task with stages]
> plan: [task with stages]
```

### 사용 시나리오

#### 시나리오: 점진적 마이그레이션

```bash
> pipeline: migrate from MongoDB to PostgreSQL
> Stages:
> 1. Schema analysis and design
> 2. PostgreSQL schema creation
> 3. Data export from MongoDB
> 4. Data transformation
> 5. Data import to PostgreSQL
> 6. Validation and testing
> 7. Switch application to PostgreSQL
> 8. Cleanup
```

**Pipeline 실행:**
```
Stage 1: Schema Analysis ████████████ 100%
  ✓ MongoDB 스키마 분석 완료
  ✓ PostgreSQL 스키마 설계 완료
  → Validation passed, proceeding to Stage 2

Stage 2: PostgreSQL Setup ████████████ 100%
  ✓ 데이터베이스 생성
  ✓ 테이블 생성
  ✓ 인덱스 설정
  → Validation passed, proceeding to Stage 3

Stage 3: Data Export █████████████ 100%
  ✓ 1,250,000 documents exported
  ✓ Export integrity verified
  → Validation passed, proceeding to Stage 4

... (각 단계 순차 실행)

Stage 8: Cleanup ████████████ 100%
  ✓ 임시 파일 삭제
  ✓ 백업 확인
  ✓ 문서 업데이트
  → Migration complete!
```

### 성능 특성

```
속도:     ★★☆☆☆ (느림 - 순차 실행)
병렬화:   ★☆☆☆☆ (없음)
완성도:   ★★★★★ (매우 높음 - 검증 단계)
비용:     ★★★☆☆ (보통)
```

## Magic Keywords 전체 정리

### 기본 모드
```bash
autopilot:    빠른 자율 실행
ultrawork:    최대 병렬 처리
ralph:        끈질긴 완성 (Ultrawork 포함)
ultrapilot:   멀티컴포넌트 시스템
ecomode:      비용 최적화
swarm:        협력적 병렬 작업
pipeline:     순차 다단계 처리
```

### 짧은 별칭
```bash
auto:         = autopilot
ulw:          = ultrawork
eco:          = ecomode
go:           = autopilot
team:         = swarm
stages:       = pipeline
```

### 조합 키워드
```bash
ralplan:      ralph + pipeline (끈질긴 + 단계별)
plan:         작업 계획만 수립 (실행 안 함)
```

### 조합 예제

#### Ralplan: 안전한 대규모 작업

```bash
> ralplan: complete system rewrite to microservices
> Stages:
> 1. Design microservice architecture
> 2. Extract domain services
> 3. Implement service mesh
> 4. Migrate data
> 5. Deploy and test
```

**Ralplan = Ralph + Pipeline:**
- 각 단계를 반드시 완료 (Ralph)
- 단계별로 순차 진행 (Pipeline)
- 검증 후 다음 단계로 (Pipeline)
- 실패 시 재시도 (Ralph)

**최적 사용:**
- 실패가 허용되지 않는 작업
- 명확한 단계가 있는 큰 프로젝트
- 프로덕션 마이그레이션

## 모드별 성능 비교

### 속도 비교 (동일 작업 기준)

```
작업: 5개 마이크로서비스 구축

Autopilot:   60분 (순차)
Ultrawork:   12분 (병렬)
Ralph:       15분 (병렬 + 재시도)
Ultrapilot:  10분 (최적화된 병렬)
Ecomode:     70분 (순차 + 최적화)
Swarm:       14분 (협력적 병렬)
Pipeline:    65분 (검증 포함 순차)
Ralplan:     18분 (검증 + 재시도 포함)
```

### 비용 비교 (API 호출 기준)

```
작업: 전체 앱 리팩토링

Autopilot:   $10.00 (100%)
Ultrawork:   $15.00 (150% - 더 많은 에이전트)
Ralph:       $18.00 (180% - 재시도 포함)
Ultrapilot:  $20.00 (200% - 최대 병렬)
Ecomode:     $5.00  (50% - 최적화)
Swarm:       $16.00 (160% - 협력 오버헤드)
Pipeline:    $12.00 (120% - 검증 오버헤드)
Ralplan:     $22.00 (220% - 재시도 + 검증)
```

### 완성도 비교

```
Autopilot:   ★★★★☆ 95% 완성
Ultrawork:   ★★★★☆ 95% 완성
Ralph:       ★★★★★ 100% 완성 (보장)
Ultrapilot:  ★★★★★ 98% 완성
Ecomode:     ★★★★☆ 93% 완성
Swarm:       ★★★★★ 97% 완성
Pipeline:    ★★★★★ 99% 완성 (검증됨)
Ralplan:     ★★★★★ 100% 완성 (검증됨)
```

## 모드 선택 결정 트리

```
시작
  │
  ├─ 비용이 가장 중요한가?
  │   └─ YES → Ecomode
  │
  ├─ 반드시 100% 완성되어야 하는가?
  │   └─ YES → Ralph 또는 Ralplan
  │
  ├─ 병렬 처리 가능한 독립적 작업인가?
  │   └─ YES → Ultrawork
  │
  ├─ 대규모 멀티컴포넌트 시스템인가?
  │   └─ YES → Ultrapilot
  │
  ├─ 명확한 순차 단계가 있는가?
  │   └─ YES → Pipeline
  │
  ├─ 에이전트 간 협력이 필요한가?
  │   └─ YES → Swarm
  │
  └─ 일반적인 빠른 작업인가?
      └─ YES → Autopilot
```

## 실전 활용 팁

### 1. 프로젝트 시작

```bash
# 새 프로젝트 빠른 시작
> autopilot: initialize a new React + Node.js project with best practices

# 대규모 프로젝트
> ultrapilot: create full-stack e-commerce platform
```

### 2. 개발 중

```bash
# 새 기능 추가 (빠르게)
> auto: add shopping cart functionality

# 여러 기능 동시 개발
> ulw: add user profiles, notifications, and settings pages

# 비용 절약하며 개발
> eco: implement all CRUD operations for the new entities
```

### 3. 리팩토링

```bash
# 빠른 리팩토링
> auto: refactor utils module to use modern ES6+ features

# 대규모 리팩토링 (완전 완성 필요)
> ralph: refactor entire codebase to TypeScript

# 단계별 안전한 리팩토링
> pipeline: gradually migrate from REST to GraphQL
```

### 4. 프로덕션 준비

```bash
# 완벽한 준비 (100% 완성 보장)
> ralph: make this app production-ready

# 빠른 배포 준비
> auto: add Docker support and CI/CD pipeline

# 예산 고려하며 준비
> eco: optimize and prepare for production deployment
```

## 다음 단계

이제 7가지 실행 모드를 모두 이해했습니다! 다음 내용을 학습하세요:

- **챕터 4: 고급 활용법** (예정) - 커스텀 워크플로우, 에이전트 조합, 최적화 전략
- **챕터 5: 실전 프로젝트** (예정) - 실제 프로젝트에서 OMC 활용하기
- **챕터 6: 팀 협업** (예정) - 팀 환경에서 OMC 도입 및 활용

## 참고 자료

- GitHub 저장소: [https://github.com/Yeachan-Heo/oh-my-claudecode](https://github.com/Yeachan-Heo/oh-my-claudecode)
- 실행 모드 비교표: [Wiki 링크 추가 필요]
- 성능 벤치마크: [Wiki 링크 추가 필요]
- 커뮤니티 베스트 프랙티스: [Forum 링크 추가 필요]
