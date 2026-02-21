---
layout: post
title: "PentAGI 가이드 (05) - 백엔드 구조: Go 서버와 GraphQL"
date: 2026-02-21
permalink: /pentagi-guide-05-backend-architecture/
author: PentAGI Team
categories: ['AI 에이전트', '보안']
tags: [PentAGI, Go, Gin, GraphQL, PostgreSQL, sqlc, gqlgen]
original_url: "https://github.com/vxcontrol/pentagi"
excerpt: "backend/를 중심으로 서버 엔트리포인트, 라우팅, GraphQL(구독), DB 접근 방식을 구조적으로 정리합니다."
---

## 백엔드의 역할

PentAGI 백엔드는 단순 API 서버가 아니라 다음을 한꺼번에 수행합니다.

- **Flow/Task/Subtask** 같은 작업 모델 관리
- LLM provider 호출 및 응답 처리
- Docker 실행 환경 오케스트레이션(컨테이너 생성/exec/정리)
- 메시지/툴/검색/터미널 로그를 DB에 저장
- GraphQL(구독 포함)로 프론트에 실시간 전달

---

## 저장소 지도: `backend/`에서 먼저 볼 것

```text
backend/
├─ cmd/                 # 실행 엔트리포인트들(pentagi, ctester, ftester, ...)
├─ pkg/
│  ├─ config/           # .env → Config
│  ├─ server/           # Gin router + middleware + HTTP 서비스
│  ├─ graph/            # gqlgen 리졸버/스키마(실시간 구독 포함)
│  ├─ controller/       # Flow/Task/Assistant 실행 상태/오케스트레이션
│  ├─ docker/           # Docker client(컨테이너 생성/exec/파일 복사)
│  ├─ providers/        # OpenAI/Anthropic/Gemini/Ollama/Custom 등
│  ├─ tools/            # 검색/브라우저/메모리 등 “함수 호출” 툴
│  └─ database/         # sqlc/gorm 연결 및 모델
├─ migrations/          # goose 마이그레이션
└─ sqlc/                # sqlc 설정/스키마
```

---

## `cmd/pentagi/main.go`: 부팅 순서

부팅 순서는 운영 디버깅에 매우 중요합니다. 흐름을 요약하면:

```text
Config(.env)
 → Observability(Langfuse/OTEL, 옵션)
 → DB 연결 + 마이그레이션(goose)
 → Docker client
 → Provider controller
 → Flow controller(기존 Flow 로드)
 → Gin router 기동
```

여기서 핵심 포인트는 2가지입니다.

1) 서버 부팅 시 **마이그레이션이 자동 실행**됩니다.  
2) PentAGI는 “LLM 호출”뿐 아니라 “Docker 실행”이 필수이므로, Docker client 초기화가 핵심 경로에 있습니다.

---

## `pkg/config`: `.env`가 사실상 API다

`pkg/config/config.go`는 `.env`를 읽어 `Config` 구조체로 파싱합니다.

- LLM/임베딩 provider 정보
- 검색 엔진 API 키
- Docker 실행 관련 옵션
- 쿠키 서명/SSL/OAuth 같은 보안 설정
- (옵션) Langfuse/OTEL/Graphiti 연동

즉, 백엔드의 기능 대부분은 “코드 수정”이 아니라 **환경 변수 변경**으로 켜고 끄는 구조입니다.

---

## `pkg/server/router.go`: Gin 라우팅과 서비스 바인딩

라우터는 `/api/v1` 아래에 서비스들을 묶고, 인증/세션/쿠키 및 CORS를 구성합니다.

특히 눈여겨볼 점:

- 프론트 SPA 라우트(`/chat`, `/flows`, `/settings` 등)도 서버에서 인지해 정적 제공/리버스 프록시로 처리할 수 있도록 설계
- `services/` 하위로 도메인별 서비스(Auth/User/Flow/Logs/Analytics 등)가 분리
- GraphQL playground/Swagger 같은 개발자 도구도 제공

---

## GraphQL(구독 포함): `pkg/server/services/graphql.go`

PentAGI는 “상태가 계속 변하는 시스템”을 UI에 보여줘야 하므로, 구독(WebSocket)이 중요합니다.

`GraphqlService`에서 확인할 수 있는 설계 포인트:

- gqlgen 기반 스키마/리졸버 사용
- HTTP(GET/POST) + WebSocket 전송을 함께 활성화
- Origin 검증(허용 목록/와일드카드)로 구독 연결을 제어
- WebSocket init 시 사용자 인증 정보를 컨텍스트에 주입

---

## DB 접근: `sqlc` + `gorm`의 조합

`main.go`에서 두 가지 DB 접근 방식이 함께 등장합니다.

- `database.New(db)` → sqlc 기반의 타입 안전 쿼리(Queries)
- `database.NewGorm(...)` → Gorm ORM(서비스 레이어에서 활용)

이 조합은 흔히 다음의 타협점을 제공합니다.

- 복잡/핵심 경로는 sqlc로 예측 가능성과 성능 확보
- 관리/화면 중심의 CRUD는 ORM으로 개발 속도 확보

---

## Flow/Task/Assistant 실행: `pkg/controller`

`pkg/controller`는 PentAGI의 “실행 엔진”에 가깝습니다.

대략의 계층 구조는 다음처럼 이해하면 편합니다.

```text
Flow
 ├─ Task (큰 단계)
 │   └─ Subtask (세부 실행)
 └─ Assistants (추가 대화/역할 단위 실행)
```

컨트롤러는 provider/도커/툴 실행기를 엮어 실행 컨텍스트를 만들고,
각 단계의 메시지/툴 호출/검색/터미널 로그를 DB에 남기며,
구독 퍼블리셔를 통해 UI에 이벤트를 발행합니다.

---

## 참고 링크

- 인덱스(전체 목차): `/blog-repo/pentagi-guide/`
- 엔트리포인트: `backend/cmd/pentagi/main.go`
- 라우터: `backend/pkg/server/router.go`
- GraphQL 서비스: `backend/pkg/server/services/graphql.go`

---

다음 글에서는 프론트엔드(React/Vite)가 GraphQL/구독을 어떻게 써서 UI를 구성하는지 살펴봅니다.

