---
layout: post
title: "code-server 완벽 가이드 (03) - 아키텍처 분석"
date: 2026-02-06
permalink: /code-server-guide-03-architecture/
author: Coder
categories: [웹 개발, 원격 개발]
tags: [code-server, 아키텍처, VS Code, Node.js, TypeScript]
original_url: "https://github.com/coder/code-server"
excerpt: "code-server의 내부 아키텍처와 VS Code 통합 방식을 분석합니다."
---

## code-server 아키텍처 개요

code-server는 **VS Code를 웹 서버로 감싸는 래퍼(wrapper)** 구조입니다.

```
┌──────────────────────────────────────────────────────────┐
│                     클라이언트                            │
│                  (브라우저)                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │         VS Code UI (HTML/JS/CSS)                 │   │
│  │  Extensions │ Editor │ Terminal │ Debug           │   │
│  └─────────────────────┬────────────────────────────┘   │
└────────────────────────┼─────────────────────────────────┘
                         │
                    WebSocket / HTTP
                         │
┌────────────────────────┼─────────────────────────────────┐
│                   code-server                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │          Node.js HTTP Server                     │   │
│  │  (Express.js + WebSocket)                        │   │
│  ├──────────────────────────────────────────────────┤   │
│  │          VS Code Backend                         │   │
│  │  (File System, Terminal, Extensions, LSP)        │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 프로젝트 디렉토리 구조

```
code-server/
├── ci/                       # CI/CD & 빌드 시스템
│   ├── build/                # 빌드 스크립트
│   │   ├── build-code-server.sh
│   │   ├── build-vscode.sh
│   │   └── build-release.sh
│   ├── dev/                  # 개발 도구
│   │   ├── watch.ts          # 개발 모드 watcher
│   │   ├── test-*.sh         # 테스트 스크립트
│   │   └── gen_icons.sh
│   ├── helm-chart/           # Kubernetes Helm 차트
│   ├── steps/                # CI 단계별 스크립트
│   │   ├── publish-npm.sh
│   │   └── docker-buildx-push.sh
│   └── release-image/        # Docker 이미지 빌드
│
├── docs/                     # 문서
│   ├── README.md             # 프로젝트 소개
│   ├── guide.md              # 설정 가이드
│   ├── install.md            # 설치 방법
│   ├── FAQ.md                # 자주 묻는 질문
│   ├── CONTRIBUTING.md       # 기여 가이드
│   ├── MAINTAINING.md        # 유지보수 가이드
│   └── ...
│
├── lib/                      # 외부 라이브러리
│   └── vscode/               # VS Code 소스 (git submodule)
│
├── patches/                  # VS Code 패치 파일들
│   ├── marketplace.diff      # 확장 마켓플레이스 패치
│   ├── disable-builtin-ext-update.diff
│   └── ...
│
├── src/                      # code-server 소스 코드
│   └── node/                 # Node.js 서버 코드
│       ├── entry.ts          # 진입점
│       ├── app.ts            # Express 앱
│       ├── cli.ts            # CLI 파서
│       ├── http.ts           # HTTP 서버
│       ├── routes/           # 라우터
│       │   ├── index.ts
│       │   ├── login.ts      # 로그인 페이지
│       │   ├── static.ts     # 정적 파일
│       │   └── vscode.ts     # VS Code 프록시
│       ├── proxy.ts          # 프록시 시스템
│       ├── socket.ts         # WebSocket 처리
│       ├── settings.ts       # 설정 관리
│       ├── i18n/             # 국제화
│       │   └── locales/
│       └── plugin.ts         # 플러그인 시스템
│
├── test/                     # 테스트
│   ├── unit/                 # 단위 테스트
│   ├── integration/          # 통합 테스트
│   ├── e2e/                  # E2E 테스트 (Playwright)
│   └── scripts/              # 테스트 스크립트
│
├── typings/                  # TypeScript 타입 정의
│
├── install.sh                # 자동 설치 스크립트
├── package.json              # npm 설정
├── tsconfig.json             # TypeScript 설정
└── eslint.config.mjs         # ESLint 설정
```

---

## 핵심 컴포넌트

### 1. VS Code 통합

code-server는 VS Code를 **git submodule**로 포함합니다:

```bash
# lib/vscode는 VS Code 원본 레포지토리
git submodule update --init
```

**VS Code 패치 적용:**

`patches/` 디렉토리에 있는 .diff 파일들로 VS Code를 수정:

- **marketplace.diff**: Open VSX Registry 사용
- **disable-builtin-ext-update.diff**: 내장 확장 자동 업데이트 비활성화
- **local-storage.diff**: 로컬 스토리지 처리

```bash
# 패치 적용 (빌드 시 자동)
cd lib/vscode
git apply ../../patches/*.diff
```

---

### 2. HTTP 서버 (src/node/http.ts)

Express.js 기반 HTTP 서버:

```typescript
// 간소화된 구조
class HttpServer {
  private readonly server: http.Server
  private readonly app: Express

  constructor() {
    this.app = express()
    this.setupMiddlewares()
    this.setupRoutes()
    this.server = http.createServer(this.app)
  }

  setupMiddlewares() {
    // 쿠키 파서
    this.app.use(cookieParser())

    // 압축
    this.app.use(compression())

    // 바디 파서
    this.app.use(express.json())
    this.app.use(express.urlencoded({ extended: true }))
  }

  setupRoutes() {
    // 로그인 페이지
    this.app.use("/login", loginRoutes)

    // 정적 파일 (CSS, JS, 이미지)
    this.app.use("/static", staticRoutes)

    // VS Code 프록시
    this.app.use("/", vscodeRoutes)
  }
}
```

---

### 3. 인증 시스템 (src/node/routes/login.ts)

비밀번호 기반 인증:

```typescript
// 로그인 라우터 (간소화)
router.post("/login", async (req, res) => {
  const { password } = req.body
  const configPassword = await settings.getPassword()

  if (password !== configPassword) {
    // Rate limiting: 2 per minute + 12 per hour
    return res.status(401).json({ error: "Incorrect password" })
  }

  // 세션 쿠키 생성
  const sessionToken = generateToken()
  res.cookie("key", sessionToken, {
    httpOnly: true,
    secure: true,
    sameSite: "lax",
  })

  res.json({ success: true })
})

// 인증 미들웨어
function authenticate(req, res, next) {
  const token = req.cookies.key
  if (isValidToken(token)) {
    next()
  } else {
    res.redirect("/login")
  }
}
```

**Rate Limiting:**
- 분당 2회 시도
- 시간당 추가 12회 시도
- 총 실패 시 일시 차단

---

### 4. WebSocket 프록시 (src/node/socket.ts)

VS Code는 브라우저와 서버 간 실시간 통신에 WebSocket을 사용합니다:

```typescript
// WebSocket 업그레이드 처리
server.on("upgrade", (req, socket, head) => {
  // 인증 확인
  if (!isAuthenticated(req)) {
    socket.write("HTTP/1.1 401 Unauthorized\r\n\r\n")
    socket.destroy()
    return
  }

  // VS Code 백엔드로 프록시
  wss.handleUpgrade(req, socket, head, (ws) => {
    wss.emit("connection", ws, req)
  })
})
```

**사용 사례:**
- 터미널 입출력
- 파일 watcher 이벤트
- 확장 프로그램 통신
- 실시간 에디터 동기화

---

### 5. 프록시 시스템 (src/node/proxy.ts)

개발 서버(React, Vue 등)를 브라우저에 노출:

```typescript
// 프록시 라우터
app.get("/proxy/:port/*", (req, res) => {
  const { port } = req.params
  const targetUrl = `http://localhost:${port}${req.path}`

  // http-proxy로 프록시
  proxy.web(req, res, {
    target: targetUrl,
    changeOrigin: true,
  })
})

// Subdomain 프록시
// 예: 3000.mydomain.com → localhost:3000
if (proxyDomain) {
  const subdomain = req.hostname.split(".")[0]
  const port = parseInt(subdomain)
  if (!isNaN(port)) {
    proxy.web(req, res, { target: `http://localhost:${port}` })
  }
}
```

---

### 6. CLI 파서 (src/node/cli.ts)

명령줄 인자 처리:

```typescript
export interface Args {
  "bind-addr"?: string           // 127.0.0.1:8080
  "config"?: string              // ~/.config/code-server/config.yaml
  "auth"?: "password" | "none"
  "password"?: string
  "cert"?: boolean | string
  "cert-key"?: string
  "user-data-dir"?: string
  "extensions-dir"?: string
  "proxy-domain"?: string
  "verbose"?: boolean
  "log"?: "trace" | "debug" | "info" | "warn" | "error"
  // ... 더 많은 옵션
}

// 설정 파일과 CLI 인자 병합
function mergeConfig(configFile, cliArgs) {
  return {
    ...loadConfigFile(configFile),
    ...cliArgs,  // CLI 인자가 우선
  }
}
```

---

## 데이터 흐름

### 1. 사용자 로그인

```
브라우저 → GET /login
  ↓
code-server → 로그인 페이지 반환
  ↓
브라우저 → POST /login (비밀번호)
  ↓
code-server → 비밀번호 검증
  ↓
         (성공) → 세션 쿠키 생성 → 메인 페이지로 리다이렉트
         (실패) → 401 에러 반환
```

### 2. VS Code 로딩

```
브라우저 → GET / (with auth cookie)
  ↓
code-server → VS Code HTML/JS/CSS 전송
  ↓
브라우저 → VS Code UI 렌더링
  ↓
         → WebSocket 연결 (실시간 통신)
         → HTTP 요청 (파일 읽기, 확장 설치 등)
```

### 3. 파일 편집

```
브라우저 (에디터) → WebSocket: 파일 열기 요청
  ↓
code-server → 파일 시스템에서 읽기
  ↓
         → WebSocket: 파일 내용 전송
  ↓
브라우저 → 에디터에 표시
  ↓
사용자 → 파일 수정
  ↓
브라우저 → WebSocket: 변경사항 전송
  ↓
code-server → 파일 시스템에 쓰기
```

### 4. 터미널 사용

```
브라우저 → WebSocket: 터미널 생성 요청
  ↓
code-server → node-pty로 PTY 생성 (bash/zsh)
  ↓
         → WebSocket: 터미널 준비 완료
  ↓
브라우저 → 터미널 렌더링 (xterm.js)
  ↓
사용자 → 명령어 입력
  ↓
브라우저 → WebSocket: 키 입력 전송
  ↓
code-server → PTY로 전달
  ↓
         → PTY 출력 수신
         → WebSocket: 출력 전송
  ↓
브라우저 → 터미널에 출력 표시
```

---

## 빌드 시스템

### 빌드 프로세스

```bash
# 전체 빌드
npm run build

# 단계별:
# 1. VS Code 빌드
npm run build:vscode

# 2. code-server 빌드
npm run build:code-server

# 3. 패키지 생성
npm run package
```

### 빌드 산출물

```
out/
├── node/                    # Node.js 서버 코드 (compiled)
│   ├── entry.js
│   ├── app.js
│   └── ...
└── browser/                 # VS Code 프론트엔드
    ├── index.html
    ├── workbench.js
    └── ...

release/                     # 릴리스 패키지
├── code-server-4.20.0-linux-amd64.tar.gz
├── code-server_4.20.0_amd64.deb
└── code-server-4.20.0-amd64.rpm
```

---

## 설정 시스템

### 설정 우선순위

```
CLI 인자 > 환경 변수 > 설정 파일 > 기본값
```

예시:
```bash
# 설정 파일
# ~/.config/code-server/config.yaml
bind-addr: 127.0.0.1:8080
auth: password
password: mypass

# 환경 변수로 덮어쓰기
export PASSWORD=newpass

# CLI 인자로 최종 덮어쓰기
code-server --auth none --bind-addr 0.0.0.0:3000
```

---

## 확장 시스템

### VS Code 확장 호환

code-server는 **대부분의 VS Code 확장**을 지원합니다:

- **공개 레지스트리**: Open VSX Registry (기본)
- **프라이빗 레지스트리**: 환경 변수로 설정 가능

```bash
# 확장 레지스트리 변경
export EXTENSIONS_GALLERY='{"serviceUrl": "https://my-registry.com"}'
```

### 확장 설치 경로

```
~/.local/share/code-server/extensions/
├── ms-python.python-2024.0.0/
├── dbaeumer.vscode-eslint-2.4.0/
└── esbenp.prettier-vscode-10.1.0/
```

---

## 성능 최적화

### 1. HTTP 압축

```typescript
import compression from "compression"

app.use(compression({
  level: 6,  // 압축 레벨 (0-9)
  threshold: 1024,  // 1KB 이상만 압축
}))
```

### 2. 정적 파일 캐싱

```typescript
app.use("/static", express.static("out/browser", {
  maxAge: "7d",  // 7일 캐시
  etag: true,
  lastModified: true,
}))
```

### 3. WebSocket Keep-Alive

```typescript
wss.on("connection", (ws) => {
  const interval = setInterval(() => {
    if (ws.readyState === ws.OPEN) {
      ws.ping()
    }
  }, 30000)  // 30초마다 ping

  ws.on("close", () => clearInterval(interval))
})
```

---

## 보안 아키텍처

### 1. 세션 관리

- HttpOnly 쿠키 사용 (XSS 방지)
- Secure 플래그 (HTTPS 전용)
- SameSite=Lax (CSRF 방지)

### 2. Rate Limiting

```typescript
// 로그인 시도 제한
const rateLimiter = {
  perMinute: 2,
  perHour: 12,
  attempts: new Map<string, number[]>(),
}

function checkRateLimit(ip: string): boolean {
  const now = Date.now()
  const attempts = rateLimiter.attempts.get(ip) || []

  // 최근 1분
  const recentMinute = attempts.filter(t => now - t < 60000)
  if (recentMinute.length >= 2) return false

  // 최근 1시간
  const recentHour = attempts.filter(t => now - t < 3600000)
  if (recentHour.length >= 14) return false  // 2 + 12

  return true
}
```

### 3. CSP (Content Security Policy)

```typescript
app.use((req, res, next) => {
  res.setHeader("Content-Security-Policy",
    "default-src 'self'; " +
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
    "style-src 'self' 'unsafe-inline';"
  )
  next()
})
```

---

## 모니터링 및 로깅

### 로그 레벨

```typescript
enum LogLevel {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warn = 3,
  Error = 4,
}

// 사용
logger.info("Server started", { port: 8080 })
logger.error("Failed to read file", { path: "/foo", error })
```

### 상태 확인 엔드포인트

```typescript
app.get("/healthz", (req, res) => {
  res.json({
    status: "ok",
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    version: packageJson.version,
  })
})
```

---

## 개발 모드

```bash
# Watch 모드로 개발
npm run watch

# 특징:
# - TypeScript 자동 컴파일
# - 파일 변경 시 자동 재시작
# - 소스맵 활성화
# - 상세한 로깅
```

---

## 테스트 구조

```
test/
├── unit/                  # 단위 테스트 (Jest)
│   ├── http.test.ts
│   ├── proxy.test.ts
│   └── cli.test.ts
├── integration/           # 통합 테스트
│   └── login.test.ts
├── e2e/                   # E2E 테스트 (Playwright)
│   ├── terminal.test.ts
│   ├── extensions.test.ts
│   └── settings.test.ts
└── scripts/
    ├── test-unit.sh
    ├── test-integration.sh
    └── test-e2e.sh
```

---

*다음 글에서는 config.yaml 및 CLI 옵션을 통한 상세한 설정 방법을 살펴봅니다.*
