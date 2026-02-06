---
layout: post
title: "code-server 완벽 가이드 (07) - 프록시 시스템"
date: 2026-02-06
permalink: /code-server-guide-07-proxy/
author: Coder
categories: [웹 개발, 원격 개발]
tags: [code-server, 프록시, 웹 서비스, React, Vue, Angular]
original_url: "https://github.com/coder/code-server"
excerpt: "code-server의 프록시 시스템으로 개발 서버를 브라우저에 노출하는 방법."
---

## 프록시 시스템이란?

code-server에서 개발 중인 웹 애플리케이션(React, Vue 등)을 code-server의 인증을 통해 브라우저에서 접근할 수 있게 해주는 기능입니다.

```
개발 서버 (localhost:3000)
       ↓
code-server 프록시
       ↓
브라우저 (https://code.yourdomain.com/proxy/3000)
```

---

## 프록시 방법 3가지

### 1. 서브패스 프록시 (/proxy/PORT)

가장 간단한 방법. 설정 불필요.

**URL 형식:**
```
https://code.yourdomain.com/proxy/<PORT>/
```

**예시:**
```bash
# React 개발 서버 시작
npm start  # localhost:3000

# 브라우저 접속
https://code.yourdomain.com/proxy/3000/
```

### 2. 절대 경로 프록시 (/absproxy/PORT)

경로를 그대로 유지합니다.

**URL 형식:**
```
https://code.yourdomain.com/absproxy/<PORT>/
```

### 3. 서브도메인 프록시

가장 깔끔한 방법. 도메인 설정 필요.

**설정:**
```bash
code-server --proxy-domain yourdomain.com
```

**URL 형식:**
```
https://3000.yourdomain.com
```

---

## 서브패스 프록시 (/proxy/PORT)

### 기본 사용법

```bash
# 개발 서버 시작
cd my-app
npm start  # localhost:3000

# 브라우저 접속
https://code.yourdomain.com/proxy/3000/
```

**중요**: **마지막 슬래시(/) 필수!**

```
✅ https://code.yourdomain.com/proxy/3000/
❌ https://code.yourdomain.com/proxy/3000
```

### 경로 처리

code-server는 `/proxy/<PORT>` 부분을 자동으로 제거합니다.

```
요청: /proxy/3000/api/users
→ 프록시: http://localhost:3000/api/users
```

이는 대부분의 프레임워크가 상대 경로를 사용하기 때문입니다.

---

## 절대 경로 프록시 (/absproxy/PORT)

경로를 그대로 유지합니다. Create React App 등에 필요.

### 사용법

```
https://code.yourdomain.com/absproxy/3000/my-app-path
```

```
요청: /absproxy/3000/my-app-path
→ 프록시: http://localhost:3000/my-app-path
```

---

## 프레임워크별 설정

### React (Create React App)

**문제**: CRA는 `/absproxy` 필요

**해결:**
```bash
# .env.local
PUBLIC_URL=/absproxy/3000
WDS_SOCKET_PATH=$PUBLIC_URL/sockjs-node
BROWSER=none

# 시작
npm start

# 접속
https://code.yourdomain.com/absproxy/3000/
```

또는 package.json:
```json
{
  "homepage": "/absproxy/3000",
  "scripts": {
    "start": "BROWSER=none PUBLIC_URL=/absproxy/3000 WDS_SOCKET_PATH=/absproxy/3000/sockjs-node react-scripts start"
  }
}
```

### Vue

```js
// vue.config.js
module.exports = {
  devServer: {
    port: 3000,
    sockPath: "sockjs-node",
  },
  publicPath: "/absproxy/3000",
}
```

```bash
npm run serve

# 접속
https://code.yourdomain.com/absproxy/3000/
```

### Angular

```json
// package.json
{
  "scripts": {
    "start": "ng serve --serve-path /absproxy/4200"
  }
}
```

```html
<!-- src/index.html -->
<base href="/./">
```

```bash
npm start

# 접속
https://code.yourdomain.com/absproxy/4200/
```

### Svelte (SvelteKit)

```js
// svelte.config.js
const config = {
  kit: {
    paths: {
      base: "/absproxy/5173",
    },
  },
}

export default config
```

```bash
npm run dev

# 접속
https://code.yourdomain.com/absproxy/5173/
```

### Next.js

```js
// next.config.js
module.exports = {
  basePath: "/absproxy/3000",
  assetPrefix: "/absproxy/3000",
}
```

```bash
npm run dev

# 접속
https://code.yourdomain.com/absproxy/3000/
```

### Vite

```js
// vite.config.js
export default {
  base: "/absproxy/5173/",
  server: {
    port: 5173,
  },
}
```

### Express.js (Node.js)

```js
// server.js
const express = require('express')
const app = express()

// 프록시 경로를 고려한 라우팅
const basePath = process.env.BASE_PATH || ''
app.use(basePath, express.static('public'))

app.listen(3000)
```

```bash
BASE_PATH=/absproxy/3000 node server.js
```

---

## 서브도메인 프록시

### 설정

**DNS 와일드카드 레코드:**
```
*.yourdomain.com → your-server-ip
```

**code-server 설정:**
```yaml
# config.yaml
proxy-domain: yourdomain.com
```

또는:
```bash
code-server --proxy-domain yourdomain.com
```

### 사용

```bash
# 개발 서버 시작
npm start  # localhost:3000

# 브라우저 접속
https://3000.yourdomain.com
```

### 장점

- 깔끔한 URL
- 프레임워크 설정 불필요
- CORS 이슈 없음

### 리버스 프록시 설정 (필수)

**Caddy:**
```
*.yourdomain.com {
  reverse_proxy 127.0.0.1:8080
}

yourdomain.com {
  reverse_proxy 127.0.0.1:8080
}
```

**NGINX:**
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com *.yourdomain.com;

    # 와일드카드 인증서
    ssl_certificate /path/to/wildcard-cert.pem;
    ssl_certificate_key /path/to/wildcard-key.pem;

    location / {
        proxy_pass http://localhost:8080/;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
    }
}
```

---

## 커스텀 프록시 URI

환경 변수로 프록시 URL 패턴 변경:

```bash
export VSCODE_PROXY_URI="https://{{port}}.dev.mycompany.com"
code-server
```

또는 상대 경로:
```bash
export VSCODE_PROXY_URI="./proxy/{{port}}"
```

**효과**: 포트 패널에 표시되는 URL이 변경됨.

---

## Preflight 요청 처리

CORS preflight 요청은 credentials를 포함하지 않아 인증 실패할 수 있습니다.

**해결:**
```bash
code-server --skip-auth-preflight
```

모든 preflight 요청을 인증 없이 허용.

---

## 서브패스 프록시 고급

### `--abs-proxy-base-path`

code-server를 서브패스에서 실행하면서 absproxy 사용:

```bash
# code-server를 /workspace에서 실행
code-server --abs-proxy-base-path=/user/123/workspace
```

**예시:**
```
https://mysite.com/user/123/workspace/  # code-server
https://mysite.com/user/123/workspace/absproxy/3000/  # 개발 서버
```

---

## 포트 자동 감지

code-server는 개발 서버가 시작되면 포트 패널에 자동으로 표시합니다.

**포트 패널 위치:**
```
View → Ports  (또는 Ctrl+Shift+P → "Ports")
```

**기능:**
- 자동 감지
- 브라우저에서 열기
- 포트 포워딩
- 접근 권한 설정 (Public/Private)

---

## 프록시 디버깅

### 1. 개발 서버가 실행 중인지 확인

```bash
# 서버에서
curl http://localhost:3000
```

### 2. code-server 로그 확인

```bash
code-server --log debug
```

### 3. 브라우저 개발자 도구

- Network 탭에서 프록시 요청 확인
- Console에서 WebSocket 연결 상태 확인

### 4. 프록시 URL 테스트

```bash
# 직접 접근
curl https://code.yourdomain.com/proxy/3000/
```

---

## 일반적인 문제

### 문제 1: 404 Not Found

**원인**: 경로 처리 이슈

**해결**:
- `/proxy/3000/` 마지막 슬래시 확인
- 프레임워크 `publicPath` 설정 확인

### 문제 2: WebSocket 연결 실패

**원인**: HMR (Hot Module Replacement) 경로 이슈

**해결** (CRA):
```bash
WDS_SOCKET_PATH=/absproxy/3000/sockjs-node npm start
```

### 문제 3: CSS/JS 파일 로드 실패

**원인**: 절대 경로 참조

**해결**: 프레임워크 `publicPath` 설정

```js
// webpack.config.js
module.exports = {
  output: {
    publicPath: "/absproxy/3000/",
  },
}
```

### 문제 4: CORS 오류

**원인**: 다른 포트 간 요청

**해결**: 서브도메인 프록시 사용 또는 CORS 설정

```js
// Express.js
const cors = require('cors')
app.use(cors({
  origin: 'https://code.yourdomain.com',
  credentials: true,
}))
```

---

## 베스트 프랙티스

### 1. 개발 환경에 맞는 프록시 선택

- **간단한 앱**: `/proxy/PORT` 사용
- **복잡한 앱 (CRA, Next.js)**: `/absproxy/PORT` 사용
- **프로덕션 비슷한 환경**: 서브도메인 프록시

### 2. 환경 변수 활용

```bash
# .env
REACT_APP_API_URL=/proxy/8000/api
```

### 3. Docker Compose로 통합

```yaml
version: "3"
services:
  code-server:
    image: codercom/code-server:latest
    ports:
      - "8080:8080"
    environment:
      - VSCODE_PROXY_URI=https://{{port}}.dev.local

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"

  backend:
    build: ./backend
    ports:
      - "8000:8000"
```

---

## 프록시와 인증

프록시된 앱은 code-server의 인증을 상속받습니다.

```
사용자 → code-server (인증) → 프록시 → 개발 서버
```

**장점**: 개발 서버에 별도 인증 불필요

---

*다음 글에서는 iPad, Termux, Android, iOS 등 다양한 플랫폼에서 code-server를 사용하는 방법을 알아봅니다.*
