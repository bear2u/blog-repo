---
layout: post
title: "code-server 완벽 가이드 (06) - 네트워크 설정"
date: 2026-02-06
permalink: /code-server-guide-06-network/
author: Coder
categories: [웹 개발, 원격 개발]
tags: [code-server, 네트워크, SSH, Caddy, NGINX, 리버스 프록시]
original_url: "https://github.com/coder/code-server"
excerpt: "SSH 포트 포워딩과 리버스 프록시를 통한 네트워크 설정 방법."
---

## 네트워크 노출 방법 비교

| 방법 | 난이도 | 보안 | iPad 지원 | 권장도 |
|------|--------|------|-----------|--------|
| **SSH 포트 포워딩** | ⭐ | ⭐⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐⭐ |
| **Caddy** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ |
| **NGINX** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ |
| **자체 서명 인증서** | ⭐ | ⭐⭐⭐ | ❌ | ⭐⭐ |
| **HTTP (인증 없음)** | ⭐ | ❌ | ✅ | ❌ (로컬만) |

---

## 방법 1: SSH 포트 포워딩 (최고 보안)

### 기본 설정

**서버 설정:**
```bash
# code-server를 localhost에만 바인딩
code-server --bind-addr 127.0.0.1:8080 --auth none
```

**로컬 설정:**
```bash
# SSH 터널 생성
ssh -N -L 8080:127.0.0.1:8080 user@your-server.com

# 브라우저: http://localhost:8080
```

### SSH 설정 파일로 간편화

```bash
# ~/.ssh/config
Host code-server
    HostName your-server.com
    User youruser
    Port 22
    LocalForward 8080 127.0.0.1:8080
    ServerAliveInterval 5
    ExitOnForwardFailure yes
```

사용:
```bash
ssh code-server -N

# 또는 백그라운드로
ssh code-server -N -f
```

### 장점
- SSH 키 인증 활용
- 추가 설정 불필요
- 최고 수준 보안

### 단점
- iPad/모바일에서 SSH 클라이언트 필요
- 연결 유지 필요

---

## 방법 2: Caddy (권장)

### Caddy 장점
- **자동 HTTPS**: Let's Encrypt 자동 발급/갱신
- **간단한 설정**: 2-3줄로 완료
- **자동 리다이렉트**: HTTP → HTTPS

### 설치

```bash
# Debian/Ubuntu
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

### 루트 경로 설정

```bash
# /etc/caddy/Caddyfile
code.yourdomain.com {
  reverse_proxy 127.0.0.1:8080
}
```

```bash
sudo systemctl reload caddy
```

접속: `https://code.yourdomain.com`

### 서브패스 설정

```bash
# /etc/caddy/Caddyfile
yourdomain.com {
  route /code/* {
    uri strip_prefix /code
    reverse_proxy 127.0.0.1:8080
  }
}
```

접속: `https://yourdomain.com/code/`

### WebSocket 지원 (자동)

Caddy는 WebSocket을 자동으로 감지하고 프록시합니다.

### 고급 설정

```bash
code.yourdomain.com {
  # HTTPS 자동 설정
  reverse_proxy 127.0.0.1:8080 {
    # 헤더 전달
    header_up Host {host}
    header_up X-Real-IP {remote_host}
    header_up X-Forwarded-For {remote_host}
    header_up X-Forwarded-Proto {scheme}

    # 타임아웃
    transport http {
      dial_timeout 10s
      response_header_timeout 30s
    }
  }

  # 로깅
  log {
    output file /var/log/caddy/code-server.log
    format json
  }

  # 압축
  encode gzip zstd
}
```

---

## 방법 3: NGINX

### 설치

```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
```

### HTTP 설정 (Let's Encrypt 발급용)

```bash
# /etc/nginx/sites-available/code-server
server {
    listen 80;
    listen [::]:80;
    server_name code.yourdomain.com;

    location / {
        proxy_pass http://localhost:8080/;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
        proxy_set_header Accept-Encoding gzip;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

활성화:
```bash
sudo ln -s /etc/nginx/sites-available/code-server /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Let's Encrypt 인증서 발급

```bash
sudo certbot --nginx -d code.yourdomain.com --non-interactive --agree-tos -m you@example.com --redirect
```

Certbot이 자동으로 HTTPS 설정 추가!

### 최종 HTTPS 설정

Certbot 실행 후 `/etc/nginx/sites-available/code-server`:

```nginx
server {
    listen 80;
    server_name code.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name code.yourdomain.com;

    # Let's Encrypt 인증서
    ssl_certificate /etc/letsencrypt/live/code.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/code.yourdomain.com/privkey.pem;
    ssl_trusted_certificate /etc/letsencrypt/live/code.yourdomain.com/chain.pem;

    # SSL 설정
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

    location / {
        proxy_pass http://localhost:8080/;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
        proxy_set_header Accept-Encoding gzip;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 타임아웃
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

### 서브패스 설정

```nginx
location /code/ {
    proxy_pass http://localhost:8080/;
    proxy_set_header Host $http_host;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection upgrade;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # 서브패스 처리
    proxy_redirect / /code/;
    rewrite ^/code/(.*)$ /$1 break;
}
```

---

## 서브도메인 와일드카드 (프록시용)

code-server는 개발 서버를 서브도메인으로 노출할 수 있습니다.

### DNS 설정

와일드카드 A 레코드 추가:
```
*.code.yourdomain.com → your-server-ip
```

### code-server 설정

```yaml
# config.yaml
proxy-domain: code.yourdomain.com
```

```bash
code-server --proxy-domain code.yourdomain.com
```

### Caddy 설정

```bash
# /etc/caddy/Caddyfile
*.code.yourdomain.com {
  reverse_proxy 127.0.0.1:8080
}

code.yourdomain.com {
  reverse_proxy 127.0.0.1:8080
}
```

### NGINX 설정

```nginx
server {
    listen 443 ssl http2;
    server_name code.yourdomain.com *.code.yourdomain.com;

    # 와일드카드 인증서
    ssl_certificate /etc/letsencrypt/live/code.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/code.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8080/;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
    }
}
```

### 와일드카드 Let's Encrypt

```bash
sudo certbot certonly --manual --preferred-challenges dns -d "*.code.yourdomain.com" -d "code.yourdomain.com"
```

DNS TXT 레코드 추가 필요:
```
_acme-challenge.code.yourdomain.com TXT "validation-string"
```

### 사용 예

- 메인: `https://code.yourdomain.com`
- React (포트 3000): `https://3000.code.yourdomain.com`
- API (포트 8000): `https://8000.code.yourdomain.com`

---

## 방법 4: 자체 서명 인증서

### 생성 및 사용

```bash
# config.yaml
bind-addr: 0.0.0.0:443
auth: password
password: strongPassword
cert: true
```

```bash
# 포트 443 권한
sudo setcap cap_net_bind_service=+ep $(which node)

# 시작
code-server
```

### mkcert로 신뢰할 수 있는 인증서

```bash
# mkcert 설치
brew install mkcert  # macOS
# 또는
sudo apt install libnss3-tools
curl -JLO "https://dl.filippo.io/mkcert/latest?for=linux/amd64"
chmod +x mkcert-v*-linux-amd64
sudo mv mkcert-v*-linux-amd64 /usr/local/bin/mkcert

# 로컬 CA 설치
mkcert -install

# 인증서 생성
mkcert code.local localhost 127.0.0.1 ::1

# code-server 설정
code-server --cert code.local+3.pem --cert-key code.local+3-key.pem
```

---

## 방화벽 설정

### UFW (Ubuntu/Debian)

```bash
sudo ufw allow 22/tcp       # SSH
sudo ufw allow 80/tcp       # HTTP
sudo ufw allow 443/tcp      # HTTPS
sudo ufw enable
```

### firewalld (Fedora/CentOS)

```bash
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

---

## 로드 밸런싱 (다중 인스턴스)

### NGINX 로드 밸런서

```nginx
upstream code-servers {
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
}

server {
    listen 443 ssl http2;
    server_name code.yourdomain.com;

    location / {
        proxy_pass http://code-servers;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;

        # Sticky session (동일 사용자 → 동일 인스턴스)
        ip_hash;
    }
}
```

---

## 문제 해결

### WebSocket 연결 실패

**증상**: 터미널, 파일 watcher 작동 안 함

**해결**: 프록시 설정 확인

NGINX:
```nginx
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection upgrade;
```

Caddy: 자동 처리 (문제 없음)

### 리다이렉트 루프

**증상**: 무한 리다이렉트

**해결**: `X-Forwarded-Proto` 헤더 추가

```nginx
proxy_set_header X-Forwarded-Proto $scheme;
```

### Let's Encrypt 발급 실패

**원인**:
- DNS A 레코드 미설정
- 방화벽에서 80/443 차단
- 이미 사용 중인 도메인

**확인**:
```bash
# DNS 확인
nslookup code.yourdomain.com

# 포트 확인
sudo netstat -tulpn | grep :80
sudo netstat -tulpn | grep :443
```

---

*다음 글에서는 code-server의 프록시 시스템을 통한 웹 서비스 접근 방법을 다룹니다.*
