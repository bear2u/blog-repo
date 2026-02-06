---
layout: post
title: "code-server 완벽 가이드 (05) - 보안 및 인증"
date: 2026-02-06
permalink: /code-server-guide-05-security/
author: Coder
categories: [웹 개발, 원격 개발]
tags: [code-server, 보안, HTTPS, 인증, OAuth]
original_url: "https://github.com/coder/code-server"
excerpt: "code-server의 보안 설정과 다양한 인증 방법을 알아봅니다."
---

## 보안 원칙

**절대 원칙: code-server를 인증/암호화 없이 인터넷에 노출하지 마세요!**

누구든지 터미널 접근 → 서버 완전 장악 가능

```
❌ 위험: bind-addr: 0.0.0.0:8080 + auth: none
✅ 안전: SSH 터널 또는 HTTPS + 인증
```

---

## 인증 방법

### 1. 비밀번호 인증 (기본)

가장 간단하고 기본적인 방법입니다.

```yaml
# config.yaml
auth: password
password: yourStrongPassword123
```

**Rate Limiting 자동 적용:**
- 분당 2회 시도
- 시간당 추가 12회 시도
- 초과 시 일시적 차단

**비밀번호 요구사항:**
- 최소 길이 없음 (but 강력한 비밀번호 권장)
- 특수문자, 대소문자 혼합 권장

**안전한 비밀번호 생성:**

```bash
# 방법 1: openssl
openssl rand -base64 32

# 방법 2: pwgen
pwgen -s 32 1

# 방법 3: 온라인 생성기
# https://passwordsgenerator.net/
```

---

### 2. 인증 없음 (로컬 테스트 전용)

```yaml
auth: none
```

**⚠️ 경고**: 로컬 개발에만 사용하세요!

**안전한 사용 예:**

```bash
# SSH 터널 + 인증 없음
# 서버:
code-server --auth none --bind-addr 127.0.0.1:8080

# 로컬:
ssh -N -L 8080:127.0.0.1:8080 user@server
# http://localhost:8080 접속 (SSH가 인증 담당)
```

---

## HTTPS 설정

### 방법 1: 자체 서명 인증서

가장 빠른 HTTPS 활성화 방법 (iPad에서는 작동 안 함).

```yaml
# config.yaml
bind-addr: 0.0.0.0:443
auth: password
password: strongPass
cert: true
```

```bash
# 포트 443 바인딩 권한 부여
sudo setcap cap_net_bind_service=+ep /usr/lib/code-server/lib/node

# 시작
code-server
```

**인증서 위치:**
```
~/.local/share/code-server/self-signed.crt
~/.local/share/code-server/self-signed.key
```

**브라우저 경고 우회:**

대부분의 브라우저에서 "고급 → 계속 진행" 클릭 필요.

---

### 방법 2: 커스텀 인증서

자신의 인증서가 있는 경우:

```yaml
cert: /path/to/fullchain.pem
cert-key: /path/to/privkey.pem
```

또는 CLI:
```bash
code-server --cert /path/to/cert.crt --cert-key /path/to/cert.key
```

---

### 방법 3: Let's Encrypt + Caddy (권장)

**자동 HTTPS 인증서 발급 및 갱신!**

#### 단계 1: Caddy 설치

```bash
# Debian/Ubuntu
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

#### 단계 2: code-server 설정

```yaml
# config.yaml
bind-addr: 127.0.0.1:8080
auth: password
password: strongPass
cert: false
```

#### 단계 3: Caddyfile 설정

```bash
# /etc/caddy/Caddyfile
code.yourdomain.com {
  reverse_proxy 127.0.0.1:8080
}
```

서브패스로 제공:
```
yourdomain.com/code/* {
  uri strip_prefix /code
  reverse_proxy 127.0.0.1:8080
}
```

#### 단계 4: Caddy 시작

```bash
sudo systemctl reload caddy
```

**완료!** `https://code.yourdomain.com` 접속 가능.

---

### 방법 4: Let's Encrypt + NGINX

#### 단계 1: NGINX & Certbot 설치

```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
```

#### 단계 2: NGINX 설정

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

#### 단계 3: 활성화

```bash
sudo ln -s /etc/nginx/sites-available/code-server /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 단계 4: Let's Encrypt 인증서 발급

```bash
sudo certbot --nginx -d code.yourdomain.com --non-interactive --agree-tos -m you@example.com --redirect
```

**자동 갱신 확인:**
```bash
sudo certbot renew --dry-run
```

---

## 외부 인증 시스템

### 1. OAuth2-Proxy

Google, GitHub, GitLab 등으로 로그인.

#### 설치

```bash
wget https://github.com/oauth2-proxy/oauth2-proxy/releases/download/v7.4.0/oauth2-proxy-v7.4.0.linux-amd64.tar.gz
tar -xzf oauth2-proxy-v7.4.0.linux-amd64.tar.gz
sudo mv oauth2-proxy-v7.4.0.linux-amd64/oauth2-proxy /usr/local/bin/
```

#### Google OAuth 설정

1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. 프로젝트 생성
3. OAuth 2.0 클라이언트 ID 생성
   - Application type: Web application
   - Authorized redirect URIs: `https://code.yourdomain.com/oauth2/callback`
4. Client ID & Secret 저장

#### oauth2-proxy 설정

```bash
# /etc/oauth2-proxy.cfg
http_address = "127.0.0.1:4180"
upstreams = ["http://127.0.0.1:8080/"]

client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_CLIENT_SECRET"

cookie_secret = "RANDOM_32_CHAR_STRING"
cookie_secure = true

provider = "google"
email_domains = ["yourdomain.com"]  # 허용할 이메일 도메인

redirect_url = "https://code.yourdomain.com/oauth2/callback"
```

#### 시작

```bash
oauth2-proxy --config /etc/oauth2-proxy.cfg
```

#### NGINX 설정 수정

```nginx
server {
    listen 443 ssl;
    server_name code.yourdomain.com;

    # Let's Encrypt 인증서
    ssl_certificate /etc/letsencrypt/live/code.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/code.yourdomain.com/privkey.pem;

    location /oauth2/ {
        proxy_pass http://127.0.0.1:4180;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Scheme $scheme;
    }

    location = /oauth2/auth {
        proxy_pass http://127.0.0.1:4180;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Original-URI $request_uri;
    }

    location / {
        auth_request /oauth2/auth;
        error_page 401 = /oauth2/sign_in;

        proxy_pass http://localhost:8080/;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
    }
}
```

---

### 2. Pomerium

엔터프라이즈급 ID-인식 프록시.

[Pomerium 가이드](https://www.pomerium.com/docs/guides/code-server.html) 참고.

---

### 3. Cloudflare Access

Cloudflare의 Zero Trust 솔루션.

1. Cloudflare 계정 생성
2. [Zero Trust Dashboard](https://one.dash.cloudflare.com/) 접속
3. Access > Applications > Add an Application
4. Self-hosted 선택
5. 도메인 설정: `code.yourdomain.com`
6. 정책 설정 (이메일, Google Workspace 등)

자세한 내용: [Cloudflare Access 문서](https://www.cloudflare.com/zero-trust/products/access/)

---

## SSH 포트 포워딩

가장 안전한 방법 중 하나.

### 기본 포트 포워딩

```bash
# 서버: 로컬호스트만 바인딩 + 인증 비활성화
code-server --auth none --bind-addr 127.0.0.1:8080

# 로컬: SSH 터널 생성
ssh -N -L 8080:127.0.0.1:8080 user@server

# 브라우저: http://localhost:8080
```

### 백그라운드 SSH 터널 (지속)

```bash
# mutagen 설치
brew install mutagen-io/mutagen/mutagen
# 또는
curl -fsSL https://mutagen.io/install.sh | sh

# 터널 생성
mutagen forward create --name=code-server tcp:127.0.0.1:8080 server:tcp:127.0.0.1:8080

# 상태 확인
mutagen forward list

# 터널 종료
mutagen forward terminate code-server
```

### SSH 설정 최적화

```bash
# ~/.ssh/config
Host code-server
    HostName server.example.com
    User myuser
    Port 22
    LocalForward 8080 127.0.0.1:8080
    ServerAliveInterval 5
    ExitOnForwardFailure yes
```

사용:
```bash
ssh code-server -N
```

---

## 추가 보안 조치

### 1. 파일 다운로드 비활성화

```bash
code-server --disable-file-downloads
```

민감한 데이터 유출 방지.

### 2. Workspace Trust 비활성화

```bash
code-server --disable-workspace-trust
```

신뢰할 수 있는 환경에서만 사용.

### 3. 확장 마켓플레이스 제한

```bash
export EXTENSIONS_GALLERY='{"serviceUrl": "https://your-registry.com"}'
```

공식 마켓플레이스 대신 프라이빗 레지스트리 사용.

### 4. 방화벽 설정

```bash
# Ubuntu/Debian (UFW)
sudo ufw allow 22/tcp       # SSH
sudo ufw allow 80/tcp       # HTTP (Certbot)
sudo ufw allow 443/tcp      # HTTPS
sudo ufw enable

# Fedora/CentOS (firewalld)
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### 5. Fail2ban (무차별 대입 공격 방지)

```bash
# 설치
sudo apt install fail2ban

# code-server jail 설정
sudo nano /etc/fail2ban/jail.local
```

```ini
[code-server]
enabled = true
port = 80,443
logpath = /var/log/syslog
filter = code-server
maxretry = 3
bantime = 3600
```

필터 생성:
```bash
# /etc/fail2ban/filter.d/code-server.conf
[Definition]
failregex = ^.*\[.*\] info.*Failed login attempt.*from <HOST>
ignoreregex =
```

재시작:
```bash
sudo systemctl restart fail2ban
```

---

## 보안 체크리스트

### 필수

- [ ] 강력한 비밀번호 사용
- [ ] HTTPS 활성화 (리버스 프록시 또는 자체 인증서)
- [ ] 최신 버전 유지
- [ ] 방화벽 설정
- [ ] 불필요한 포트 닫기

### 권장

- [ ] SSH 포트 포워딩 또는 OAuth 사용
- [ ] 파일 다운로드 비활성화 (민감 환경)
- [ ] 정기적인 비밀번호 변경
- [ ] 로그 모니터링
- [ ] Fail2ban 설치

### 고급

- [ ] VPN 사용 (WireGuard, OpenVPN)
- [ ] 2FA (OAuth 프록시를 통해)
- [ ] IP 화이트리스트
- [ ] 감사 로그 수집

---

## 보안 감사

### 로그 확인

```bash
# Systemd 로그
journalctl -u code-server@$USER -f

# 로그인 시도 확인
grep "Failed login" /var/log/syslog

# 접속 IP 확인
grep "HTTP request" /var/log/syslog | awk '{print $8}' | sort | uniq -c
```

### 네트워크 스캔

```bash
# 열린 포트 확인
sudo netstat -tulpn | grep code-server

# nmap 스캔
nmap -sV -p 8080 localhost
```

---

## 사고 대응

### 비밀번호 유출 시

```bash
# 1. 즉시 서비스 중지
sudo systemctl stop code-server@$USER

# 2. 비밀번호 변경
nano ~/.config/code-server/config.yaml
# password: NEW_SECURE_PASSWORD

# 3. 로그 확인
journalctl -u code-server@$USER --since "1 hour ago"

# 4. 서비스 재시작
sudo systemctl start code-server@$USER
```

### 무단 접근 의심 시

```bash
# 1. 서비스 중지
sudo systemctl stop code-server@$USER

# 2. 로그 분석
journalctl -u code-server@$USER | grep -i "login\|error\|warning"

# 3. 파일 변경 확인
find ~/ -type f -mtime -1

# 4. 설정 초기화 (필요시)
rm -rf ~/.local/share/code-server
rm ~/.config/code-server/config.yaml

# 5. 재설치
code-server --install-extension ... (필요한 확장만)
```

---

*다음 글에서는 SSH, Caddy, NGINX를 통한 네트워크 설정을 자세히 다룹니다.*
