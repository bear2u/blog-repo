---
layout: post
title: "code-server 완벽 가이드 (04) - 설정 및 구성"
date: 2026-02-06
permalink: /code-server-guide-04-configuration/
author: Coder
categories: [웹 개발, 원격 개발]
tags: [code-server, 설정, config.yaml, CLI 옵션]
original_url: "https://github.com/coder/code-server"
excerpt: "config.yaml과 CLI 옵션을 통한 code-server 설정 방법을 알아봅니다."
---

## 설정 파일 위치

code-server는 YAML 형식의 설정 파일을 사용합니다:

```bash
~/.config/code-server/config.yaml
```

**커스텀 경로 지정:**

```bash
code-server --config /path/to/custom/config.yaml
```

---

## 기본 config.yaml

```yaml
bind-addr: 127.0.0.1:8080
auth: password
password: a1b2c3d4e5f6g7h8i9j0
cert: false
```

---

## 전체 설정 옵션

### 네트워크 설정

#### bind-addr (바인딩 주소)

```yaml
# 로컬호스트만 (기본)
bind-addr: 127.0.0.1:8080

# 모든 인터페이스
bind-addr: 0.0.0.0:8080

# 특정 IP
bind-addr: 192.168.1.100:8080

# IPv6
bind-addr: "[::]:8080"
```

CLI:
```bash
code-server --bind-addr 0.0.0.0:3000
```

---

### 인증 설정

#### auth (인증 방식)

```yaml
# 비밀번호 인증 (기본)
auth: password

# 인증 없음 (로컬 테스트용)
auth: none
```

CLI:
```bash
code-server --auth none
```

#### password (비밀번호)

```yaml
password: mySecurePassword123
```

CLI:
```bash
code-server --password mySecurePassword123
```

환경 변수:
```bash
export PASSWORD=mySecurePassword123
code-server
```

---

### HTTPS/TLS 설정

#### cert (인증서)

```yaml
# 자체 서명 인증서 자동 생성
cert: true

# 인증서 비활성화
cert: false

# 커스텀 인증서 경로
cert: /path/to/cert.crt
```

#### cert-key (인증서 키)

```yaml
cert: /path/to/cert.crt
cert-key: /path/to/cert.key
```

CLI:
```bash
code-server --cert /path/to/cert.crt --cert-key /path/to/cert.key
```

**자체 서명 인증서 생성 위치:**
```
~/.local/share/code-server/self-signed.crt
~/.local/share/code-server/self-signed.key
```

---

### 디렉토리 설정

#### user-data-dir (사용자 데이터)

VS Code 사용자 설정, 확장 프로그램, 상태가 저장되는 디렉토리.

```yaml
user-data-dir: ~/.local/share/code-server
```

CLI:
```bash
code-server --user-data-dir /custom/path
```

#### extensions-dir (확장 디렉토리)

```yaml
extensions-dir: ~/.local/share/code-server/extensions
```

CLI:
```bash
code-server --extensions-dir /custom/extensions
```

---

### 프록시 설정

#### proxy-domain (서브도메인 프록시)

```yaml
proxy-domain: mydomain.com
```

사용 예:
- 서버: `mydomain.com`
- 포트 3000 앱: `3000.mydomain.com`
- 포트 8000 앱: `8000.mydomain.com`

CLI:
```bash
code-server --proxy-domain mydomain.com
```

---

### 로깅 설정

#### log (로그 레벨)

```yaml
log: info
```

레벨: `trace`, `debug`, `info`, `warn`, `error`

CLI:
```bash
code-server --log debug
```

#### verbose (상세 로깅)

```yaml
verbose: true
```

CLI:
```bash
code-server --verbose
```

---

## CLI 전용 옵션

### --version

```bash
code-server --version
# 4.20.0 48f4ab06c8fa0622b8614e027fe03fd75c076c18
```

### --help

```bash
code-server --help
```

### --open

```bash
# 시작 후 브라우저 자동 열기
code-server --open
```

### --disable-telemetry

```bash
# VS Code 텔레메트리 비활성화
code-server --disable-telemetry
```

### --disable-update-check

```bash
# 자동 업데이트 확인 비활성화
code-server --disable-update-check
```

### --disable-file-downloads

```bash
# 파일 다운로드 비활성화
code-server --disable-file-downloads
```

### --disable-workspace-trust

```bash
# Workspace Trust 프롬프트 비활성화
code-server --disable-workspace-trust
```

### --link

```bash
# Coder 클라우드에 링크 (managed tunneling)
code-server --link
```

---

## 국제화 설정

### --locale

```bash
# VS Code 언어 설정
code-server --locale ko
```

지원 언어: `en`, `ko`, `zh-cn`, `zh-tw`, `ja`, `fr`, `de`, `es`, `it`, `ru`, `pt-br`

### --i18n (커스텀 문자열)

```yaml
i18n: /path/to/custom-strings.json
```

`custom-strings.json`:
```json
{
  "WELCOME": "{{app}}에 오신 것을 환영합니다",
  "LOGIN_TITLE": "{{app}} 로그인",
  "PASSWORD_PLACEHOLDER": "비밀번호를 입력하세요"
}
```

CLI:
```bash
code-server --i18n /path/to/custom-strings.json
```

### --app-name

```bash
# 애플리케이션 이름 변경 ({{app}} 플레이스홀더)
code-server --app-name "내 개발 환경"
```

---

## 환경 변수

### PASSWORD

```bash
export PASSWORD=mypass
code-server
```

### VSCODE_PROXY_URI

```yaml
# 프록시 URL 템플릿
export VSCODE_PROXY_URI="https://{{port}}.mydomain.com"
```

또는 상대 경로:
```bash
export VSCODE_PROXY_URI="./proxy/{{port}}"
```

---

## 설정 우선순위

```
1. CLI 인자 (최우선)
2. 환경 변수
3. config.yaml
4. 기본값
```

예시:
```bash
# config.yaml
password: config_password

# 환경 변수
export PASSWORD=env_password

# CLI 인자
code-server --password cli_password

# 결과: cli_password 사용
```

---

## 일반적인 설정 시나리오

### 1. 로컬 개발 (인증 없음)

```yaml
# config.yaml
bind-addr: 127.0.0.1:8080
auth: none
cert: false
```

```bash
code-server
```

### 2. 원격 서버 (SSH 터널)

```yaml
# config.yaml
bind-addr: 127.0.0.1:8080
auth: password
password: strongPassword123
cert: false
```

SSH 포트 포워딩:
```bash
ssh -N -L 8080:127.0.0.1:8080 user@server
```

### 3. 인터넷 노출 (Let's Encrypt)

```yaml
# config.yaml
bind-addr: 127.0.0.1:8080
auth: password
password: strongPassword123
cert: false
```

Caddy 리버스 프록시 (자동 HTTPS):
```
mydomain.com {
  reverse_proxy 127.0.0.1:8080
}
```

### 4. 자체 서명 인증서

```yaml
# config.yaml
bind-addr: 0.0.0.0:443
auth: password
password: strongPassword123
cert: true
```

```bash
# 포트 443 바인딩 권한 부여
sudo setcap cap_net_bind_service=+ep /usr/lib/code-server/lib/node

# 시작
code-server
```

### 5. 팀 환경 (다중 인스턴스)

각 사용자마다 별도 인스턴스:

```bash
# 사용자 1
code-server --bind-addr 127.0.0.1:8080 --user-data-dir ~/.code-server/user1

# 사용자 2
code-server --bind-addr 127.0.0.1:8081 --user-data-dir ~/.code-server/user2
```

---

## 고급 설정

### Systemd 서비스 커스터마이즈

```bash
# 서비스 파일 편집
sudo systemctl edit code-server@$USER
```

추가 설정:
```ini
[Service]
# 환경 변수
Environment="PASSWORD=mypassword"
Environment="VSCODE_PROXY_URI=https://{{port}}.mydomain.com"

# 메모리 제한
MemoryLimit=4G

# CPU 제한
CPUQuota=200%

# 재시작 정책
Restart=always
RestartSec=10
```

### Docker 환경 변수

```bash
docker run -it -p 127.0.0.1:8080:8080 \
  -e PASSWORD=mypassword \
  -e SUDO_PASSWORD=sudopass \
  -v "$PWD:/home/coder/project" \
  codercom/code-server:latest
```

### 프록시 뒤에서 실행

```yaml
# config.yaml
bind-addr: 127.0.0.1:8080
auth: password
password: mypass
cert: false
proxy-domain: dev.example.com
```

NGINX 설정:
```nginx
server {
    listen 80;
    server_name dev.example.com *.dev.example.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 설정 파일 예제

### 최소 설정

```yaml
bind-addr: 127.0.0.1:8080
auth: password
password: simplepass
cert: false
```

### 프로덕션 설정

```yaml
bind-addr: 127.0.0.1:8080
auth: password
password: veryStrongPassword!2024
cert: false
user-data-dir: /var/lib/code-server
extensions-dir: /var/lib/code-server/extensions
log: info
disable-telemetry: true
disable-update-check: false
proxy-domain: code.company.com
```

### 개발 설정

```yaml
bind-addr: 127.0.0.1:8080
auth: none
cert: false
log: debug
verbose: true
disable-telemetry: true
```

---

## 설정 검증

### 현재 설정 확인

```bash
# config.yaml 내용 출력
cat ~/.config/code-server/config.yaml

# 실행 중인 code-server 설정 확인
code-server --version  # 버전 정보와 함께 설정 경로 표시
```

### 설정 테스트

```bash
# 드라이런 (설정 파싱만)
code-server --help

# 로그 레벨 높여서 시작
code-server --log trace
```

---

## 문제 해결

### 설정 파일이 적용되지 않음

```bash
# 설정 파일 위치 확인
ls -la ~/.config/code-server/config.yaml

# 권한 확인
chmod 600 ~/.config/code-server/config.yaml

# 문법 오류 확인 (YAML 유효성)
cat ~/.config/code-server/config.yaml | python3 -c "import yaml, sys; yaml.safe_load(sys.stdin)"
```

### 비밀번호 변경 후 로그인 안됨

```bash
# 설정 파일 확인
cat ~/.config/code-server/config.yaml

# 서비스 재시작
sudo systemctl restart code-server@$USER

# 또는 프로세스 재시작
pkill code-server
code-server
```

### 포트 충돌

```bash
# 포트 사용 중인 프로세스 확인
sudo lsof -i :8080

# 다른 포트 사용
code-server --bind-addr 127.0.0.1:9000
```

---

## 보안 권장사항

### 1. 강력한 비밀번호 사용

```bash
# 안전한 랜덤 비밀번호 생성
openssl rand -base64 32
```

### 2. 외부 노출 시 HTTPS 필수

```yaml
# 절대 이렇게 하지 마세요!
bind-addr: 0.0.0.0:8080  # ❌
auth: none                # ❌
cert: false               # ❌
```

올바른 방법:
```yaml
bind-addr: 127.0.0.1:8080  # ✅ 로컬만
auth: password              # ✅ 인증 활성화
password: strong_password   # ✅ 강력한 비밀번호
```

그리고 리버스 프록시(Caddy/NGINX)로 HTTPS 제공.

### 3. 정기적인 비밀번호 변경

```bash
# 새 비밀번호 생성
NEW_PASS=$(openssl rand -base64 32)

# config.yaml 업데이트
sed -i "s/^password:.*/password: $NEW_PASS/" ~/.config/code-server/config.yaml

# 재시작
sudo systemctl restart code-server@$USER

# 새 비밀번호 출력
echo "New password: $NEW_PASS"
```

---

*다음 글에서는 HTTPS, 인증, 외부 인증 통합 등 보안 설정을 자세히 살펴봅니다.*
