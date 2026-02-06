---
layout: post
title: "code-server 완벽 가이드 (10) - 확장 및 커스터마이징"
date: 2026-02-06
permalink: /code-server-guide-10-customization/
author: Coder
categories: [웹 개발, 원격 개발]
tags: [code-server, 커스터마이징, 국제화, 베스트 프랙티스]
original_url: "https://github.com/coder/code-server"
excerpt: "code-server의 국제화, 커스터마이징 및 운영 베스트 프랙티스."
---

## 국제화 (i18n)

### 언어 변경

code-server와 VS Code의 언어를 변경할 수 있습니다.

```bash
# 한국어
code-server --locale ko

# 일본어
code-server --locale ja

# 중국어 (간체)
code-server --locale zh-cn
```

config.yaml:
```yaml
locale: ko
```

### 지원 언어

- `en` - English (기본)
- `ko` - 한국어
- `ja` - 日本語
- `zh-cn` - 简体中文
- `zh-tw` - 繁體中文
- `fr` - Français
- `de` - Deutsch
- `es` - Español
- `it` - Italiano
- `ru` - Русский
- `pt-br` - Português (Brasil)

---

## 커스텀 문자열

로그인 페이지 및 UI 텍스트를 커스터마이징할 수 있습니다.

### custom-strings.json 생성

```json
{
  "WELCOME": "{{app}}에 오신 것을 환영합니다!",
  "LOGIN_TITLE": "{{app}} 로그인",
  "LOGIN_BELOW": "계속하려면 로그인하세요",
  "PASSWORD_PLACEHOLDER": "비밀번호 입력",
  "LOGIN_FAILED": "로그인 실패",
  "LOGIN_BUTTON": "로그인",
  "RATE_LIMIT": "너무 많은 시도. 잠시 후 다시 시도하세요."
}
```

### 적용

```yaml
# config.yaml
i18n: /path/to/custom-strings.json
app-name: "내 개발 환경"
```

또는:
```bash
code-server --i18n /path/to/custom-strings.json --app-name "내 개발 환경"
```

### 사용 가능한 키

전체 목록은 `src/node/i18n/locales/en.json` 참고:

```json
{
  "WELCOME": "Welcome to {{app}}",
  "LOGIN_TITLE": "{{app}} Login",
  "LOGIN_BELOW": "Please log in below",
  "PASSWORD_PLACEHOLDER": "Password",
  "LOGIN_BUTTON": "Log in",
  "LOGIN_FAILED": "Incorrect password",
  "RATE_LIMIT": "Too many login attempts. Please try again later.",
  "SESSION_EXPIRED": "Your session has expired. Please log in again."
}
```

---

## 테마 및 외관

### VS Code 테마 적용

```bash
# 명령 팔레트 (Cmd+Shift+P)
> Preferences: Color Theme

# 인기 테마 설치
code-server --install-extension zhuangtongfa.material-theme
code-server --install-extension GitHub.github-vscode-theme
code-server --install-extension monokai.theme-monokai-pro-vscode
```

### 폰트 변경

```json
// settings.json
{
  "editor.fontFamily": "'Fira Code', 'JetBrains Mono', Consolas, monospace",
  "editor.fontSize": 14,
  "editor.fontLigatures": true,
  "terminal.integrated.fontFamily": "'MesloLGS NF', monospace",
  "terminal.integrated.fontSize": 13
}
```

### UI 밀도

```json
{
  "workbench.activityBar.location": "top",
  "window.density.editorTabHeight": "compact",
  "workbench.tree.indent": 16
}
```

---

## 확장 프로그램 관리

### 필수 확장 사전 설치

```bash
# Dockerfile
FROM codercom/code-server:latest

USER coder

# 언어 지원
RUN code-server --install-extension ms-python.python && \
    code-server --install-extension golang.go && \
    code-server --install-extension rust-lang.rust-analyzer

# 도구
RUN code-server --install-extension dbaeumer.vscode-eslint && \
    code-server --install-extension esbenp.prettier-vscode && \
    code-server --install-extension eamodio.gitlens

# 테마
RUN code-server --install-extension GitHub.github-vscode-theme
```

### 확장 목록 내보내기/가져오기

```bash
# 내보내기
code-server --list-extensions > extensions.txt

# 가져오기
cat extensions.txt | xargs -L 1 code-server --install-extension
```

### 프라이빗 확장 레지스트리

```bash
# 환경 변수로 설정
export EXTENSIONS_GALLERY='{"serviceUrl": "https://extensions.company.com/api"}'
code-server
```

---

## 설정 동기화

### Settings Sync (GitHub Gist)

1. 명령 팔레트: `> Settings Sync: Turn On`
2. GitHub 계정 로그인
3. 자동 동기화 활성화

**동기화 항목:**
- 설정 (settings.json)
- 키 바인딩 (keybindings.json)
- 확장 프로그램
- UI 상태
- Snippets

### 수동 동기화

```bash
# 설정 백업
cp ~/.local/share/code-server/User/settings.json ~/backup/

# 설정 복원
cp ~/backup/settings.json ~/.local/share/code-server/User/
```

---

## 베스트 프랙티스

### 1. 보안

✅ **Do:**
- 강력한 비밀번호 사용
- HTTPS 필수 (Let's Encrypt)
- SSH 포트 포워딩 또는 OAuth
- 정기적인 업데이트
- 방화벽 설정

❌ **Don't:**
- `--auth none` + 인터넷 노출
- 자체 서명 인증서 + iPad
- 기본 비밀번호 사용
- HTTP로 인터넷 노출

### 2. 성능

```json
// settings.json
{
  // 파일 Watcher 최적화
  "files.watcherExclude": {
    "**/.git/**": true,
    "**/node_modules/**": true,
    "**/.venv/**": true,
    "**/dist/**": true,
    "**/build/**": true
  },

  // 자동 저장
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,

  // 검색 제외
  "search.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/.git": true
  },

  // 텔레메트리 비활성화
  "telemetry.telemetryLevel": "off"
}
```

### 3. 워크플로우

#### 프로젝트 템플릿

```bash
# ~/.config/code-server/templates/
templates/
├── react-app/
│   ├── .vscode/
│   │   ├── settings.json
│   │   └── extensions.json
│   ├── package.json
│   └── README.md
├── python-project/
└── golang-service/
```

#### 스니펫

```json
// ~/.local/share/code-server/User/snippets/javascript.json
{
  "Console Log": {
    "prefix": "cl",
    "body": "console.log('$1', $1);",
    "description": "빠른 console.log"
  },
  "Arrow Function": {
    "prefix": "af",
    "body": "const $1 = ($2) => {\n  $3\n};",
    "description": "화살표 함수"
  }
}
```

---

## 팀 환경 구축

### 1. 다중 사용자 인스턴스

```bash
# 사용자별 인스턴스
# user1
code-server --bind-addr 127.0.0.1:8080 --user-data-dir /data/user1

# user2
code-server --bind-addr 127.0.0.1:8081 --user-data-dir /data/user2

# user3
code-server --bind-addr 127.0.0.1:8082 --user-data-dir /data/user3
```

NGINX로 라우팅:
```nginx
map $http_host $backend {
    user1.dev.company.com 127.0.0.1:8080;
    user2.dev.company.com 127.0.0.1:8081;
    user3.dev.company.com 127.0.0.1:8082;
}

server {
    listen 443 ssl;
    server_name *.dev.company.com;

    location / {
        proxy_pass http://$backend;
    }
}
```

### 2. 표준화된 환경

```dockerfile
# Dockerfile
FROM codercom/code-server:latest

USER root

# 팀 표준 도구 설치
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    python3 \
    python3-pip \
    nodejs \
    npm \
    docker.io

# Python 패키지
RUN pip3 install black flake8 mypy pytest

# Node.js 패키지
RUN npm install -g eslint prettier typescript

USER coder

# 팀 표준 확장
RUN code-server --install-extension ms-python.python && \
    code-server --install-extension dbaeumer.vscode-eslint && \
    code-server --install-extension esbenp.prettier-vscode && \
    code-server --install-extension eamodio.gitlens

# 팀 설정
COPY team-settings.json /home/coder/.local/share/code-server/User/settings.json
```

### 3. 온보딩 자동화

```bash
#!/bin/bash
# onboard-new-dev.sh

NEW_USER=$1
PORT=$2

# 사용자 생성
sudo useradd -m $NEW_USER

# code-server 설치
sudo -u $NEW_USER bash -c "curl -fsSL https://code-server.dev/install.sh | sh"

# 설정
sudo -u $NEW_USER bash -c "cat > ~/.config/code-server/config.yaml <<EOF
bind-addr: 127.0.0.1:$PORT
auth: password
password: $(openssl rand -base64 16)
cert: false
EOF"

# Systemd 서비스 시작
sudo systemctl enable --now code-server@$NEW_USER

echo "사용자 $NEW_USER 설정 완료 (포트: $PORT)"
echo "비밀번호: $(sudo cat /home/$NEW_USER/.config/code-server/config.yaml | grep password)"
```

---

## 문제 해결 체크리스트

### 접속 불가

```bash
# 1. code-server 실행 중인지 확인
sudo systemctl status code-server@$USER

# 2. 포트 열려있는지 확인
sudo netstat -tulpn | grep 8080

# 3. 방화벽 확인
sudo ufw status

# 4. 로그 확인
journalctl -u code-server@$USER -f
```

### 성능 저하

```bash
# CPU/메모리 사용량 확인
top
htop

# 프로세스 확인
ps aux | grep code-server

# Node.js 메모리 증가
export NODE_OPTIONS=--max-old-space-size=4096
code-server
```

### 확장 프로그램 설치 실패

```bash
# 레지스트리 확인
echo $EXTENSIONS_GALLERY

# 수동 다운로드
wget https://marketplace.visualstudio.com/_apis/public/gallery/publishers/ms-python/vsextensions/python/latest/vspackage

# .vsix 파일로 설치
code-server --install-extension python.vsix
```

---

## 유용한 리소스

### 공식 문서

- [code-server 문서](https://coder.com/docs/code-server)
- [GitHub 레포](https://github.com/coder/code-server)
- [Discord 커뮤니티](https://discord.gg/coder)

### 확장 레지스트리

- [Open VSX Registry](https://open-vsx.org/)
- [VS Code Marketplace](https://marketplace.visualstudio.com/)

### 유사 프로젝트

- **Coder**: 팀/기업용 버전
- **Gitpod**: 클라우드 IDE
- **GitHub Codespaces**: GitHub 통합 IDE

---

## 마무리

code-server를 사용하면:

✅ 어디서든 일관된 개발 환경
✅ 강력한 서버 리소스 활용
✅ iPad에서도 풀스택 개발
✅ 팀 협업 간소화
✅ 보안 및 백업 용이

**핵심 권장사항:**

1. **보안**: HTTPS + 강력한 비밀번호
2. **백업**: 정기적인 데이터 백업
3. **모니터링**: 로그 및 성능 모니터링
4. **업데이트**: 최신 버전 유지
5. **문서화**: 팀 설정 문서화

---

## 감사의 글

이 가이드가 code-server를 시작하는 데 도움이 되었기를 바랍니다!

**피드백 및 기여:**
- GitHub Issues: [coder/code-server/issues](https://github.com/coder/code-server/issues)
- Discussions: [GitHub Discussions](https://github.com/coder/code-server/discussions)
- Discord: [discord.gg/coder](https://discord.gg/coder)

Happy Coding! 🚀

---

## 라이선스

code-server는 MIT License로 배포됩니다.

---

*이것으로 code-server 완벽 가이드 시리즈를 마칩니다. 즐거운 원격 개발 되세요!*
