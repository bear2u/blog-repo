---
layout: post
title: "code-server 완벽 가이드 (02) - 설치 및 시작"
date: 2026-02-06
permalink: /code-server-guide-02-installation/
author: Coder
categories: [웹 개발, 원격 개발]
tags: [code-server, 설치, install.sh, npm, Docker, Helm]
original_url: "https://github.com/coder/code-server"
excerpt: "code-server를 설치하는 다양한 방법을 상세히 알아봅니다."
---

## 설치 방법 개요

code-server는 **7가지 설치 방법**을 제공합니다:

| 방법 | 플랫폼 | 난이도 | 권장 대상 |
|------|--------|--------|----------|
| **install.sh** | Linux, macOS, FreeBSD | ⭐ | 가장 쉬움, 권장 |
| **npm** | 모든 플랫폼 | ⭐⭐ | Windows, 특수 아키텍처 |
| **Standalone** | Linux, macOS | ⭐⭐ | 수동 설치 선호 시 |
| **Debian/Ubuntu** | Debian/Ubuntu | ⭐ | .deb 패키지 |
| **Fedora/RHEL** | Fedora, CentOS, RHEL | ⭐ | .rpm 패키지 |
| **Docker** | 모든 플랫폼 | ⭐⭐ | 컨테이너 환경 |
| **Helm** | Kubernetes | ⭐⭐⭐ | K8s 클러스터 |

---

## 방법 1: install.sh (권장)

가장 쉽고 권장되는 방법입니다. 시스템 패키지 매니저를 자동으로 감지하고 사용합니다.

### 설치 전 미리보기

```bash
# 설치 스크립트가 무엇을 할지 미리 확인
curl -fsSL https://code-server.dev/install.sh | sh -s -- --dry-run
```

### 설치 실행

```bash
# 자동 설치
curl -fsSL https://code-server.dev/install.sh | sh
```

### 설치 스크립트 옵션

```bash
# 특정 버전 설치
curl -fsSL https://code-server.dev/install.sh | sh -s -- --version=4.9.1

# standalone 방식으로 ~/.local에 설치
curl -fsSL https://code-server.dev/install.sh | sh -s -- --method=standalone

# /usr/local에 시스템 전역 설치
curl -fsSL https://code-server.dev/install.sh | sh -s -- --prefix=/usr/local

# Edge 버전 (pre-release) 설치
curl -fsSL https://code-server.dev/install.sh | sh -s -- --edge
```

### 자동 감지 규칙

install.sh는 다음 순서로 설치 방법을 자동 선택합니다:

| OS | 설치 방법 |
|-----|----------|
| Debian, Ubuntu | `.deb` 패키지 |
| Fedora, CentOS, RHEL, openSUSE | `.rpm` 패키지 |
| Arch Linux | AUR 패키지 |
| macOS (Homebrew 있음) | Homebrew |
| macOS (Homebrew 없음) | Standalone (`~/.local`) |
| FreeBSD | npm |
| 기타 Linux | Standalone (`~/.local`) |

### 설치 후 시작

```bash
# Systemd로 자동 시작 (Linux)
sudo systemctl enable --now code-server@$USER

# 수동 실행
code-server

# 접속: http://127.0.0.1:8080
# 비밀번호: ~/.config/code-server/config.yaml
```

---

## 방법 2: npm

Windows나 특수 아키텍처(amd64/arm64가 아닌 경우)에 권장됩니다.

### 사전 요구사항

npm 설치 시 네이티브 모듈을 빌드하므로 C 컴파일러가 필요합니다.

#### Ubuntu/Debian

```bash
sudo apt-get install -y \
  build-essential \
  pkg-config \
  python3
npm config set python python3
```

#### Fedora/CentOS/RHEL

```bash
sudo yum groupinstall -y 'Development Tools'
sudo yum config-manager --set-enabled PowerTools # CentOS 8
sudo yum install -y python3
npm config set python python3
```

#### macOS

```bash
xcode-select --install
```

#### Windows

```powershell
# Node.js 설치 시 "Tools for Native Modules" 옵션 선택
# 또는 Visual Studio Build Tools 설치
npm install --global windows-build-tools
```

### npm으로 설치

```bash
# 전역 설치
npm install --global code-server

# 실행
code-server
```

### npm 설치가 필요한 경우

1. amd64/arm64가 아닌 아키텍처 (예: armv7, ppc64le)
2. Windows
3. glibc < v2.28 또는 glibcxx < v3.4.21인 Linux
4. Alpine Linux 또는 non-glibc libc

---

## 방법 3: Standalone 릴리스

시스템 패키지 매니저 없이 수동으로 설치할 때 사용합니다.

### 다운로드 및 설치

```bash
# 버전 설정
VERSION=4.20.0

# 다운로드 및 압축 해제
mkdir -p ~/.local/lib ~/.local/bin
curl -fL https://github.com/coder/code-server/releases/download/v$VERSION/code-server-$VERSION-linux-amd64.tar.gz \
  | tar -C ~/.local/lib -xz

# 심볼릭 링크 생성
mv ~/.local/lib/code-server-$VERSION-linux-amd64 ~/.local/lib/code-server-$VERSION
ln -s ~/.local/lib/code-server-$VERSION/bin/code-server ~/.local/bin/code-server

# PATH 추가
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 실행
code-server
```

### 요구사항

- **Linux**: glibc >= 2.28, glibcxx >= 3.4.21
- **macOS**: 최소 요구사항 없음

---

## 방법 4: Debian/Ubuntu (.deb)

```bash
VERSION=4.20.0

# .deb 파일 다운로드
curl -fOL https://github.com/coder/code-server/releases/download/v$VERSION/code-server_${VERSION}_amd64.deb

# 설치
sudo dpkg -i code-server_${VERSION}_amd64.deb

# Systemd로 자동 시작
sudo systemctl enable --now code-server@$USER

# 접속: http://127.0.0.1:8080
# 비밀번호: ~/.config/code-server/config.yaml
```

**주의**: arm64 .deb는 Ubuntu 16.04 이하를 지원하지 않습니다.

---

## 방법 5: Fedora/CentOS/RHEL (.rpm)

```bash
VERSION=4.20.0

# .rpm 파일 다운로드
curl -fOL https://github.com/coder/code-server/releases/download/v$VERSION/code-server-$VERSION-amd64.rpm

# 설치
sudo rpm -i code-server-$VERSION-amd64.rpm

# Systemd로 자동 시작
sudo systemctl enable --now code-server@$USER

# 접속: http://127.0.0.1:8080
# 비밀번호: ~/.config/code-server/config.yaml
```

**주의**: arm64 .rpm은 CentOS 7을 지원하지 않습니다.

---

## 방법 6: Arch Linux (AUR)

### yay 사용

```bash
# yay로 설치
yay -S code-server

# Systemd로 자동 시작
sudo systemctl enable --now code-server@$USER
```

### makepkg 사용

```bash
# AUR 레포지토리 클론
git clone https://aur.archlinux.org/code-server.git
cd code-server

# 빌드 및 설치
makepkg -si

# Systemd로 자동 시작
sudo systemctl enable --now code-server@$USER
```

---

## 방법 7: macOS (Homebrew)

```bash
# Homebrew로 설치
brew install code-server

# 백그라운드 서비스로 시작
brew services start code-server

# 접속: http://127.0.0.1:8080
# 비밀번호: ~/.config/code-server/config.yaml
```

---

## 방법 8: Docker

### 기본 실행

```bash
# 현재 디렉토리를 프로젝트로 마운트
mkdir -p ~/.config
docker run -it --name code-server -p 127.0.0.1:8080:8080 \
  -v "$HOME/.local:/home/coder/.local" \
  -v "$HOME/.config:/home/coder/.config" \
  -v "$PWD:/home/coder/project" \
  -u "$(id -u):$(id -g)" \
  -e "DOCKER_USER=$USER" \
  codercom/code-server:latest
```

### Docker Compose

```yaml
version: "3"
services:
  code-server:
    image: codercom/code-server:latest
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./project:/home/coder/project
      - ~/.config:/home/coder/.config
    environment:
      - PASSWORD=your-password-here
    restart: unless-stopped
```

```bash
docker-compose up -d
```

### 지원 아키텍처

- **공식 이미지**: amd64, arm64
- **커뮤니티 이미지**: arm32 (linuxserver/code-server)

---

## 방법 9: Kubernetes (Helm)

```bash
# Helm 레포지토리 추가
helm repo add coder https://helm.coder.com
helm repo update

# 설치
helm install code-server coder/code-server \
  --set persistence.enabled=true \
  --set persistence.size=10Gi

# 서비스 확인
kubectl get svc code-server

# 포트 포워딩
kubectl port-forward svc/code-server 8080:8080
```

자세한 내용은 [Helm 문서](https://coder.com/docs/code-server/latest/helm)를 참고하세요.

---

## 플랫폼별 특수 설치

### Raspberry Pi

npm으로 설치 권장:

```bash
# 의존성 설치
sudo apt-get install -y build-essential

# npm으로 설치
npm install --global code-server
```

node-gyp 오류 발생 시 [이슈 #5174](https://github.com/coder/code-server/issues/5174) 참고.

### Termux (Android)

```bash
# 패키지 설치
pkg install tur-repo
pkg install code-server

# 실행
code-server
```

자세한 내용은 [Termux 문서](https://coder.com/docs/code-server/latest/termux) 참고.

### Windows

```powershell
# npm으로 설치
npm install --global code-server

# 실행
code-server
```

**참고**: Windows용 네이티브 빌드는 제공하지 않습니다. npm 사용 필수.

---

## 클라우드 배포 (원클릭)

Coder 팀은 주요 클라우드 제공업체를 위한 원클릭 배포를 제공합니다:

- **DigitalOcean**
- **Railway**
- **Heroku**
- **Azure**
- **AWS**
- **Google Cloud**

[deploy-code-server](https://github.com/coder/deploy-code-server) 레포지토리 참고.

---

## 설치 확인

```bash
# 버전 확인
code-server --version

# 도움말
code-server --help

# 설정 파일 위치 확인
ls -la ~/.config/code-server/config.yaml

# 테스트 실행
code-server --bind-addr 127.0.0.1:8080
```

---

## 첫 실행 및 로그인

### 1. code-server 시작

```bash
code-server
```

출력:
```
[2026-02-06T13:00:00.000Z] info  code-server 4.20.0 48f4ab06c8fa0622b8614e027fe03fd75c076c18
[2026-02-06T13:00:00.000Z] info  Using user-data-dir ~/.local/share/code-server
[2026-02-06T13:00:00.000Z] info  Using config file ~/.config/code-server/config.yaml
[2026-02-06T13:00:00.000Z] info  HTTP server listening on http://127.0.0.1:8080/
[2026-02-06T13:00:00.000Z] info    - Authentication is enabled
[2026-02-06T13:00:00.000Z] info      - Using password from ~/.config/code-server/config.yaml
[2026-02-06T13:00:00.000Z] info    - Not serving HTTPS
```

### 2. 비밀번호 확인

```bash
cat ~/.config/code-server/config.yaml
```

출력:
```yaml
bind-addr: 127.0.0.1:8080
auth: password
password: a1b2c3d4e5f6g7h8
cert: false
```

### 3. 브라우저에서 접속

```
http://127.0.0.1:8080
```

비밀번호 입력 → VS Code 화면!

---

## 기본 설정 파일

`~/.config/code-server/config.yaml`:

```yaml
# 바인딩 주소 (기본: localhost만)
bind-addr: 127.0.0.1:8080

# 인증 방식 (password | none)
auth: password

# 비밀번호 (auth: password일 때)
password: randompassword

# HTTPS 인증서 (false | true | /path/to/cert.crt)
cert: false

# 인증서 키 파일 (cert가 경로일 때)
cert-key: /path/to/cert.key
```

---

## Systemd 서비스 관리

### 서비스 시작/중지

```bash
# 시작
sudo systemctl start code-server@$USER

# 중지
sudo systemctl stop code-server@$USER

# 재시작
sudo systemctl restart code-server@$USER

# 상태 확인
sudo systemctl status code-server@$USER
```

### 부팅 시 자동 시작

```bash
# 활성화
sudo systemctl enable code-server@$USER

# 비활성화
sudo systemctl disable code-server@$USER
```

### 로그 확인

```bash
# 실시간 로그
journalctl -u code-server@$USER -f

# 최근 로그
journalctl -u code-server@$USER -n 50
```

---

## 제거 (Uninstall)

### 데이터 및 설정 삭제

```bash
# 사용자 데이터 삭제
rm -rf ~/.local/share/code-server ~/.config/code-server
```

### install.sh로 설치한 경우

```bash
rm -rf ~/.local/lib/code-server-*
rm ~/.local/bin/code-server
```

### Homebrew

```bash
brew uninstall code-server
```

### npm

```bash
npm uninstall --global code-server
```

### Debian/Ubuntu

```bash
sudo apt remove code-server
```

### Fedora/CentOS/RHEL

```bash
sudo rpm -e code-server
```

---

## 문제 해결

### 포트 충돌

```bash
# 다른 포트 사용
code-server --bind-addr 127.0.0.1:3000
```

### 방화벽 이슈

```bash
# Ubuntu/Debian
sudo ufw allow 8080/tcp

# Fedora/CentOS
sudo firewall-cmd --add-port=8080/tcp --permanent
sudo firewall-cmd --reload
```

### 비밀번호 분실

```bash
# 새 비밀번호 설정
sed -i "s/^password:.*/password: mynewpassword/" ~/.config/code-server/config.yaml

# 서비스 재시작
sudo systemctl restart code-server@$USER
```

---

## 다음 단계

설치가 완료되었다면:

1. **네트워크 설정**: 외부에서 접근하도록 설정 (SSH 포트 포워딩, 리버스 프록시)
2. **HTTPS 설정**: Let's Encrypt로 보안 강화
3. **확장 프로그램 설치**: VS Code 확장 프로그램 설치
4. **팀 환경 구축**: 여러 사용자를 위한 인스턴스 관리

---

*다음 글에서는 code-server의 아키텍처와 VS Code 통합 방식을 살펴봅니다.*
