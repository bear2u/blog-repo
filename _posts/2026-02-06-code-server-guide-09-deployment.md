---
layout: post
title: "code-server 완벽 가이드 (09) - 배포 및 운영"
date: 2026-02-06
permalink: /code-server-guide-09-deployment/
author: Coder
categories: [웹 개발, 원격 개발]
tags: [code-server, Docker, Kubernetes, Helm, 배포, 클라우드]
original_url: "https://github.com/coder/code-server"
excerpt: "Docker, Kubernetes, 클라우드를 활용한 code-server 배포 및 운영 가이드."
---

## Docker로 배포

### 기본 실행

```bash
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
# docker-compose.yml
version: "3.9"

services:
  code-server:
    image: codercom/code-server:latest
    container_name: code-server
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./project:/home/coder/project
      - ./config:/home/coder/.config
      - ./local:/home/coder/.local
    environment:
      - PASSWORD=yourSecurePassword
      - TZ=Asia/Seoul
    restart: unless-stopped
    user: "${UID}:${GID}"
```

실행:
```bash
export UID=$(id -u)
export GID=$(id -g)
docker-compose up -d
```

### 커스텀 Dockerfile

```dockerfile
# Dockerfile
FROM codercom/code-server:latest

# 추가 패키지 설치
USER root
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# 확장 프로그램 사전 설치
USER coder
RUN code-server --install-extension ms-python.python && \
    code-server --install-extension dbaeumer.vscode-eslint && \
    code-server --install-extension esbenp.prettier-vscode

# 설정 파일 복사
COPY config.yaml /home/coder/.config/code-server/config.yaml
```

빌드:
```bash
docker build -t my-code-server .
docker run -it -p 8080:8080 my-code-server
```

---

## Kubernetes (Helm)

### Helm 설치

```bash
# Helm 레포지토리 추가
helm repo add coder https://helm.coder.com
helm repo update
```

### 기본 설치

```bash
helm install code-server coder/code-server \
  --set persistence.enabled=true \
  --set persistence.size=10Gi \
  --set service.type=ClusterIP
```

### values.yaml로 커스터마이징

```yaml
# values.yaml
image:
  repository: codercom/code-server
  tag: latest
  pullPolicy: IfNotPresent

# 복제본 수
replicaCount: 1

# 서비스
service:
  type: ClusterIP
  port: 8080

# Ingress
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: code.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: code-server-tls
      hosts:
        - code.yourdomain.com

# 영구 볼륨
persistence:
  enabled: true
  size: 10Gi
  storageClass: standard
  accessMode: ReadWriteOnce

# 리소스 제한
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

# 환경 변수
extraEnvs:
  - name: PASSWORD
    valueFrom:
      secretKeyRef:
        name: code-server-secret
        key: password
  - name: TZ
    value: Asia/Seoul

# 보안 컨텍스트
securityContext:
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
```

설치:
```bash
# Secret 생성
kubectl create secret generic code-server-secret \
  --from-literal=password='yourSecurePassword'

# Helm 설치
helm install code-server coder/code-server -f values.yaml
```

### 상태 확인

```bash
# Pod 확인
kubectl get pods

# 로그 확인
kubectl logs -f deployment/code-server

# 서비스 확인
kubectl get svc code-server

# Ingress 확인
kubectl get ingress
```

### 포트 포워딩 (로컬 테스트)

```bash
kubectl port-forward svc/code-server 8080:8080

# http://localhost:8080 접속
```

---

## 클라우드 배포

### DigitalOcean

**원클릭 배포:**
1. [DigitalOcean Marketplace](https://marketplace.digitalocean.com/) 접속
2. "code-server" 검색
3. Droplet 생성
4. SSH 접속 후 비밀번호 확인:
   ```bash
   cat ~/.config/code-server/config.yaml
   ```

**수동 배포:**
```bash
# Droplet 생성 (Ubuntu 22.04)
# SSH 접속
ssh root@your-droplet-ip

# code-server 설치
curl -fsSL https://code-server.dev/install.sh | sh

# Caddy 설치
sudo apt install -y caddy

# Caddyfile 설정
echo "code.yourdomain.com {
  reverse_proxy 127.0.0.1:8080
}" | sudo tee /etc/caddy/Caddyfile

# 서비스 시작
sudo systemctl enable --now code-server@$USER
sudo systemctl enable --now caddy
```

### AWS EC2

```bash
# 1. EC2 인스턴스 생성 (Ubuntu 22.04)
# 2. Security Group: 80, 443, 22 포트 열기

# 3. SSH 접속
ssh -i key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# 4. code-server 설치
curl -fsSL https://code-server.dev/install.sh | sh

# 5. NGINX + Let's Encrypt
sudo apt install -y nginx certbot python3-certbot-nginx
# ... (NGINX 설정은 이전 챕터 참고)
```

### Google Cloud Platform

```bash
# gcloud CLI 사용
gcloud compute instances create code-server \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=e2-medium \
  --zone=us-central1-a \
  --tags=http-server,https-server

# 방화벽 규칙
gcloud compute firewall-rules create allow-http --allow tcp:80
gcloud compute firewall-rules create allow-https --allow tcp:443

# SSH 접속
gcloud compute ssh code-server --zone=us-central1-a

# code-server 설치
curl -fsSL https://code-server.dev/install.sh | sh
```

### Azure

```bash
# Azure CLI
az vm create \
  --resource-group MyResourceGroup \
  --name code-server \
  --image UbuntuLTS \
  --admin-username azureuser \
  --generate-ssh-keys

# 포트 열기
az vm open-port --port 80 --resource-group MyResourceGroup --name code-server
az vm open-port --port 443 --resource-group MyResourceGroup --name code-server

# SSH 접속
ssh azureuser@<public-ip>
```

---

## Railway

가장 쉬운 배포 방법!

1. [Railway.app](https://railway.app/) 접속
2. GitHub로 로그인
3. "New Project" → "Deploy code-server"
4. 자동 배포 완료!

---

## Heroku

```bash
# Heroku CLI 설치
curl https://cli-assets.heroku.com/install.sh | sh

# 로그인
heroku login

# 앱 생성
heroku create my-code-server

# Buildpack 추가
heroku buildpacks:add https://github.com/coder/heroku-buildpack-code-server

# 환경 변수 설정
heroku config:set PASSWORD=yourSecurePassword

# 배포
git push heroku main

# 접속
heroku open
```

---

## 모니터링

### Prometheus + Grafana

```yaml
# docker-compose.yml
version: "3.9"

services:
  code-server:
    image: codercom/code-server:latest
    ports:
      - "8080:8080"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### 로그 수집

```bash
# Systemd 로그
journalctl -u code-server@$USER -f

# Docker 로그
docker logs -f code-server

# Kubernetes 로그
kubectl logs -f deployment/code-server
```

---

## 백업 전략

### 1. 사용자 데이터 백업

```bash
# 백업
tar -czf code-server-backup-$(date +%Y%m%d).tar.gz \
  ~/.config/code-server \
  ~/.local/share/code-server

# 복원
tar -xzf code-server-backup-20260206.tar.gz -C ~
```

### 2. 자동 백업 스크립트

```bash
#!/bin/bash
# backup-code-server.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d-%H%M%S)

tar -czf "$BACKUP_DIR/code-server-$DATE.tar.gz" \
  ~/.config/code-server \
  ~/.local/share/code-server

# 7일 이상 된 백업 삭제
find $BACKUP_DIR -name "code-server-*.tar.gz" -mtime +7 -delete
```

Cron:
```bash
# 매일 새벽 2시
0 2 * * * /path/to/backup-code-server.sh
```

### 3. S3 백업

```bash
#!/bin/bash
# s3-backup.sh

BACKUP_FILE="code-server-$(date +%Y%m%d).tar.gz"

tar -czf "/tmp/$BACKUP_FILE" \
  ~/.config/code-server \
  ~/.local/share/code-server

aws s3 cp "/tmp/$BACKUP_FILE" "s3://my-bucket/backups/"
rm "/tmp/$BACKUP_FILE"
```

---

## 고가용성 (HA)

### 로드 밸런서 + 다중 인스턴스

```nginx
# NGINX 로드 밸런서
upstream code-servers {
    ip_hash;  # Sticky session
    server 10.0.1.10:8080;
    server 10.0.1.11:8080;
    server 10.0.1.12:8080;
}

server {
    listen 443 ssl http2;
    server_name code.yourdomain.com;

    location / {
        proxy_pass http://code-servers;
        proxy_set_header Host $http_host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
    }
}
```

### Shared Storage (NFS)

```bash
# NFS 서버
sudo apt install nfs-kernel-server
sudo mkdir -p /export/code-server
echo "/export/code-server *(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
sudo exportfs -ra

# 각 code-server 인스턴스
sudo mount nfs-server:/export/code-server ~/.local/share/code-server
```

---

## 성능 튜닝

### 1. Node.js 메모리 증가

```bash
# Systemd
sudo systemctl edit code-server@$USER
```

```ini
[Service]
Environment="NODE_OPTIONS=--max-old-space-size=4096"
```

### 2. CPU 제한 (Docker)

```yaml
services:
  code-server:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
```

### 3. 네트워크 최적화

```nginx
# NGINX
http {
    # Gzip 압축
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;

    # 캐싱
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=code_cache:10m max_size=1g inactive=60m;
}
```

---

## 보안 강화

### 1. Fail2Ban

```bash
# /etc/fail2ban/jail.local
[code-server]
enabled = true
filter = code-server
logpath = /var/log/syslog
maxretry = 3
bantime = 3600
```

### 2. IP 화이트리스트 (NGINX)

```nginx
server {
    location / {
        allow 203.0.113.0/24;
        allow 198.51.100.0/24;
        deny all;

        proxy_pass http://localhost:8080;
    }
}
```

---

*다음 글에서는 국제화, 커스터마이징, 베스트 프랙티스를 다룹니다.*
