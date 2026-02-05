---
layout: post
title: "Beads 완벽 가이드 (5) - 데몬 시스템"
date: 2025-02-04
permalink: /beads-guide-05-daemon/
author: Steve Yegge
categories: [AI 에이전트, Beads]
tags: [Beads, Daemon, RPC, Background Process, Auto-Sync]
original_url: "https://github.com/steveyegge/beads"
excerpt: "Beads의 백그라운드 데몬 아키텍처와 RPC 통신을 분석합니다."
---

## 데몬 개요

Beads는 **워크스페이스당 하나의 백그라운드 데몬**을 실행하여 자동 동기화, RPC 작업, 실시간 모니터링을 처리합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                 Per-Workspace Model (LSP-style)                  │
│                                                                  │
│    MCP Server (하나의 인스턴스)                                  │
│         ↓                                                        │
│    Per-Project Daemons (워크스페이스당 하나)                     │
│         ↓                                                        │
│    SQLite Databases (완전 격리)                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 데몬이 필요한 경우

### 데몬 활성화 (기본값)

| 시나리오 | 이점 |
|----------|------|
| **다중 에이전트 워크플로우** | 데이터베이스 락 충돌 방지 |
| **팀 협업** | 백그라운드에서 JSONL을 Git에 자동 동기화 |
| **긴 코딩 세션** | `bd sync` 잊어도 변경 저장 |
| **실시간 모니터링** | `bd watch` 및 상태 업데이트 활성화 |

### 데몬 비활성화

| 시나리오 | 비활성화 방법 |
|----------|---------------|
| **Git worktrees (sync-branch 없음)** | 자동 비활성화 |
| **CI/CD 파이프라인** | `BEADS_NO_DAEMON=true` |
| **오프라인 작업** | `--no-daemon` |
| **리소스 제한** | `BEADS_NO_DAEMON=true` |
| **결정론적 테스트** | 배타적 락 사용 |

---

## 데몬 아키텍처

```go
// internal/daemon/daemon.go

type Daemon struct {
    workspace  string
    store      storage.Store
    rpcServer  *RPCServer
    syncMgr    *SyncManager
    flushMgr   *FlushManager
    socketPath string
    pid        int
    logger     *log.Logger
}

func NewDaemon(workspace string) (*Daemon, error) {
    store, err := sqlite.NewStore(
        filepath.Join(workspace, ".beads", "beads.db"),
    )
    if err != nil {
        return nil, err
    }

    d := &Daemon{
        workspace:  workspace,
        store:      store,
        socketPath: filepath.Join(workspace, ".beads", "bd.sock"),
    }

    d.rpcServer = NewRPCServer(d)
    d.syncMgr = NewSyncManager(d)
    d.flushMgr = NewFlushManager(d)

    return d, nil
}

func (d *Daemon) Start() error {
    // PID 파일 생성
    if err := d.writePIDFile(); err != nil {
        return err
    }

    // Unix 소켓 리스닝 시작
    listener, err := net.Listen("unix", d.socketPath)
    if err != nil {
        return err
    }

    // RPC 서버 시작
    go d.rpcServer.Serve(listener)

    // 자동 동기화 시작
    go d.syncMgr.Start()

    // 플러시 매니저 시작
    go d.flushMgr.Start()

    d.logger.Printf("Daemon started for %s (pid %d)", d.workspace, os.Getpid())

    return nil
}
```

---

## RPC 프로토콜

### 프로토콜 정의

```go
// internal/rpc/protocol.go

type Request struct {
    ID     string          `json:"id"`
    Method string          `json:"method"`
    Params json.RawMessage `json:"params,omitempty"`
}

type Response struct {
    ID     string          `json:"id"`
    Result json.RawMessage `json:"result,omitempty"`
    Error  *RPCError       `json:"error,omitempty"`
}

type RPCError struct {
    Code    int    `json:"code"`
    Message string `json:"message"`
}

// 지원 메서드
const (
    MethodCreate     = "create"
    MethodList       = "list"
    MethodShow       = "show"
    MethodUpdate     = "update"
    MethodClose      = "close"
    MethodReady      = "ready"
    MethodDepAdd     = "dep.add"
    MethodDepRemove  = "dep.remove"
    MethodSync       = "sync"
    MethodPing       = "ping"
    MethodShutdown   = "shutdown"
)
```

### RPC 서버 구현

```go
// internal/rpc/server.go

type RPCServer struct {
    daemon   *Daemon
    handlers map[string]HandlerFunc
}

type HandlerFunc func(params json.RawMessage) (interface{}, error)

func NewRPCServer(d *Daemon) *RPCServer {
    s := &RPCServer{
        daemon:   d,
        handlers: make(map[string]HandlerFunc),
    }

    // 핸들러 등록
    s.handlers[MethodCreate] = s.handleCreate
    s.handlers[MethodList] = s.handleList
    s.handlers[MethodShow] = s.handleShow
    s.handlers[MethodUpdate] = s.handleUpdate
    s.handlers[MethodClose] = s.handleClose
    s.handlers[MethodReady] = s.handleReady
    s.handlers[MethodSync] = s.handleSync
    s.handlers[MethodPing] = s.handlePing

    return s
}

func (s *RPCServer) Serve(listener net.Listener) {
    for {
        conn, err := listener.Accept()
        if err != nil {
            continue
        }
        go s.handleConnection(conn)
    }
}

func (s *RPCServer) handleConnection(conn net.Conn) {
    defer conn.Close()

    decoder := json.NewDecoder(conn)
    encoder := json.NewEncoder(conn)

    for {
        var req Request
        if err := decoder.Decode(&req); err != nil {
            return
        }

        resp := s.processRequest(&req)
        encoder.Encode(resp)
    }
}

func (s *RPCServer) processRequest(req *Request) *Response {
    handler, ok := s.handlers[req.Method]
    if !ok {
        return &Response{
            ID: req.ID,
            Error: &RPCError{
                Code:    -32601,
                Message: "Method not found",
            },
        }
    }

    result, err := handler(req.Params)
    if err != nil {
        return &Response{
            ID: req.ID,
            Error: &RPCError{
                Code:    -32000,
                Message: err.Error(),
            },
        }
    }

    resultJSON, _ := json.Marshal(result)
    return &Response{
        ID:     req.ID,
        Result: resultJSON,
    }
}
```

---

## RPC 클라이언트

```go
// internal/rpc/client.go

type RPCClient struct {
    socketPath string
    conn       net.Conn
    encoder    *json.Encoder
    decoder    *json.Decoder
    nextID     int64
}

func NewRPCClient(socketPath string) (*RPCClient, error) {
    conn, err := net.Dial("unix", socketPath)
    if err != nil {
        return nil, err
    }

    return &RPCClient{
        socketPath: socketPath,
        conn:       conn,
        encoder:    json.NewEncoder(conn),
        decoder:    json.NewDecoder(conn),
    }, nil
}

func (c *RPCClient) Call(method string, params interface{}) (json.RawMessage, error) {
    id := atomic.AddInt64(&c.nextID, 1)

    paramsJSON, err := json.Marshal(params)
    if err != nil {
        return nil, err
    }

    req := Request{
        ID:     fmt.Sprintf("%d", id),
        Method: method,
        Params: paramsJSON,
    }

    if err := c.encoder.Encode(req); err != nil {
        return nil, err
    }

    var resp Response
    if err := c.decoder.Decode(&resp); err != nil {
        return nil, err
    }

    if resp.Error != nil {
        return nil, fmt.Errorf("RPC error: %s", resp.Error.Message)
    }

    return resp.Result, nil
}
```

---

## CLI에서 데몬 사용

```go
// cmd/bd/backend.go

func getBackend() Backend {
    // 데몬 소켓 확인
    socketPath := filepath.Join(".beads", "bd.sock")

    if _, err := os.Stat(socketPath); err == nil {
        // 데몬에 연결 시도
        client, err := rpc.NewRPCClient(socketPath)
        if err == nil {
            return &DaemonBackend{client: client}
        }
    }

    // 폴백: 직접 DB 접근
    store, err := sqlite.NewStore(".beads/beads.db")
    if err != nil {
        log.Fatal(err)
    }

    return &DirectBackend{store: store}
}
```

---

## 데몬 관리 명령어

```bash
# 실행 중인 데몬 목록
bd daemons list --json

# 출력:
# [
#   {
#     "workspace": "/Users/alice/projects/webapp",
#     "pid": 12345,
#     "socket": "/Users/alice/projects/webapp/.beads/bd.sock",
#     "version": "0.21.0",
#     "uptime_seconds": 3600
#   }
# ]

# 데몬 상태 확인
bd daemons status

# 데몬 중지
bd daemons stop

# 데몬 재시작
bd daemons restart

# 모든 데몬 중지
bd daemons stop --all
```

---

## 자동 시작

```go
// cmd/bd/autostart.go

func ensureDaemon() error {
    if os.Getenv("BEADS_NO_DAEMON") == "true" {
        return nil
    }

    socketPath := filepath.Join(".beads", "bd.sock")

    // 이미 실행 중인지 확인
    if isSocketAlive(socketPath) {
        return nil
    }

    // 새 데몬 프로세스 시작
    cmd := exec.Command(os.Args[0], "daemons", "start", "--background")
    cmd.Start()

    // 데몬 준비 대기
    return waitForDaemon(socketPath, 5*time.Second)
}
```

---

## Windows 지원

Windows에서는 Named Pipes를 사용합니다.

```go
// internal/daemon/daemon_windows.go

func getSocketPath(workspace string) string {
    // Windows Named Pipe
    return fmt.Sprintf(`\\.\pipe\beads-%s`, hashWorkspace(workspace))
}

func listen(path string) (net.Listener, error) {
    return winio.ListenPipe(path, nil)
}
```

---

*다음 글에서는 동기화 메커니즘을 살펴봅니다.*
