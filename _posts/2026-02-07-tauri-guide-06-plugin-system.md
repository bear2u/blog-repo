---
layout: post
title: "Tauri 완벽 가이드 (06) - 플러그인 시스템"
date: 2026-02-07
permalink: /tauri-guide-06-plugin-system/
author: Tauri Programme
categories: [웹 개발, 데스크톱]
tags: [Tauri, Plugins, Extensions, Rust]
original_url: "https://github.com/tauri-apps/tauri"
excerpt: "Tauri 플러그인 시스템으로 기능 확장하기"
---

## 플러그인 시스템 개요

Tauri 플러그인은 앱에 추가 기능을 제공하는 재사용 가능한 모듈입니다. 공식 플러그인과 커스텀 플러그인을 모두 사용할 수 있습니다.

---

## 공식 플러그인

### 주요 공식 플러그인

| 플러그인 | 기능 |
|---------|------|
| `tauri-plugin-fs` | 파일 시스템 API |
| `tauri-plugin-shell` | 셸 명령 실행 |
| `tauri-plugin-http` | HTTP 클라이언트 |
| `tauri-plugin-sql` | SQL 데이터베이스 |
| `tauri-plugin-store` | 키-값 저장소 |
| `tauri-plugin-window-state` | 윈도우 상태 저장 |
| `tauri-plugin-log` | 로깅 |
| `tauri-plugin-upload` | 파일 업로드 |

---

## 플러그인 설치

### 1. SQL 플러그인 예제

```toml
# Cargo.toml
[dependencies]
tauri-plugin-sql = { git = "https://github.com/tauri-apps/plugins-workspace", features = ["sqlite"] }
```

```rust
// main.rs
use tauri_plugin_sql::{TauriSql, Migration, MigrationKind};

fn main() {
    let migrations = vec![
        Migration {
            version: 1,
            description: "create_initial_tables",
            sql: "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
            kind: MigrationKind::Up,
        }
    ];

    tauri::Builder::default()
        .plugin(
            TauriSql::default()
                .add_migrations("sqlite:app.db", migrations)
        )
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**프론트엔드 사용:**

```typescript
import Database from 'tauri-plugin-sql-api';

const db = await Database.load('sqlite:app.db');

await db.execute('INSERT INTO users (name) VALUES (?)', ['Alice']);

const users = await db.select('SELECT * FROM users');
```

---

## 커스텀 플러그인 생성

### 기본 구조

```rust
use tauri::{
    plugin::{Builder, TauriPlugin},
    Runtime,
};

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("my-plugin")
        .invoke_handler(tauri::generate_handler![
            my_plugin_command
        ])
        .setup(|app| {
            // 플러그인 초기화
            Ok(())
        })
        .build()
}

#[tauri::command]
fn my_plugin_command(message: String) -> String {
    format!("Plugin received: {}", message)
}
```

### 플러그인 등록

```rust
mod my_plugin;

fn main() {
    tauri::Builder::default()
        .plugin(my_plugin::init())
        .run(tauri::generate_context!())
        .unwrap();
}
```

---

## 실전 예제: Logger 플러그인

```rust
// plugins/logger.rs
use tauri::{
    plugin::{Builder, TauriPlugin},
    Runtime, Manager,
};
use std::fs::OpenOptions;
use std::io::Write;

pub struct Logger {
    log_file: String,
}

impl Logger {
    fn log(&self, level: &str, message: &str) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file)
            .unwrap();

        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        writeln!(file, "[{}] {}: {}", timestamp, level, message).unwrap();
    }
}

#[tauri::command]
fn log_info(logger: tauri::State<Logger>, message: String) {
    logger.log("INFO", &message);
}

#[tauri::command]
fn log_error(logger: tauri::State<Logger>, message: String) {
    logger.log("ERROR", &message);
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("logger")
        .invoke_handler(tauri::generate_handler![log_info, log_error])
        .setup(|app| {
            app.manage(Logger {
                log_file: "app.log".to_string(),
            });
            Ok(())
        })
        .build()
}
```

**프론트엔드:**

```typescript
import { invoke } from '@tauri-apps/api/tauri';

await invoke('plugin:logger|log_info', {
    message: 'Application started'
});

await invoke('plugin:logger|log_error', {
    message: 'An error occurred'
});
```

---

*다음 글에서는 Tauri 앱 번들링과 배포를 살펴봅니다.*
