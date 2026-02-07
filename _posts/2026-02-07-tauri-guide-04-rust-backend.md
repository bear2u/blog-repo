---
layout: post
title: "Tauri 완벽 가이드 (04) - Rust Backend"
date: 2026-02-07
permalink: /tauri-guide-04-rust-backend/
author: Tauri Programme
categories: [웹 개발, 데스크톱]
tags: [Tauri, Rust, Backend, Commands, State Management]
original_url: "https://github.com/tauri-apps/tauri"
excerpt: "Tauri Rust Backend와 명령어 시스템 마스터하기"
---

## Rust Backend 개요

Tauri의 백엔드는 Rust로 작성되며, 프론트엔드에서 호출할 수 있는 **명령어(Commands)** 를 제공합니다. Rust의 성능과 안전성을 활용하여 시스템 API, 파일 작업, 네트워크 요청 등을 처리합니다.

---

## 명령어 시스템 (Commands)

### 기본 명령어

```rust
#[tauri::command]
fn simple_command() -> String {
    "Hello from Rust!".to_string()
}
```

**프론트엔드 호출:**

```typescript
import { invoke } from '@tauri-apps/api/tauri';

const message = await invoke('simple_command');
console.log(message); // "Hello from Rust!"
```

### 매개변수가 있는 명령어

```rust
#[tauri::command]
fn greet(name: String, age: i32) -> String {
    format!("{} is {} years old", name, age)
}
```

```typescript
const message = await invoke('greet', {
    name: 'Alice',
    age: 30
});
```

**중요**: JavaScript의 camelCase가 Rust의 snake_case로 자동 변환됩니다.

```typescript
// JavaScript
invoke('myCommand', { firstName: 'John' })

// Rust
#[tauri::command]
fn my_command(first_name: String) {}
```

---

## 반환 타입

### 간단한 타입

```rust
#[tauri::command]
fn get_number() -> i32 {
    42
}

#[tauri::command]
fn get_bool() -> bool {
    true
}

#[tauri::command]
fn get_array() -> Vec<String> {
    vec!["a".to_string(), "b".to_string()]
}
```

### 구조체 (Serde)

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct User {
    id: i32,
    name: String,
    email: String,
}

#[tauri::command]
fn get_user() -> User {
    User {
        id: 1,
        name: "Alice".to_string(),
        email: "alice@example.com".to_string(),
    }
}
```

```typescript
interface User {
    id: number;
    name: string;
    email: string;
}

const user = await invoke<User>('get_user');
console.log(user.name); // "Alice"
```

---

## 에러 처리

### Result 타입

```rust
#[tauri::command]
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Cannot divide by zero".to_string())
    } else {
        Ok(a / b)
    }
}
```

```typescript
try {
    const result = await invoke('divide', { a: 10, b: 0 });
} catch (error) {
    console.error(error); // "Cannot divide by zero"
}
```

### 커스텀 에러 타입

```rust
use serde::Serialize;

#[derive(Debug, Serialize)]
struct MyError {
    code: i32,
    message: String,
}

impl std::fmt::Display for MyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for MyError {}

#[tauri::command]
fn risky_operation() -> Result<String, MyError> {
    Err(MyError {
        code: 404,
        message: "Not found".to_string(),
    })
}
```

---

## 상태 관리 (State)

### 전역 상태

```rust
use std::sync::Mutex;
use tauri::State;

struct AppState {
    counter: Mutex<i32>,
}

#[tauri::command]
fn increment(state: State<AppState>) -> i32 {
    let mut counter = state.counter.lock().unwrap();
    *counter += 1;
    *counter
}

#[tauri::command]
fn get_count(state: State<AppState>) -> i32 {
    *state.counter.lock().unwrap()
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            counter: Mutex::new(0),
        })
        .invoke_handler(tauri::generate_handler![increment, get_count])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

```typescript
await invoke('increment'); // 1
await invoke('increment'); // 2
const count = await invoke('get_count'); // 2
```

### 복잡한 상태

```rust
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

struct Database {
    users: Arc<Mutex<HashMap<i32, User>>>,
}

impl Database {
    fn new() -> Self {
        Self {
            users: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn add_user(&self, user: User) {
        let mut users = self.users.lock().unwrap();
        users.insert(user.id, user);
    }

    fn get_user(&self, id: i32) -> Option<User> {
        let users = self.users.lock().unwrap();
        users.get(&id).cloned()
    }
}

#[tauri::command]
fn add_user(db: State<Database>, user: User) {
    db.add_user(user);
}

#[tauri::command]
fn get_user(db: State<Database>, id: i32) -> Option<User> {
    db.get_user(id)
}
```

---

## Window와 AppHandle

### Window 접근

```rust
use tauri::Window;

#[tauri::command]
fn close_window(window: Window) {
    window.close().unwrap();
}

#[tauri::command]
fn set_title(window: Window, title: String) {
    window.set_title(&title).unwrap();
}
```

### AppHandle 사용

```rust
use tauri::{AppHandle, Manager};

#[tauri::command]
fn create_new_window(app: AppHandle) {
    tauri::WindowBuilder::new(
        &app,
        "new-window",
        tauri::WindowUrl::App("index.html".into())
    )
    .title("New Window")
    .build()
    .unwrap();
}
```

---

## 비동기 명령어

### async/await

```rust
#[tauri::command]
async fn fetch_data(url: String) -> Result<String, String> {
    let response = reqwest::get(&url)
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;

    Ok(response)
}
```

**Cargo.toml:**

```toml
[dependencies]
reqwest = "0.11"
tokio = { version = "1", features = ["full"] }
```

### 병렬 처리

```rust
use tokio::task;

#[tauri::command]
async fn parallel_tasks() -> Vec<i32> {
    let task1 = task::spawn(async { 1 });
    let task2 = task::spawn(async { 2 });
    let task3 = task::spawn(async { 3 });

    let results = tokio::join!(task1, task2, task3);

    vec![
        results.0.unwrap(),
        results.1.unwrap(),
        results.2.unwrap(),
    ]
}
```

---

## 파일 시스템

### 파일 읽기

```rust
use std::fs;

#[tauri::command]
fn read_file(path: String) -> Result<String, String> {
    fs::read_to_string(path)
        .map_err(|e| e.to_string())
}
```

### 파일 쓰기

```rust
#[tauri::command]
fn write_file(path: String, content: String) -> Result<(), String> {
    fs::write(path, content)
        .map_err(|e| e.to_string())
}
```

### 경로 헬퍼

```rust
use tauri::api::path::{app_data_dir, app_log_dir};
use tauri::Config;

#[tauri::command]
fn get_app_dir(config: tauri::Config) -> Option<String> {
    app_data_dir(&config)
        .map(|p| p.display().to_string())
}
```

---

## 시스템 정보

```rust
use std::env;

#[tauri::command]
fn get_os() -> String {
    env::consts::OS.to_string()
}

#[tauri::command]
fn get_arch() -> String {
    env::consts::ARCH.to_string()
}

#[tauri::command]
fn get_env_var(key: String) -> Option<String> {
    env::var(key).ok()
}
```

---

## 명령어 등록

```rust
fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            simple_command,
            greet,
            divide,
            increment,
            get_count,
            fetch_data,
            read_file,
            write_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

---

## 라이프사이클 훅

### Setup Hook

```rust
fn main() {
    tauri::Builder::default()
        .setup(|app| {
            println!("App is starting...");
            // 초기화 로직
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 이벤트 리스너

```rust
use tauri::{RunEvent, Manager};

fn main() {
    tauri::Builder::default()
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| match event {
            RunEvent::Ready => {
                println!("App is ready!");
            }
            RunEvent::ExitRequested { api, .. } => {
                api.prevent_exit();
                println!("Preventing exit");
            }
            _ => {}
        });
}
```

---

## 멀티 윈도우

```rust
use tauri::{CustomMenuItem, Menu, MenuItem, Submenu};

fn main() {
    let menu = Menu::new()
        .add_submenu(Submenu::new(
            "File",
            Menu::new()
                .add_item(CustomMenuItem::new("new", "New Window"))
                .add_native_item(MenuItem::Separator)
                .add_native_item(MenuItem::Quit),
        ));

    tauri::Builder::default()
        .menu(menu)
        .on_menu_event(|event| {
            match event.menu_item_id() {
                "new" => {
                    // 새 윈도우 생성
                }
                _ => {}
            }
        })
        .run(tauri::generate_context!())
        .unwrap();
}
```

---

## 성능 최적화

### 1. Serde 최적화

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### 2. Rayon으로 병렬 처리

```rust
use rayon::prelude::*;

#[tauri::command]
fn process_large_data(data: Vec<i32>) -> Vec<i32> {
    data.par_iter()
        .map(|x| x * 2)
        .collect()
}
```

### 3. 메모리 최적화

```rust
#[tauri::command]
fn large_operation() {
    // Box로 힙 할당
    let large_data = Box::new([0u8; 1024 * 1024]);
    // 처리...
} // 자동으로 메모리 해제
```

---

*다음 글에서는 프론트엔드 통합과 @tauri-apps/api 사용법을 살펴봅니다.*
