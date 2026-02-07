---
layout: post
title: "Tauri 완벽 가이드 (08) - 시스템 통합"
date: 2026-02-07
permalink: /tauri-guide-08-system-integration/
author: Tauri Programme
categories: [웹 개발, 데스크톱]
tags: [Tauri, System Integration, Tray, Menu, Notifications]
original_url: "https://github.com/tauri-apps/tauri"
excerpt: "시스템 트레이, 메뉴, 알림 등 네이티브 기능 통합"
---

## 시스템 트레이

### 기본 트레이

```rust
use tauri::{CustomMenuItem, SystemTray, SystemTrayMenu, SystemTrayEvent};
use tauri::Manager;

fn main() {
    let quit = CustomMenuItem::new("quit".to_string(), "Quit");
    let show = CustomMenuItem::new("show".to_string(), "Show");

    let tray_menu = SystemTrayMenu::new()
        .add_item(show)
        .add_native_item(SystemTrayMenuItem::Separator)
        .add_item(quit);

    let system_tray = SystemTray::new().with_menu(tray_menu);

    tauri::Builder::default()
        .system_tray(system_tray)
        .on_system_tray_event(|app, event| match event {
            SystemTrayEvent::LeftClick {
                position: _,
                size: _,
                ..
            } => {
                let window = app.get_window("main").unwrap();
                window.show().unwrap();
            }
            SystemTrayEvent::MenuItemClick { id, .. } => {
                match id.as_str() {
                    "quit" => {
                        std::process::exit(0);
                    }
                    "show" => {
                        let window = app.get_window("main").unwrap();
                        window.show().unwrap();
                    }
                    _ => {}
                }
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .unwrap();
}
```

---

## 메뉴

### 애플리케이션 메뉴

```rust
use tauri::{CustomMenuItem, Menu, MenuItem, Submenu};

fn main() {
    let menu = Menu::new()
        .add_submenu(Submenu::new(
            "File",
            Menu::new()
                .add_item(CustomMenuItem::new("new", "New File").accelerator("Cmd+N"))
                .add_item(CustomMenuItem::new("open", "Open...").accelerator("Cmd+O"))
                .add_native_item(MenuItem::Separator)
                .add_native_item(MenuItem::Quit),
        ))
        .add_submenu(Submenu::new(
            "Edit",
            Menu::new()
                .add_native_item(MenuItem::Cut)
                .add_native_item(MenuItem::Copy)
                .add_native_item(MenuItem::Paste),
        ));

    tauri::Builder::default()
        .menu(menu)
        .on_menu_event(|event| {
            match event.menu_item_id() {
                "new" => {
                    // 새 파일 생성
                }
                "open" => {
                    // 파일 열기
                }
                _ => {}
            }
        })
        .run(tauri::generate_context!())
        .unwrap();
}
```

---

## 네이티브 알림

```typescript
import { sendNotification, isPermissionGranted, requestPermission } from '@tauri-apps/api/notification';

let permissionGranted = await isPermissionGranted();

if (!permissionGranted) {
    const permission = await requestPermission();
    permissionGranted = permission === 'granted';
}

if (permissionGranted) {
    sendNotification({
        title: 'New Message',
        body: 'You have 3 new messages',
        icon: 'icon.png'
    });
}
```

---

## 전역 단축키

```typescript
import { register, unregister } from '@tauri-apps/api/globalShortcut';

await register('CommandOrControl+Shift+T', () => {
    console.log('Global shortcut triggered');
});

// 앱 종료 시
await unregister('CommandOrControl+Shift+T');
```

---

## 다중 윈도우

```rust
use tauri::WindowBuilder;

#[tauri::command]
fn open_settings(app: tauri::AppHandle) {
    WindowBuilder::new(
        &app,
        "settings",
        tauri::WindowUrl::App("settings.html".into())
    )
    .title("Settings")
    .inner_size(600.0, 400.0)
    .build()
    .unwrap();
}
```

```typescript
import { WebviewWindow } from '@tauri-apps/api/window';

const settingsWindow = new WebviewWindow('settings', {
    url: 'settings.html',
    title: 'Settings',
    width: 600,
    height: 400
});

settingsWindow.once('tauri://created', () => {
    console.log('Window created');
});

settingsWindow.once('tauri://error', (e) => {
    console.error('Window error:', e);
});
```

---

*다음 글에서는 모바일 플랫폼 지원을 살펴봅니다.*
