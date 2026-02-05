---
layout: post
title: "UI-TARS 완벽 가이드 (6) - Operators"
date: 2025-02-04
permalink: /ui-tars-guide-06-operators/
author: ByteDance
categories: [AI 에이전트, UI-TARS]
tags: [UI-TARS, Operators, Browser, NutJS, ADB, Puppeteer]
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "UI-TARS의 Operator들을 분석합니다. Browser, NutJS, ADB Operator의 구현과 동작 원리를 살펴봅니다."
---

## Operator 개요

**Operator**는 GUI Agent가 실제로 화면을 제어하는 실행 엔진입니다. 각 플랫폼(브라우저, 데스크톱, 모바일)에 맞는 Operator를 제공합니다.

```
multimodal/gui-agent/operators/
├── src/
│   ├── types.ts           # 공통 인터페이스
│   ├── base.ts            # 기본 클래스
│   ├── browser/           # 브라우저 Operator
│   │   ├── BrowserOperator.ts
│   │   └── RemoteBrowserOperator.ts
│   ├── nutjs/             # 데스크톱 Operator
│   │   ├── NutJSOperator.ts
│   │   └── RemoteNutJSOperator.ts
│   └── adb/               # 모바일 Operator
│       └── ADBOperator.ts
└── index.ts
```

---

## Operator 인터페이스

### 공통 인터페이스 정의

```typescript
// operators/src/types.ts

export interface Operator {
  /**
   * Operator 초기화
   */
  initialize(): Promise<void>;

  /**
   * 스크린샷 캡처
   */
  screenshot(): Promise<Buffer>;

  /**
   * 클릭 (좌표는 정규화된 값 0-1 또는 픽셀 값)
   */
  click(coordinate: [number, number]): Promise<void>;

  /**
   * 더블 클릭
   */
  doubleClick(coordinate: [number, number]): Promise<void>;

  /**
   * 우클릭
   */
  rightClick(coordinate: [number, number]): Promise<void>;

  /**
   * 텍스트 입력
   */
  type(text: string): Promise<void>;

  /**
   * 키 입력
   */
  key(key: string): Promise<void>;

  /**
   * 단축키 입력
   */
  hotkey(keys: string[]): Promise<void>;

  /**
   * 스크롤
   */
  scroll(
    coordinate: [number, number],
    direction: 'up' | 'down' | 'left' | 'right',
    amount?: number
  ): Promise<void>;

  /**
   * 드래그
   */
  drag(
    from: [number, number],
    to: [number, number]
  ): Promise<void>;

  /**
   * 화면 크기 조회
   */
  getScreenSize(): Promise<{ width: number; height: number }>;

  /**
   * 리소스 정리
   */
  cleanup(): Promise<void>;
}

export interface OperatorOptions {
  screenshotResize?: boolean;
  resizeWidth?: number;
  resizeHeight?: number;
}
```

### 기본 클래스

```typescript
// operators/src/base.ts

export abstract class BaseOperator implements Operator {
  protected screenSize: { width: number; height: number } = {
    width: 1920,
    height: 1080
  };
  protected options: OperatorOptions;

  constructor(options: OperatorOptions = {}) {
    this.options = options;
  }

  abstract initialize(): Promise<void>;
  abstract screenshot(): Promise<Buffer>;
  abstract click(coordinate: [number, number]): Promise<void>;
  abstract type(text: string): Promise<void>;
  abstract key(key: string): Promise<void>;
  abstract cleanup(): Promise<void>;

  /**
   * 정규화된 좌표를 픽셀 좌표로 변환
   */
  protected toPixelCoordinate(
    coord: [number, number]
  ): [number, number] {
    // 이미 픽셀 좌표인 경우
    if (coord[0] > 1 || coord[1] > 1) {
      return coord;
    }

    // 정규화된 좌표를 변환
    return [
      Math.round(coord[0] * this.screenSize.width),
      Math.round(coord[1] * this.screenSize.height)
    ];
  }

  /**
   * 스크린샷 리사이즈
   */
  protected async resizeScreenshot(buffer: Buffer): Promise<Buffer> {
    if (!this.options.screenshotResize) {
      return buffer;
    }

    const sharp = (await import('sharp')).default;
    return sharp(buffer)
      .resize(
        this.options.resizeWidth || 1280,
        this.options.resizeHeight || 720,
        { fit: 'inside' }
      )
      .toBuffer();
  }

  async getScreenSize(): Promise<{ width: number; height: number }> {
    return this.screenSize;
  }

  // 기본 구현 (서브클래스에서 오버라이드 가능)
  async doubleClick(coordinate: [number, number]): Promise<void> {
    await this.click(coordinate);
    await this.click(coordinate);
  }

  async rightClick(coordinate: [number, number]): Promise<void> {
    // 서브클래스에서 구현
    throw new Error('Not implemented');
  }

  async hotkey(keys: string[]): Promise<void> {
    // 서브클래스에서 구현
    throw new Error('Not implemented');
  }

  async scroll(
    coordinate: [number, number],
    direction: 'up' | 'down' | 'left' | 'right',
    amount?: number
  ): Promise<void> {
    // 서브클래스에서 구현
    throw new Error('Not implemented');
  }

  async drag(from: [number, number], to: [number, number]): Promise<void> {
    // 서브클래스에서 구현
    throw new Error('Not implemented');
  }
}
```

---

## Browser Operator

### Puppeteer 기반 브라우저 제어

```typescript
// operators/src/browser/BrowserOperator.ts

import puppeteer, { Browser, Page } from 'puppeteer';
import { BaseOperator, OperatorOptions } from '../base';

export interface BrowserOperatorOptions extends OperatorOptions {
  headless?: boolean;
  executablePath?: string;
  userAgent?: string;
  viewport?: { width: number; height: number };
  proxy?: string;
  searchEngine?: 'google' | 'bing' | 'baidu';
}

export class BrowserOperator extends BaseOperator {
  private browser: Browser | null = null;
  private page: Page | null = null;
  private options: BrowserOperatorOptions;

  constructor(options: BrowserOperatorOptions = {}) {
    super(options);
    this.options = {
      headless: false,
      viewport: { width: 1280, height: 720 },
      searchEngine: 'google',
      ...options
    };
  }

  async initialize(): Promise<void> {
    const launchOptions: puppeteer.LaunchOptions = {
      headless: this.options.headless,
      executablePath: this.options.executablePath,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        `--window-size=${this.options.viewport!.width},${this.options.viewport!.height}`
      ]
    };

    // 프록시 설정
    if (this.options.proxy) {
      launchOptions.args!.push(`--proxy-server=${this.options.proxy}`);
    }

    this.browser = await puppeteer.launch(launchOptions);
    this.page = await this.browser.newPage();

    // 뷰포트 설정
    await this.page.setViewport(this.options.viewport!);

    // User-Agent 설정
    if (this.options.userAgent) {
      await this.page.setUserAgent(this.options.userAgent);
    }

    // 화면 크기 업데이트
    this.screenSize = this.options.viewport!;

    // 초기 페이지 로드
    const searchUrl = this.getSearchEngineUrl();
    await this.page.goto(searchUrl, { waitUntil: 'networkidle0' });
  }

  private getSearchEngineUrl(): string {
    switch (this.options.searchEngine) {
      case 'bing':
        return 'https://www.bing.com';
      case 'baidu':
        return 'https://www.baidu.com';
      case 'google':
      default:
        return 'https://www.google.com';
    }
  }

  async screenshot(): Promise<Buffer> {
    if (!this.page) throw new Error('Browser not initialized');

    const buffer = await this.page.screenshot({
      type: 'png',
      fullPage: false
    });

    return this.resizeScreenshot(buffer as Buffer);
  }

  async click(coordinate: [number, number]): Promise<void> {
    if (!this.page) throw new Error('Browser not initialized');

    const [x, y] = this.toPixelCoordinate(coordinate);
    await this.page.mouse.click(x, y);
  }

  async doubleClick(coordinate: [number, number]): Promise<void> {
    if (!this.page) throw new Error('Browser not initialized');

    const [x, y] = this.toPixelCoordinate(coordinate);
    await this.page.mouse.click(x, y, { clickCount: 2 });
  }

  async rightClick(coordinate: [number, number]): Promise<void> {
    if (!this.page) throw new Error('Browser not initialized');

    const [x, y] = this.toPixelCoordinate(coordinate);
    await this.page.mouse.click(x, y, { button: 'right' });
  }

  async type(text: string): Promise<void> {
    if (!this.page) throw new Error('Browser not initialized');

    await this.page.keyboard.type(text, { delay: 50 });
  }

  async key(key: string): Promise<void> {
    if (!this.page) throw new Error('Browser not initialized');

    // 키 이름 매핑
    const keyMap: Record<string, string> = {
      'enter': 'Enter',
      'return': 'Enter',
      'tab': 'Tab',
      'escape': 'Escape',
      'esc': 'Escape',
      'backspace': 'Backspace',
      'delete': 'Delete',
      'up': 'ArrowUp',
      'down': 'ArrowDown',
      'left': 'ArrowLeft',
      'right': 'ArrowRight',
      'home': 'Home',
      'end': 'End',
      'pageup': 'PageUp',
      'pagedown': 'PageDown'
    };

    const mappedKey = keyMap[key.toLowerCase()] || key;
    await this.page.keyboard.press(mappedKey);
  }

  async hotkey(keys: string[]): Promise<void> {
    if (!this.page) throw new Error('Browser not initialized');

    // 수정자 키 매핑
    const modifierMap: Record<string, string> = {
      'ctrl': 'Control',
      'control': 'Control',
      'cmd': 'Meta',
      'command': 'Meta',
      'meta': 'Meta',
      'alt': 'Alt',
      'option': 'Alt',
      'shift': 'Shift'
    };

    // 수정자 키 분리
    const modifiers: string[] = [];
    let mainKey = '';

    for (const key of keys) {
      const lowerKey = key.toLowerCase();
      if (modifierMap[lowerKey]) {
        modifiers.push(modifierMap[lowerKey]);
      } else {
        mainKey = key;
      }
    }

    // 수정자 키 누르기
    for (const modifier of modifiers) {
      await this.page.keyboard.down(modifier);
    }

    // 메인 키 누르기
    await this.page.keyboard.press(mainKey);

    // 수정자 키 떼기
    for (const modifier of modifiers.reverse()) {
      await this.page.keyboard.up(modifier);
    }
  }

  async scroll(
    coordinate: [number, number],
    direction: 'up' | 'down' | 'left' | 'right',
    amount: number = 300
  ): Promise<void> {
    if (!this.page) throw new Error('Browser not initialized');

    const [x, y] = this.toPixelCoordinate(coordinate);

    // 마우스 위치 이동
    await this.page.mouse.move(x, y);

    // 스크롤 방향에 따른 델타 계산
    let deltaX = 0;
    let deltaY = 0;

    switch (direction) {
      case 'up':
        deltaY = -amount;
        break;
      case 'down':
        deltaY = amount;
        break;
      case 'left':
        deltaX = -amount;
        break;
      case 'right':
        deltaX = amount;
        break;
    }

    await this.page.mouse.wheel({ deltaX, deltaY });
  }

  async drag(from: [number, number], to: [number, number]): Promise<void> {
    if (!this.page) throw new Error('Browser not initialized');

    const [fromX, fromY] = this.toPixelCoordinate(from);
    const [toX, toY] = this.toPixelCoordinate(to);

    await this.page.mouse.move(fromX, fromY);
    await this.page.mouse.down();
    await this.page.mouse.move(toX, toY, { steps: 10 });
    await this.page.mouse.up();
  }

  /**
   * URL로 이동
   */
  async navigate(url: string): Promise<void> {
    if (!this.page) throw new Error('Browser not initialized');

    await this.page.goto(url, { waitUntil: 'networkidle0' });
  }

  /**
   * 현재 URL 조회
   */
  async getCurrentUrl(): Promise<string> {
    if (!this.page) throw new Error('Browser not initialized');

    return this.page.url();
  }

  async cleanup(): Promise<void> {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
      this.page = null;
    }
  }
}
```

### 원격 브라우저 Operator

```typescript
// operators/src/browser/RemoteBrowserOperator.ts

import { connect, Browser, Page } from 'puppeteer';
import { BaseOperator } from '../base';

export interface RemoteBrowserOptions {
  cdpEndpoint: string;  // Chrome DevTools Protocol 엔드포인트
  targetId?: string;    // 특정 탭 ID
}

export class RemoteBrowserOperator extends BaseOperator {
  private browser: Browser | null = null;
  private page: Page | null = null;
  private options: RemoteBrowserOptions;

  constructor(options: RemoteBrowserOptions) {
    super();
    this.options = options;
  }

  async initialize(): Promise<void> {
    // CDP를 통해 원격 브라우저에 연결
    this.browser = await connect({
      browserWSEndpoint: this.options.cdpEndpoint
    });

    // 특정 탭에 연결하거나 첫 번째 탭 사용
    const pages = await this.browser.pages();

    if (this.options.targetId) {
      this.page = pages.find(
        p => p.target().targetId() === this.options.targetId
      ) || pages[0];
    } else {
      this.page = pages[0] || await this.browser.newPage();
    }

    // 화면 크기 조회
    const viewport = this.page.viewport();
    if (viewport) {
      this.screenSize = { width: viewport.width, height: viewport.height };
    }
  }

  async screenshot(): Promise<Buffer> {
    if (!this.page) throw new Error('Browser not initialized');
    return (await this.page.screenshot({ type: 'png' })) as Buffer;
  }

  // ... 나머지 메서드는 BrowserOperator와 동일

  async cleanup(): Promise<void> {
    if (this.browser) {
      this.browser.disconnect();
      this.browser = null;
      this.page = null;
    }
  }
}
```

---

## NutJS Operator

### 데스크톱 자동화

```typescript
// operators/src/nutjs/NutJSOperator.ts

import {
  mouse,
  keyboard,
  screen,
  Point,
  Button,
  Key
} from '@nut-tree-fork/nut-js';
import { BaseOperator } from '../base';

export class NutJSOperator extends BaseOperator {
  constructor() {
    super();

    // NutJS 설정
    mouse.config.autoDelayMs = 100;
    keyboard.config.autoDelayMs = 50;
  }

  async initialize(): Promise<void> {
    // 화면 크기 조회
    const screenWidth = await screen.width();
    const screenHeight = await screen.height();

    this.screenSize = {
      width: screenWidth,
      height: screenHeight
    };
  }

  async screenshot(): Promise<Buffer> {
    // 전체 화면 캡처
    const image = await screen.grab();

    // RGB 데이터를 PNG로 변환
    const { Jimp } = await import('jimp');
    const jimpImage = new Jimp({
      width: image.width,
      height: image.height,
      data: Buffer.from(image.data)
    });

    return jimpImage.getBuffer('image/png');
  }

  async click(coordinate: [number, number]): Promise<void> {
    const [x, y] = this.toPixelCoordinate(coordinate);

    await mouse.setPosition(new Point(x, y));
    await mouse.click(Button.LEFT);
  }

  async doubleClick(coordinate: [number, number]): Promise<void> {
    const [x, y] = this.toPixelCoordinate(coordinate);

    await mouse.setPosition(new Point(x, y));
    await mouse.doubleClick(Button.LEFT);
  }

  async rightClick(coordinate: [number, number]): Promise<void> {
    const [x, y] = this.toPixelCoordinate(coordinate);

    await mouse.setPosition(new Point(x, y));
    await mouse.click(Button.RIGHT);
  }

  async type(text: string): Promise<void> {
    await keyboard.type(text);
  }

  async key(key: string): Promise<void> {
    const nutKey = this.mapKey(key);
    await keyboard.pressKey(nutKey);
    await keyboard.releaseKey(nutKey);
  }

  async hotkey(keys: string[]): Promise<void> {
    const nutKeys = keys.map(k => this.mapKey(k));

    // 모든 키 누르기
    for (const key of nutKeys) {
      await keyboard.pressKey(key);
    }

    // 모든 키 떼기 (역순)
    for (const key of nutKeys.reverse()) {
      await keyboard.releaseKey(key);
    }
  }

  private mapKey(key: string): Key {
    const keyMap: Record<string, Key> = {
      'enter': Key.Enter,
      'return': Key.Enter,
      'tab': Key.Tab,
      'escape': Key.Escape,
      'esc': Key.Escape,
      'backspace': Key.Backspace,
      'delete': Key.Delete,
      'up': Key.Up,
      'down': Key.Down,
      'left': Key.Left,
      'right': Key.Right,
      'home': Key.Home,
      'end': Key.End,
      'pageup': Key.PageUp,
      'pagedown': Key.PageDown,
      'ctrl': Key.LeftControl,
      'control': Key.LeftControl,
      'alt': Key.LeftAlt,
      'shift': Key.LeftShift,
      'cmd': Key.LeftCmd,
      'command': Key.LeftCmd,
      'meta': Key.LeftCmd,
      'space': Key.Space,
      'a': Key.A,
      'b': Key.B,
      'c': Key.C,
      // ... 나머지 알파벳
      'f1': Key.F1,
      'f2': Key.F2,
      // ... 나머지 기능키
    };

    return keyMap[key.toLowerCase()] || Key.Space;
  }

  async scroll(
    coordinate: [number, number],
    direction: 'up' | 'down' | 'left' | 'right',
    amount: number = 3
  ): Promise<void> {
    const [x, y] = this.toPixelCoordinate(coordinate);

    // 마우스 위치 이동
    await mouse.setPosition(new Point(x, y));

    // 스크롤 실행
    switch (direction) {
      case 'up':
        await mouse.scrollUp(amount);
        break;
      case 'down':
        await mouse.scrollDown(amount);
        break;
      case 'left':
        await mouse.scrollLeft(amount);
        break;
      case 'right':
        await mouse.scrollRight(amount);
        break;
    }
  }

  async drag(from: [number, number], to: [number, number]): Promise<void> {
    const [fromX, fromY] = this.toPixelCoordinate(from);
    const [toX, toY] = this.toPixelCoordinate(to);

    await mouse.setPosition(new Point(fromX, fromY));
    await mouse.pressButton(Button.LEFT);
    await mouse.move([new Point(toX, toY)]);
    await mouse.releaseButton(Button.LEFT);
  }

  async cleanup(): Promise<void> {
    // NutJS는 별도 정리 불필요
  }
}
```

### 원격 컴퓨터 Operator

```typescript
// operators/src/nutjs/RemoteNutJSOperator.ts

import { BaseOperator } from '../base';

export interface ProxyClient {
  screenshot(): Promise<Buffer>;
  click(x: number, y: number): Promise<void>;
  type(text: string): Promise<void>;
  key(key: string): Promise<void>;
  getScreenSize(): Promise<{ width: number; height: number }>;
}

export class RemoteNutJSOperator extends BaseOperator {
  private client: ProxyClient;

  constructor(options: { proxyClient: ProxyClient }) {
    super();
    this.client = options.proxyClient;
  }

  async initialize(): Promise<void> {
    this.screenSize = await this.client.getScreenSize();
  }

  async screenshot(): Promise<Buffer> {
    return this.client.screenshot();
  }

  async click(coordinate: [number, number]): Promise<void> {
    const [x, y] = this.toPixelCoordinate(coordinate);
    await this.client.click(x, y);
  }

  async type(text: string): Promise<void> {
    await this.client.type(text);
  }

  async key(key: string): Promise<void> {
    await this.client.key(key);
  }

  async cleanup(): Promise<void> {
    // 프록시 연결 해제
  }
}
```

---

## ADB Operator

### 안드로이드 디바이스 제어

```typescript
// operators/src/adb/ADBOperator.ts

import { exec } from 'child_process';
import { promisify } from 'util';
import { BaseOperator } from '../base';

const execAsync = promisify(exec);

export interface ADBOperatorOptions {
  deviceId?: string;      // 특정 디바이스 ID
  adbPath?: string;       // ADB 실행 파일 경로
}

export class ADBOperator extends BaseOperator {
  private deviceId: string | null = null;
  private adbPath: string;

  constructor(options: ADBOperatorOptions = {}) {
    super();
    this.adbPath = options.adbPath || 'adb';
    this.deviceId = options.deviceId || null;
  }

  private get deviceFlag(): string {
    return this.deviceId ? `-s ${this.deviceId}` : '';
  }

  private async adb(command: string): Promise<string> {
    const { stdout } = await execAsync(
      `${this.adbPath} ${this.deviceFlag} ${command}`
    );
    return stdout.trim();
  }

  async initialize(): Promise<void> {
    // 디바이스 확인
    const devices = await this.adb('devices');
    const lines = devices.split('\n').slice(1);
    const connected = lines.filter(l => l.includes('device'));

    if (connected.length === 0) {
      throw new Error('No Android devices connected');
    }

    // 첫 번째 디바이스 선택
    if (!this.deviceId) {
      this.deviceId = connected[0].split('\t')[0];
    }

    // 화면 크기 조회
    const sizeOutput = await this.adb('shell wm size');
    const match = sizeOutput.match(/(\d+)x(\d+)/);

    if (match) {
      this.screenSize = {
        width: parseInt(match[1]),
        height: parseInt(match[2])
      };
    }
  }

  async screenshot(): Promise<Buffer> {
    // 스크린샷 캡처 및 전송
    const tempPath = '/sdcard/screenshot.png';

    await this.adb(`shell screencap -p ${tempPath}`);
    const { stdout } = await execAsync(
      `${this.adbPath} ${this.deviceFlag} exec-out cat ${tempPath}`,
      { encoding: 'buffer', maxBuffer: 50 * 1024 * 1024 }
    );

    return stdout as unknown as Buffer;
  }

  async click(coordinate: [number, number]): Promise<void> {
    const [x, y] = this.toPixelCoordinate(coordinate);
    await this.adb(`shell input tap ${x} ${y}`);
  }

  async doubleClick(coordinate: [number, number]): Promise<void> {
    const [x, y] = this.toPixelCoordinate(coordinate);

    // 빠른 연속 탭
    await this.adb(`shell "input tap ${x} ${y} && input tap ${x} ${y}"`);
  }

  async type(text: string): Promise<void> {
    // 공백과 특수문자 이스케이프
    const escaped = text
      .replace(/\\/g, '\\\\')
      .replace(/"/g, '\\"')
      .replace(/ /g, '%s');

    await this.adb(`shell input text "${escaped}"`);
  }

  async key(key: string): Promise<void> {
    const keyCode = this.mapKeyCode(key);
    await this.adb(`shell input keyevent ${keyCode}`);
  }

  private mapKeyCode(key: string): number {
    const keyCodeMap: Record<string, number> = {
      'home': 3,
      'back': 4,
      'call': 5,
      'endcall': 6,
      'dpad_up': 19,
      'dpad_down': 20,
      'dpad_left': 21,
      'dpad_right': 22,
      'dpad_center': 23,
      'volume_up': 24,
      'volume_down': 25,
      'power': 26,
      'camera': 27,
      'clear': 28,
      'enter': 66,
      'del': 67,
      'backspace': 67,
      'tab': 61,
      'space': 62,
      'escape': 111,
      'menu': 82,
      'search': 84,
      'app_switch': 187
    };

    return keyCodeMap[key.toLowerCase()] || 0;
  }

  async scroll(
    coordinate: [number, number],
    direction: 'up' | 'down' | 'left' | 'right',
    amount: number = 500
  ): Promise<void> {
    const [x, y] = this.toPixelCoordinate(coordinate);

    let endX = x;
    let endY = y;

    switch (direction) {
      case 'up':
        endY = y - amount;
        break;
      case 'down':
        endY = y + amount;
        break;
      case 'left':
        endX = x - amount;
        break;
      case 'right':
        endX = x + amount;
        break;
    }

    await this.adb(`shell input swipe ${x} ${y} ${endX} ${endY} 300`);
  }

  async drag(from: [number, number], to: [number, number]): Promise<void> {
    const [fromX, fromY] = this.toPixelCoordinate(from);
    const [toX, toY] = this.toPixelCoordinate(to);

    await this.adb(`shell input swipe ${fromX} ${fromY} ${toX} ${toY} 500`);
  }

  /**
   * 앱 실행
   */
  async launchApp(packageName: string): Promise<void> {
    await this.adb(
      `shell monkey -p ${packageName} -c android.intent.category.LAUNCHER 1`
    );
  }

  /**
   * 현재 액티비티 조회
   */
  async getCurrentActivity(): Promise<string> {
    const output = await this.adb(
      'shell dumpsys activity activities | grep mResumedActivity'
    );
    return output;
  }

  async cleanup(): Promise<void> {
    // ADB는 별도 정리 불필요
  }
}
```

---

## Operator 팩토리

```typescript
// operators/src/index.ts

import { Operator } from './types';
import { BrowserOperator, BrowserOperatorOptions } from './browser/BrowserOperator';
import { RemoteBrowserOperator } from './browser/RemoteBrowserOperator';
import { NutJSOperator } from './nutjs/NutJSOperator';
import { RemoteNutJSOperator, ProxyClient } from './nutjs/RemoteNutJSOperator';
import { ADBOperator, ADBOperatorOptions } from './adb/ADBOperator';

export type OperatorType =
  | 'local-browser'
  | 'remote-browser'
  | 'local-computer'
  | 'remote-computer'
  | 'adb';

export interface OperatorFactoryOptions {
  type: OperatorType;
  browser?: BrowserOperatorOptions;
  cdpEndpoint?: string;
  proxyClient?: ProxyClient;
  adb?: ADBOperatorOptions;
}

export function createOperator(options: OperatorFactoryOptions): Operator {
  switch (options.type) {
    case 'local-browser':
      return new BrowserOperator(options.browser);

    case 'remote-browser':
      if (!options.cdpEndpoint) {
        throw new Error('cdpEndpoint is required for remote-browser');
      }
      return new RemoteBrowserOperator({ cdpEndpoint: options.cdpEndpoint });

    case 'local-computer':
      return new NutJSOperator();

    case 'remote-computer':
      if (!options.proxyClient) {
        throw new Error('proxyClient is required for remote-computer');
      }
      return new RemoteNutJSOperator({ proxyClient: options.proxyClient });

    case 'adb':
      return new ADBOperator(options.adb);

    default:
      throw new Error(`Unknown operator type: ${options.type}`);
  }
}

export {
  Operator,
  BrowserOperator,
  RemoteBrowserOperator,
  NutJSOperator,
  RemoteNutJSOperator,
  ADBOperator
};
```

---

*다음 글에서는 Tarko 프레임워크를 분석합니다.*
