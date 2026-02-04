---
layout: post
title: "UI-TARS 완벽 가이드 (5) - GUI Agent SDK"
date: 2025-02-04
permalink: /ui-tars-guide-05-gui-agent/
author: ByteDance
category: AI
tags: [UI-TARS, GUI Agent, Action Parser, SDK, Vision Model]
series: ui-tars-guide
part: 5
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "GUI Agent SDK의 액션 파서와 에이전트 SDK 구조를 분석합니다. LLM 출력을 구조화된 액션으로 변환하는 과정을 살펴봅니다."
---

## GUI Agent SDK 개요

**GUI Agent SDK**는 Vision-Language Model을 활용하여 스크린샷 기반으로 UI를 자동화하는 핵심 모듈입니다.

```
multimodal/gui-agent/
├── action-parser/           # LLM 출력 파싱
│   └── src/
│       ├── index.ts         # 파서 진입점
│       ├── parsers/         # 형식별 파서
│       └── types.ts         # 액션 타입
├── agent-sdk/               # GUIAgent 클래스
│   └── src/
│       ├── GUIAgent.ts      # 메인 클래스
│       ├── loop.ts          # 에이전트 루프
│       └── prompts/         # 시스템 프롬프트
└── operators/               # 플랫폼별 Operator
    └── src/
        ├── browser/         # 브라우저 Operator
        ├── nutjs/           # 데스크톱 Operator
        └── adb/             # 모바일 Operator
```

---

## Action Parser

### 액션 타입 정의

```typescript
// action-parser/src/types.ts

export enum ActionType {
  // 마우스 액션
  CLICK = 'click',
  DOUBLE_CLICK = 'double_click',
  RIGHT_CLICK = 'right_click',
  DRAG = 'drag',
  SCROLL = 'scroll',
  HOVER = 'hover',

  // 키보드 액션
  TYPE = 'type',
  KEY = 'key',
  HOTKEY = 'hotkey',

  // 대기 액션
  WAIT = 'wait',

  // 종료 액션
  FINISHED = 'finished',
  CALL_USER = 'call_user'
}

export interface BaseAction {
  type: ActionType;
  thought?: string;      // 모델의 추론 과정
  confidence?: number;   // 신뢰도 (0-1)
}

export interface ClickAction extends BaseAction {
  type: ActionType.CLICK;
  coordinate: [number, number];
  element?: string;      // 요소 설명
}

export interface TypeAction extends BaseAction {
  type: ActionType.TYPE;
  text: string;
  coordinate?: [number, number];
}

export interface KeyAction extends BaseAction {
  type: ActionType.KEY;
  key: string;           // Enter, Escape, Tab 등
}

export interface HotkeyAction extends BaseAction {
  type: ActionType.HOTKEY;
  keys: string[];        // ['ctrl', 'c'] 등
}

export interface ScrollAction extends BaseAction {
  type: ActionType.SCROLL;
  coordinate: [number, number];
  direction: 'up' | 'down' | 'left' | 'right';
  amount?: number;
}

export interface DragAction extends BaseAction {
  type: ActionType.DRAG;
  startCoordinate: [number, number];
  endCoordinate: [number, number];
}

export interface WaitAction extends BaseAction {
  type: ActionType.WAIT;
  duration?: number;     // 밀리초
}

export interface FinishedAction extends BaseAction {
  type: ActionType.FINISHED;
  summary?: string;
}

export type Action =
  | ClickAction
  | TypeAction
  | KeyAction
  | HotkeyAction
  | ScrollAction
  | DragAction
  | WaitAction
  | FinishedAction;
```

### 파서 인터페이스

```typescript
// action-parser/src/parsers/base.ts

export interface ActionParser {
  /**
   * LLM 출력을 파싱하여 액션 배열 반환
   */
  parse(output: string): Action[];

  /**
   * 파서가 해당 형식을 지원하는지 확인
   */
  canParse(output: string): boolean;

  /**
   * 파서 이름
   */
  readonly name: string;
}

export abstract class BaseActionParser implements ActionParser {
  abstract readonly name: string;

  abstract parse(output: string): Action[];

  canParse(output: string): boolean {
    try {
      const actions = this.parse(output);
      return actions.length > 0;
    } catch {
      return false;
    }
  }

  protected normalizeCoordinate(
    coord: [number, number],
    screenSize: { width: number; height: number }
  ): [number, number] {
    // 정규화된 좌표(0-1)를 픽셀 좌표로 변환
    if (coord[0] <= 1 && coord[1] <= 1) {
      return [
        Math.round(coord[0] * screenSize.width),
        Math.round(coord[1] * screenSize.height)
      ];
    }
    return coord;
  }
}
```

### UI-TARS 형식 파서

```typescript
// action-parser/src/parsers/uitars.ts

/**
 * UI-TARS 모델의 출력 형식을 파싱
 *
 * 형식 예시:
 * Thought: I need to click the search button
 * Action: click(0.5, 0.3)
 */
export class UITARSParser extends BaseActionParser {
  readonly name = 'uitars';

  private readonly actionPatterns = {
    click: /click\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)/i,
    doubleClick: /double_click\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)/i,
    rightClick: /right_click\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)/i,
    type: /type\s*\(\s*["'](.+?)["']\s*\)/i,
    key: /key\s*\(\s*["'](.+?)["']\s*\)/i,
    hotkey: /hotkey\s*\(\s*["'](.+?)["']\s*\)/i,
    scroll: /scroll\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*["'](\w+)["']\s*\)/i,
    drag: /drag\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)/i,
    wait: /wait\s*\(\s*(\d+)?\s*\)/i,
    finished: /finished\s*\(\s*["']?(.*)["']?\s*\)/i
  };

  parse(output: string): Action[] {
    const actions: Action[] = [];

    // Thought 추출
    const thoughtMatch = output.match(/Thought:\s*(.+?)(?=Action:|$)/is);
    const thought = thoughtMatch?.[1]?.trim();

    // Action 추출
    const actionMatch = output.match(/Action:\s*(.+?)$/is);
    if (!actionMatch) return actions;

    const actionStr = actionMatch[1].trim();

    // 각 패턴 매칭
    for (const [actionType, pattern] of Object.entries(this.actionPatterns)) {
      const match = actionStr.match(pattern);
      if (match) {
        const action = this.createAction(actionType, match, thought);
        if (action) actions.push(action);
        break;
      }
    }

    return actions;
  }

  private createAction(
    type: string,
    match: RegExpMatchArray,
    thought?: string
  ): Action | null {
    switch (type) {
      case 'click':
        return {
          type: ActionType.CLICK,
          coordinate: [parseFloat(match[1]), parseFloat(match[2])],
          thought
        };

      case 'doubleClick':
        return {
          type: ActionType.DOUBLE_CLICK,
          coordinate: [parseFloat(match[1]), parseFloat(match[2])],
          thought
        };

      case 'type':
        return {
          type: ActionType.TYPE,
          text: match[1],
          thought
        };

      case 'key':
        return {
          type: ActionType.KEY,
          key: match[1],
          thought
        };

      case 'hotkey':
        return {
          type: ActionType.HOTKEY,
          keys: match[1].split('+').map(k => k.trim()),
          thought
        };

      case 'scroll':
        return {
          type: ActionType.SCROLL,
          coordinate: [parseFloat(match[1]), parseFloat(match[2])],
          direction: match[3] as ScrollAction['direction'],
          thought
        };

      case 'drag':
        return {
          type: ActionType.DRAG,
          startCoordinate: [parseFloat(match[1]), parseFloat(match[2])],
          endCoordinate: [parseFloat(match[3]), parseFloat(match[4])],
          thought
        };

      case 'wait':
        return {
          type: ActionType.WAIT,
          duration: match[1] ? parseInt(match[1]) : undefined,
          thought
        };

      case 'finished':
        return {
          type: ActionType.FINISHED,
          summary: match[1] || undefined,
          thought
        };

      default:
        return null;
    }
  }
}
```

### JSON 형식 파서

```typescript
// action-parser/src/parsers/json.ts

/**
 * JSON 형식의 액션을 파싱
 *
 * 형식 예시:
 * {
 *   "thought": "I need to click the search button",
 *   "action": "click",
 *   "coordinate": [0.5, 0.3]
 * }
 */
export class JSONActionParser extends BaseActionParser {
  readonly name = 'json';

  parse(output: string): Action[] {
    // JSON 블록 추출
    const jsonMatch = output.match(/```json\s*([\s\S]*?)\s*```/);
    const jsonStr = jsonMatch ? jsonMatch[1] : output;

    try {
      const parsed = JSON.parse(jsonStr);

      // 배열인 경우
      if (Array.isArray(parsed)) {
        return parsed.map(item => this.parseActionObject(item));
      }

      // 단일 객체인 경우
      return [this.parseActionObject(parsed)];
    } catch (e) {
      return [];
    }
  }

  private parseActionObject(obj: any): Action {
    const type = this.normalizeActionType(obj.action || obj.type);

    switch (type) {
      case ActionType.CLICK:
        return {
          type: ActionType.CLICK,
          coordinate: obj.coordinate || obj.coords || [obj.x, obj.y],
          thought: obj.thought,
          element: obj.element
        };

      case ActionType.TYPE:
        return {
          type: ActionType.TYPE,
          text: obj.text || obj.content,
          coordinate: obj.coordinate,
          thought: obj.thought
        };

      case ActionType.SCROLL:
        return {
          type: ActionType.SCROLL,
          coordinate: obj.coordinate || [0.5, 0.5],
          direction: obj.direction,
          amount: obj.amount,
          thought: obj.thought
        };

      // ... 다른 액션 타입들

      default:
        return {
          type: ActionType.FINISHED,
          thought: obj.thought
        };
    }
  }

  private normalizeActionType(type: string): ActionType {
    const typeMap: Record<string, ActionType> = {
      'click': ActionType.CLICK,
      'left_click': ActionType.CLICK,
      'double_click': ActionType.DOUBLE_CLICK,
      'right_click': ActionType.RIGHT_CLICK,
      'type': ActionType.TYPE,
      'input': ActionType.TYPE,
      'key': ActionType.KEY,
      'press': ActionType.KEY,
      'hotkey': ActionType.HOTKEY,
      'shortcut': ActionType.HOTKEY,
      'scroll': ActionType.SCROLL,
      'drag': ActionType.DRAG,
      'wait': ActionType.WAIT,
      'sleep': ActionType.WAIT,
      'finished': ActionType.FINISHED,
      'done': ActionType.FINISHED
    };

    return typeMap[type.toLowerCase()] || ActionType.FINISHED;
  }
}
```

### 파서 팩토리

```typescript
// action-parser/src/index.ts

import { UITARSParser } from './parsers/uitars';
import { JSONActionParser } from './parsers/json';
import { ActionParser, Action } from './types';

export class ActionParserFactory {
  private parsers: ActionParser[] = [];

  constructor() {
    // 기본 파서 등록
    this.register(new UITARSParser());
    this.register(new JSONActionParser());
  }

  register(parser: ActionParser): void {
    this.parsers.push(parser);
  }

  parse(output: string): Action[] {
    // 적합한 파서 찾기
    for (const parser of this.parsers) {
      if (parser.canParse(output)) {
        return parser.parse(output);
      }
    }

    // 기본 파서로 시도
    return this.parsers[0].parse(output);
  }

  getParser(name: string): ActionParser | undefined {
    return this.parsers.find(p => p.name === name);
  }
}

// 싱글톤 인스턴스
export const actionParser = new ActionParserFactory();
```

---

## GUIAgent 클래스

### 메인 클래스 구조

```typescript
// agent-sdk/src/GUIAgent.ts

import { EventEmitter } from 'events';
import { Action, ActionType, actionParser } from '@anthropic-ai/action-parser';
import { Operator } from './operators/types';

export interface GUIAgentOptions {
  operator: Operator;
  model: LLMClient;
  systemPrompt?: string;
  maxLoopCount?: number;
  loopIntervalInMs?: number;
  onAction?: (action: Action) => void;
  onScreenshot?: (screenshot: Buffer) => void;
}

export class GUIAgent extends EventEmitter {
  private operator: Operator;
  private model: LLMClient;
  private systemPrompt: string;
  private maxLoopCount: number;
  private loopIntervalInMs: number;
  private abortController: AbortController | null = null;

  private conversationHistory: Message[] = [];
  private actionHistory: Action[] = [];

  constructor(options: GUIAgentOptions) {
    super();

    this.operator = options.operator;
    this.model = options.model;
    this.systemPrompt = options.systemPrompt || DEFAULT_SYSTEM_PROMPT;
    this.maxLoopCount = options.maxLoopCount ?? 25;
    this.loopIntervalInMs = options.loopIntervalInMs ?? 500;
  }

  async run(
    instruction: string,
    options?: { signal?: AbortSignal }
  ): Promise<GUIAgentResult> {
    this.abortController = new AbortController();
    const signal = options?.signal || this.abortController.signal;

    // 1. Operator 초기화
    await this.operator.initialize();

    // 2. 대화 히스토리 초기화
    this.conversationHistory = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: instruction }
    ];

    this.emit('status', 'running');

    let loopCount = 0;

    // 3. 에이전트 루프
    while (loopCount < this.maxLoopCount) {
      if (signal.aborted) {
        this.emit('status', 'stopped');
        break;
      }

      loopCount++;
      this.emit('loop', { count: loopCount, max: this.maxLoopCount });

      try {
        // 스크린샷 캡처
        const screenshot = await this.operator.screenshot();
        this.emit('screenshot', screenshot);

        // LLM 호출
        const response = await this.callModel(screenshot);
        this.emit('response', response);

        // 액션 파싱
        const actions = actionParser.parse(response.content);
        if (actions.length === 0) {
          console.warn('No action parsed from response');
          continue;
        }

        const action = actions[0];
        this.actionHistory.push(action);
        this.emit('action', action);

        // 종료 조건 확인
        if (action.type === ActionType.FINISHED) {
          this.emit('status', 'completed');
          break;
        }

        // 액션 실행
        await this.executeAction(action);

        // 대기
        await this.delay(this.loopIntervalInMs);

      } catch (error) {
        this.emit('error', error);

        if (!this.isRecoverableError(error)) {
          throw error;
        }
      }
    }

    // 4. 정리
    await this.operator.cleanup();

    return {
      success: this.actionHistory.some(a => a.type === ActionType.FINISHED),
      loopCount,
      actions: this.actionHistory,
      conversationHistory: this.conversationHistory
    };
  }

  stop(): void {
    this.abortController?.abort();
  }

  private async callModel(screenshot: Buffer): Promise<ModelResponse> {
    // 스크린샷을 메시지에 추가
    const imageMessage: Message = {
      role: 'user',
      content: [
        {
          type: 'image',
          source: {
            type: 'base64',
            media_type: 'image/png',
            data: screenshot.toString('base64')
          }
        },
        {
          type: 'text',
          text: 'What should I do next based on this screenshot?'
        }
      ]
    };

    this.conversationHistory.push(imageMessage);

    // LLM 호출
    const response = await this.model.chat({
      messages: this.conversationHistory
    });

    // 응답을 히스토리에 추가
    this.conversationHistory.push({
      role: 'assistant',
      content: response.content
    });

    return response;
  }

  private async executeAction(action: Action): Promise<void> {
    switch (action.type) {
      case ActionType.CLICK:
        await this.operator.click(action.coordinate);
        break;

      case ActionType.DOUBLE_CLICK:
        await this.operator.doubleClick(action.coordinate);
        break;

      case ActionType.RIGHT_CLICK:
        await this.operator.rightClick(action.coordinate);
        break;

      case ActionType.TYPE:
        if (action.coordinate) {
          await this.operator.click(action.coordinate);
        }
        await this.operator.type(action.text);
        break;

      case ActionType.KEY:
        await this.operator.key(action.key);
        break;

      case ActionType.HOTKEY:
        await this.operator.hotkey(action.keys);
        break;

      case ActionType.SCROLL:
        await this.operator.scroll(
          action.coordinate,
          action.direction,
          action.amount
        );
        break;

      case ActionType.DRAG:
        await this.operator.drag(
          action.startCoordinate,
          action.endCoordinate
        );
        break;

      case ActionType.WAIT:
        await this.delay(action.duration || 1000);
        break;
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private isRecoverableError(error: any): boolean {
    // 네트워크 에러, 타임아웃 등은 복구 가능
    return error.code === 'ETIMEDOUT' ||
           error.code === 'ECONNRESET' ||
           error.message?.includes('rate limit');
  }
}
```

---

## 시스템 프롬프트

### 기본 시스템 프롬프트

```typescript
// agent-sdk/src/prompts/default.ts

export const DEFAULT_SYSTEM_PROMPT = `You are a GUI automation agent. Your task is to help users complete tasks by interacting with graphical user interfaces.

## Input
You will receive:
1. A screenshot of the current screen
2. The user's instruction or the previous action result

## Output Format
Respond with:
1. Thought: Your reasoning about what to do next
2. Action: The action to perform

## Available Actions

### Mouse Actions
- click(x, y): Click at coordinates (normalized 0-1)
- double_click(x, y): Double click at coordinates
- right_click(x, y): Right click at coordinates
- drag(x1, y1, x2, y2): Drag from (x1,y1) to (x2,y2)
- scroll(x, y, direction): Scroll at coordinates (direction: up/down/left/right)

### Keyboard Actions
- type("text"): Type text
- key("key_name"): Press a single key (Enter, Escape, Tab, etc.)
- hotkey("key1+key2"): Press key combination (ctrl+c, cmd+v, etc.)

### Control Actions
- wait(ms): Wait for specified milliseconds
- finished("summary"): Task completed with summary

## Guidelines
1. Always analyze the screenshot carefully before acting
2. Use precise coordinates based on visible UI elements
3. Prefer clicking on buttons, links, or input fields
4. Type text only after clicking on the appropriate input field
5. Use finished() when the task is complete

## Example
Thought: I see a search input field at the top of the page. I need to click on it and type the search query.
Action: click(0.5, 0.1)
`;
```

### 언어별 프롬프트

```typescript
// agent-sdk/src/prompts/index.ts

import { DEFAULT_SYSTEM_PROMPT } from './default';
import { CHINESE_SYSTEM_PROMPT } from './chinese';

export function getSystemPrompt(language: string): string {
  switch (language) {
    case 'zh':
    case 'chinese':
      return CHINESE_SYSTEM_PROMPT;

    case 'en':
    case 'english':
    default:
      return DEFAULT_SYSTEM_PROMPT;
  }
}
```

---

## SoM (Set-of-Mark) 시각화

### SoM 생성기

```typescript
// agent-sdk/src/som/SoMGenerator.ts

export interface SoMElement {
  id: number;
  bbox: [number, number, number, number]; // x, y, width, height
  label: string;
  type: 'button' | 'input' | 'link' | 'text' | 'image' | 'other';
}

export class SoMGenerator {
  /**
   * 스크린샷에 Set-of-Mark 오버레이 생성
   */
  async generate(screenshot: Buffer): Promise<{
    annotatedImage: Buffer;
    elements: SoMElement[];
  }> {
    // 1. 요소 감지 (Vision 모델 또는 DOM 분석)
    const elements = await this.detectElements(screenshot);

    // 2. 이미지에 마커 추가
    const annotatedImage = await this.drawMarkers(screenshot, elements);

    return { annotatedImage, elements };
  }

  private async detectElements(screenshot: Buffer): Promise<SoMElement[]> {
    // Vision 모델을 사용하여 UI 요소 감지
    const response = await this.visionModel.analyze({
      image: screenshot,
      prompt: `Detect all interactive UI elements in this screenshot.
               Return JSON array with: id, bbox [x,y,w,h], label, type`
    });

    return JSON.parse(response);
  }

  private async drawMarkers(
    screenshot: Buffer,
    elements: SoMElement[]
  ): Promise<Buffer> {
    const { createCanvas, loadImage } = await import('canvas');

    const image = await loadImage(screenshot);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');

    // 원본 이미지 그리기
    ctx.drawImage(image, 0, 0);

    // 각 요소에 마커 추가
    for (const element of elements) {
      const [x, y, w, h] = element.bbox;

      // 바운딩 박스
      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // ID 라벨
      ctx.fillStyle = '#FF0000';
      ctx.font = 'bold 14px Arial';
      ctx.fillText(`[${element.id}]`, x, y - 5);
    }

    return canvas.toBuffer('image/png');
  }
}
```

---

## 결과 타입

```typescript
// agent-sdk/src/types.ts

export interface GUIAgentResult {
  success: boolean;
  loopCount: number;
  actions: Action[];
  conversationHistory: Message[];
  error?: Error;
}

export interface GUIAgentData {
  status: 'idle' | 'running' | 'completed' | 'stopped' | 'error';
  currentLoop: number;
  maxLoop: number;
  lastScreenshot?: string;
  lastAction?: Action;
  errorMessage?: string;
}
```

---

*다음 글에서는 Operators (Browser, NutJS, ADB)를 분석합니다.*
