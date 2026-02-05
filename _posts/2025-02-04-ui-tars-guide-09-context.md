---
layout: post
title: "UI-TARS 완벽 가이드 (9) - Context Engineering"
date: 2025-02-04
permalink: /ui-tars-guide-09-context/
author: ByteDance
categories: [AI 에이전트, UI-TARS]
tags: [UI-TARS, Context Engineering, Memory, Prompt, Token Management]
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "UI-TARS의 컨텍스트 엔지니어링을 분석합니다. 컨텍스트 관리, 메모리 시스템, 프롬프트 최적화를 살펴봅니다."
---

## Context Engineering 개요

**Context Engineering**은 LLM에 전달되는 컨텍스트를 최적화하여 에이전트의 성능을 향상시키는 핵심 모듈입니다.

```
multimodal/tarko/context-engineer/
├── src/
│   ├── ContextEngineer.ts     # 메인 클래스
│   ├── strategies/            # 컨텍스트 전략
│   │   ├── SlidingWindow.ts   # 슬라이딩 윈도우
│   │   ├── Summarization.ts   # 요약 기반
│   │   └── Hybrid.ts          # 하이브리드
│   ├── memory/                # 메모리 시스템
│   │   ├── ShortTermMemory.ts
│   │   ├── LongTermMemory.ts
│   │   └── WorkingMemory.ts
│   └── tokenizer/             # 토큰 관리
│       └── TokenCounter.ts
```

---

## Context Engineer 클래스

### 메인 인터페이스

```typescript
// context-engineer/src/ContextEngineer.ts

export interface ContextEngineerOptions {
  maxTokens: number;
  strategy: 'sliding-window' | 'summarization' | 'hybrid';
  reservedTokens?: {
    system?: number;
    tools?: number;
    response?: number;
  };
  model?: string;
}

export interface ProcessedContext {
  messages: Message[];
  tokenCount: number;
  truncated: boolean;
  summary?: string;
}

export class ContextEngineer {
  private options: ContextEngineerOptions;
  private strategy: ContextStrategy;
  private tokenCounter: TokenCounter;
  private memory: MemorySystem;

  constructor(options: ContextEngineerOptions) {
    this.options = {
      reservedTokens: {
        system: 2000,
        tools: 3000,
        response: 4000
      },
      ...options
    };

    this.tokenCounter = new TokenCounter(options.model);
    this.strategy = this.createStrategy(options.strategy);
    this.memory = new MemorySystem();
  }

  private createStrategy(type: string): ContextStrategy {
    switch (type) {
      case 'sliding-window':
        return new SlidingWindowStrategy(this.tokenCounter);
      case 'summarization':
        return new SummarizationStrategy(this.tokenCounter);
      case 'hybrid':
      default:
        return new HybridStrategy(this.tokenCounter);
    }
  }

  /**
   * 컨텍스트 처리
   */
  async process(
    messages: Message[],
    options?: { tools?: Tool[]; systemPrompt?: string }
  ): Promise<ProcessedContext> {
    // 사용 가능한 토큰 계산
    const availableTokens = this.calculateAvailableTokens(options);

    // 현재 토큰 수 계산
    const currentTokens = this.tokenCounter.countMessages(messages);

    // 토큰이 충분하면 그대로 반환
    if (currentTokens <= availableTokens) {
      return {
        messages,
        tokenCount: currentTokens,
        truncated: false
      };
    }

    // 전략에 따라 컨텍스트 압축
    const processed = await this.strategy.compress(messages, availableTokens);

    // 메모리에 저장
    await this.memory.store(messages);

    return {
      ...processed,
      truncated: true
    };
  }

  private calculateAvailableTokens(
    options?: { tools?: Tool[]; systemPrompt?: string }
  ): number {
    let available = this.options.maxTokens;

    // 예약 토큰 차감
    available -= this.options.reservedTokens!.response!;

    if (options?.systemPrompt) {
      const systemTokens = this.tokenCounter.count(options.systemPrompt);
      available -= systemTokens;
    } else {
      available -= this.options.reservedTokens!.system!;
    }

    if (options?.tools) {
      const toolsTokens = this.tokenCounter.countTools(options.tools);
      available -= toolsTokens;
    } else {
      available -= this.options.reservedTokens!.tools!;
    }

    return Math.max(available, 1000); // 최소 1000 토큰 보장
  }

  /**
   * 메모리에서 관련 컨텍스트 검색
   */
  async retrieveRelevant(query: string, k: number = 5): Promise<Message[]> {
    return this.memory.search(query, k);
  }

  /**
   * 컨텍스트 요약 생성
   */
  async summarize(messages: Message[]): Promise<string> {
    return this.strategy.summarize(messages);
  }
}
```

---

## 컨텍스트 전략

### 슬라이딩 윈도우 전략

```typescript
// context-engineer/src/strategies/SlidingWindow.ts

export class SlidingWindowStrategy implements ContextStrategy {
  private tokenCounter: TokenCounter;

  constructor(tokenCounter: TokenCounter) {
    this.tokenCounter = tokenCounter;
  }

  async compress(
    messages: Message[],
    maxTokens: number
  ): Promise<ProcessedContext> {
    // 시스템 메시지는 항상 유지
    const systemMessages = messages.filter(m => m.role === 'system');
    const otherMessages = messages.filter(m => m.role !== 'system');

    let systemTokens = this.tokenCounter.countMessages(systemMessages);
    let availableForOthers = maxTokens - systemTokens;

    // 최신 메시지부터 역순으로 추가
    const selectedMessages: Message[] = [];
    let currentTokens = 0;

    for (let i = otherMessages.length - 1; i >= 0; i--) {
      const message = otherMessages[i];
      const messageTokens = this.tokenCounter.countMessage(message);

      if (currentTokens + messageTokens <= availableForOthers) {
        selectedMessages.unshift(message);
        currentTokens += messageTokens;
      } else {
        break;
      }
    }

    return {
      messages: [...systemMessages, ...selectedMessages],
      tokenCount: systemTokens + currentTokens,
      truncated: selectedMessages.length < otherMessages.length
    };
  }

  async summarize(messages: Message[]): Promise<string> {
    // 간단한 요약: 마지막 몇 개 메시지의 내용
    const recentMessages = messages.slice(-5);
    return recentMessages
      .map(m => `${m.role}: ${this.truncateContent(m.content)}`)
      .join('\n');
  }

  private truncateContent(content: string | any[]): string {
    if (typeof content === 'string') {
      return content.slice(0, 100) + (content.length > 100 ? '...' : '');
    }
    return '[multimodal content]';
  }
}
```

### 요약 기반 전략

```typescript
// context-engineer/src/strategies/Summarization.ts

export class SummarizationStrategy implements ContextStrategy {
  private tokenCounter: TokenCounter;
  private summarizer: LLMClient;

  constructor(tokenCounter: TokenCounter, summarizer?: LLMClient) {
    this.tokenCounter = tokenCounter;
    this.summarizer = summarizer || this.createDefaultSummarizer();
  }

  private createDefaultSummarizer(): LLMClient {
    return new LLMClient({
      provider: 'openai',
      model: 'gpt-4o-mini' // 빠르고 저렴한 모델 사용
    });
  }

  async compress(
    messages: Message[],
    maxTokens: number
  ): Promise<ProcessedContext> {
    // 시스템 메시지 유지
    const systemMessages = messages.filter(m => m.role === 'system');
    const otherMessages = messages.filter(m => m.role !== 'system');

    const systemTokens = this.tokenCounter.countMessages(systemMessages);
    const availableTokens = maxTokens - systemTokens;

    // 오래된 메시지들을 요약
    const oldMessages = otherMessages.slice(0, -10); // 최근 10개 제외
    const recentMessages = otherMessages.slice(-10);

    let summary = '';
    if (oldMessages.length > 0) {
      summary = await this.summarize(oldMessages);
    }

    // 요약 메시지 생성
    const summaryMessage: Message = {
      role: 'system',
      content: `Previous conversation summary:\n${summary}`
    };

    const summaryTokens = this.tokenCounter.countMessage(summaryMessage);

    // 최근 메시지를 토큰 제한 내에서 선택
    const selectedRecent: Message[] = [];
    let recentTokens = 0;
    const availableForRecent = availableTokens - summaryTokens;

    for (let i = recentMessages.length - 1; i >= 0; i--) {
      const msg = recentMessages[i];
      const msgTokens = this.tokenCounter.countMessage(msg);

      if (recentTokens + msgTokens <= availableForRecent) {
        selectedRecent.unshift(msg);
        recentTokens += msgTokens;
      } else {
        break;
      }
    }

    return {
      messages: [...systemMessages, summaryMessage, ...selectedRecent],
      tokenCount: systemTokens + summaryTokens + recentTokens,
      truncated: true,
      summary
    };
  }

  async summarize(messages: Message[]): Promise<string> {
    const conversationText = messages
      .map(m => {
        const content = typeof m.content === 'string'
          ? m.content
          : '[multimodal]';
        return `${m.role}: ${content}`;
      })
      .join('\n');

    const response = await this.summarizer.chat({
      messages: [
        {
          role: 'system',
          content: 'Summarize the following conversation concisely, preserving key information and context. Focus on: decisions made, information gathered, and current task state.'
        },
        {
          role: 'user',
          content: conversationText
        }
      ]
    });

    return response.content;
  }
}
```

### 하이브리드 전략

```typescript
// context-engineer/src/strategies/Hybrid.ts

export class HybridStrategy implements ContextStrategy {
  private slidingWindow: SlidingWindowStrategy;
  private summarization: SummarizationStrategy;
  private tokenCounter: TokenCounter;

  constructor(tokenCounter: TokenCounter) {
    this.tokenCounter = tokenCounter;
    this.slidingWindow = new SlidingWindowStrategy(tokenCounter);
    this.summarization = new SummarizationStrategy(tokenCounter);
  }

  async compress(
    messages: Message[],
    maxTokens: number
  ): Promise<ProcessedContext> {
    const currentTokens = this.tokenCounter.countMessages(messages);

    // 토큰이 약간 초과하면 슬라이딩 윈도우 사용
    if (currentTokens < maxTokens * 1.5) {
      return this.slidingWindow.compress(messages, maxTokens);
    }

    // 많이 초과하면 요약 전략 사용
    return this.summarization.compress(messages, maxTokens);
  }

  async summarize(messages: Message[]): Promise<string> {
    return this.summarization.summarize(messages);
  }
}
```

---

## 메모리 시스템

### 단기 메모리

```typescript
// context-engineer/src/memory/ShortTermMemory.ts

export class ShortTermMemory {
  private buffer: Message[] = [];
  private maxSize: number;

  constructor(maxSize: number = 100) {
    this.maxSize = maxSize;
  }

  add(message: Message): void {
    this.buffer.push(message);

    // 크기 초과 시 오래된 메시지 제거
    if (this.buffer.length > this.maxSize) {
      this.buffer.shift();
    }
  }

  addAll(messages: Message[]): void {
    messages.forEach(m => this.add(m));
  }

  getRecent(n: number): Message[] {
    return this.buffer.slice(-n);
  }

  getAll(): Message[] {
    return [...this.buffer];
  }

  clear(): void {
    this.buffer = [];
  }

  get size(): number {
    return this.buffer.length;
  }
}
```

### 장기 메모리 (벡터 저장소)

```typescript
// context-engineer/src/memory/LongTermMemory.ts

export interface MemoryEntry {
  id: string;
  content: string;
  embedding: number[];
  metadata: {
    timestamp: number;
    role: string;
    importance: number;
  };
}

export class LongTermMemory {
  private entries: MemoryEntry[] = [];
  private embedder: EmbeddingModel;

  constructor(embedder?: EmbeddingModel) {
    this.embedder = embedder || new OpenAIEmbeddings();
  }

  async store(message: Message): Promise<void> {
    const content = this.extractContent(message);
    const embedding = await this.embedder.embed(content);

    const entry: MemoryEntry = {
      id: `mem_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      content,
      embedding,
      metadata: {
        timestamp: Date.now(),
        role: message.role,
        importance: this.calculateImportance(message)
      }
    };

    this.entries.push(entry);
  }

  async search(query: string, k: number = 5): Promise<MemoryEntry[]> {
    const queryEmbedding = await this.embedder.embed(query);

    // 코사인 유사도 계산
    const withScores = this.entries.map(entry => ({
      entry,
      score: this.cosineSimilarity(queryEmbedding, entry.embedding)
    }));

    // 점수 순 정렬
    withScores.sort((a, b) => b.score - a.score);

    return withScores.slice(0, k).map(ws => ws.entry);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private extractContent(message: Message): string {
    if (typeof message.content === 'string') {
      return message.content;
    }

    // 멀티모달 메시지에서 텍스트 추출
    return message.content
      .filter(part => part.type === 'text')
      .map(part => (part as any).text)
      .join('\n');
  }

  private calculateImportance(message: Message): number {
    // 중요도 계산 로직
    let importance = 0.5;

    // 도구 호출 결과는 중요
    if (message.role === 'tool') {
      importance += 0.2;
    }

    // 긴 메시지는 더 중요할 수 있음
    const content = this.extractContent(message);
    if (content.length > 500) {
      importance += 0.1;
    }

    // 특정 키워드 포함 시 중요도 증가
    const importantKeywords = ['error', 'success', 'completed', 'failed'];
    for (const keyword of importantKeywords) {
      if (content.toLowerCase().includes(keyword)) {
        importance += 0.1;
        break;
      }
    }

    return Math.min(importance, 1.0);
  }

  /**
   * 중요도가 낮은 오래된 메모리 정리
   */
  prune(maxEntries: number = 1000): void {
    if (this.entries.length <= maxEntries) return;

    // 중요도와 시간 기반 점수 계산
    const now = Date.now();
    const scored = this.entries.map(entry => ({
      entry,
      score: entry.metadata.importance -
             (now - entry.metadata.timestamp) / (1000 * 60 * 60 * 24) * 0.01
    }));

    // 점수 순 정렬 후 상위 유지
    scored.sort((a, b) => b.score - a.score);
    this.entries = scored.slice(0, maxEntries).map(s => s.entry);
  }
}
```

### 작업 메모리

```typescript
// context-engineer/src/memory/WorkingMemory.ts

export interface WorkingMemoryState {
  currentTask: string | null;
  subTasks: string[];
  completedSteps: string[];
  relevantContext: Message[];
  variables: Record<string, any>;
}

export class WorkingMemory {
  private state: WorkingMemoryState = {
    currentTask: null,
    subTasks: [],
    completedSteps: [],
    relevantContext: [],
    variables: {}
  };

  setTask(task: string): void {
    this.state.currentTask = task;
    this.state.subTasks = [];
    this.state.completedSteps = [];
  }

  addSubTask(subTask: string): void {
    this.state.subTasks.push(subTask);
  }

  completeStep(step: string): void {
    this.state.completedSteps.push(step);
    // subTasks에서 제거
    const index = this.state.subTasks.indexOf(step);
    if (index > -1) {
      this.state.subTasks.splice(index, 1);
    }
  }

  setVariable(key: string, value: any): void {
    this.state.variables[key] = value;
  }

  getVariable(key: string): any {
    return this.state.variables[key];
  }

  addRelevantContext(message: Message): void {
    this.state.relevantContext.push(message);
    // 최대 10개 유지
    if (this.state.relevantContext.length > 10) {
      this.state.relevantContext.shift();
    }
  }

  getState(): WorkingMemoryState {
    return { ...this.state };
  }

  /**
   * 작업 메모리를 시스템 프롬프트로 변환
   */
  toSystemPrompt(): string {
    const parts: string[] = [];

    if (this.state.currentTask) {
      parts.push(`Current Task: ${this.state.currentTask}`);
    }

    if (this.state.subTasks.length > 0) {
      parts.push(`Remaining Sub-tasks:\n${this.state.subTasks.map(t => `- ${t}`).join('\n')}`);
    }

    if (this.state.completedSteps.length > 0) {
      parts.push(`Completed Steps:\n${this.state.completedSteps.map(s => `- ${s}`).join('\n')}`);
    }

    if (Object.keys(this.state.variables).length > 0) {
      const vars = Object.entries(this.state.variables)
        .map(([k, v]) => `- ${k}: ${JSON.stringify(v)}`)
        .join('\n');
      parts.push(`Variables:\n${vars}`);
    }

    return parts.join('\n\n');
  }

  clear(): void {
    this.state = {
      currentTask: null,
      subTasks: [],
      completedSteps: [],
      relevantContext: [],
      variables: {}
    };
  }
}
```

---

## 토큰 카운터

```typescript
// context-engineer/src/tokenizer/TokenCounter.ts

import { encoding_for_model, TiktokenModel } from 'tiktoken';

export class TokenCounter {
  private encoding: any;
  private model: string;

  constructor(model: string = 'gpt-4') {
    this.model = model;
    this.encoding = this.createEncoding(model);
  }

  private createEncoding(model: string): any {
    // 모델에 맞는 인코딩 선택
    const modelMap: Record<string, TiktokenModel> = {
      'gpt-4': 'gpt-4',
      'gpt-4o': 'gpt-4o',
      'gpt-4o-mini': 'gpt-4o-mini',
      'gpt-3.5-turbo': 'gpt-3.5-turbo',
      'claude-3': 'cl100k_base' as any
    };

    const tiktokenModel = modelMap[model] || 'gpt-4';

    try {
      return encoding_for_model(tiktokenModel);
    } catch {
      // 폴백: cl100k_base 사용
      return encoding_for_model('gpt-4');
    }
  }

  /**
   * 문자열 토큰 수 계산
   */
  count(text: string): number {
    return this.encoding.encode(text).length;
  }

  /**
   * 메시지 토큰 수 계산
   */
  countMessage(message: Message): number {
    let tokens = 4; // 메시지 구조 오버헤드

    // role
    tokens += this.count(message.role);

    // content
    if (typeof message.content === 'string') {
      tokens += this.count(message.content);
    } else {
      // 멀티모달 콘텐츠
      for (const part of message.content) {
        if (part.type === 'text') {
          tokens += this.count((part as any).text);
        } else if (part.type === 'image') {
          // 이미지는 고정 토큰으로 추정
          tokens += 765; // GPT-4V 기준
        }
      }
    }

    return tokens;
  }

  /**
   * 메시지 배열 토큰 수 계산
   */
  countMessages(messages: Message[]): number {
    let tokens = 3; // 대화 시작 오버헤드

    for (const message of messages) {
      tokens += this.countMessage(message);
    }

    return tokens;
  }

  /**
   * 도구 스키마 토큰 수 계산
   */
  countTools(tools: Tool[]): number {
    let tokens = 0;

    for (const tool of tools) {
      tokens += this.count(tool.name);
      tokens += this.count(tool.description || '');
      tokens += this.count(JSON.stringify(tool.parameters || {}));
      tokens += 10; // 구조 오버헤드
    }

    return tokens;
  }

  /**
   * 텍스트를 토큰 제한에 맞게 자르기
   */
  truncate(text: string, maxTokens: number): string {
    const tokens = this.encoding.encode(text);

    if (tokens.length <= maxTokens) {
      return text;
    }

    const truncated = tokens.slice(0, maxTokens);
    return this.encoding.decode(truncated);
  }
}
```

---

## 프롬프트 최적화

### 동적 프롬프트 생성

```typescript
// context-engineer/src/PromptOptimizer.ts

export class PromptOptimizer {
  private templates: Map<string, string> = new Map();

  registerTemplate(name: string, template: string): void {
    this.templates.set(name, template);
  }

  /**
   * 컨텍스트에 맞게 프롬프트 최적화
   */
  optimize(
    templateName: string,
    variables: Record<string, any>,
    options: {
      maxTokens?: number;
      tokenCounter?: TokenCounter;
    } = {}
  ): string {
    const template = this.templates.get(templateName);
    if (!template) {
      throw new Error(`Template not found: ${templateName}`);
    }

    // 변수 대체
    let prompt = template;
    for (const [key, value] of Object.entries(variables)) {
      const placeholder = `{{${key}}}`;
      prompt = prompt.replace(
        new RegExp(placeholder, 'g'),
        String(value)
      );
    }

    // 토큰 제한 적용
    if (options.maxTokens && options.tokenCounter) {
      prompt = options.tokenCounter.truncate(prompt, options.maxTokens);
    }

    return prompt;
  }

  /**
   * 조건부 섹션 처리
   */
  processConditionals(
    prompt: string,
    conditions: Record<string, boolean>
  ): string {
    // {{#if condition}}...{{/if}} 처리
    for (const [condition, value] of Object.entries(conditions)) {
      const pattern = new RegExp(
        `{{#if ${condition}}}([\\s\\S]*?){{/if}}`,
        'g'
      );

      prompt = prompt.replace(pattern, (_, content) => {
        return value ? content : '';
      });
    }

    return prompt;
  }
}
```

---

*다음 글에서는 활용 가이드와 결론을 다룹니다.*
