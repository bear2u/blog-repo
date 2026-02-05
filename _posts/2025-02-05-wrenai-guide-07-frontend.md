---
layout: post
title: "WrenAI 완벽 가이드 (7) - 프론트엔드 구조"
date: 2025-02-05
permalink: /wrenai-guide-07-frontend/
author: Canner
categories: [AI 에이전트, WrenAI]
tags: [WrenAI, Next.js, GraphQL, Apollo, React]
original_url: "https://github.com/Canner/WrenAI"
excerpt: "WrenAI 프론트엔드의 Next.js, Apollo GraphQL 구조를 분석합니다."
---

## 프론트엔드 기술 스택

```
┌─────────────────────────────────────────────────────────────┐
│                    Wren UI 기술 스택                         │
├─────────────────────────────────────────────────────────────┤
│  🎨 프레임워크: Next.js 14.2 (Pages Router)                 │
│  📘 언어: TypeScript 5.2                                    │
│  🔗 API: Apollo Server/Client (GraphQL)                    │
│  🎯 UI: Ant Design 4.20                                    │
│  📊 차트: Vega-Lite 6.2                                    │
│  📈 다이어그램: React Flow 11.10                           │
│  💾 DB: Knex + SQLite/PostgreSQL                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 디렉토리 구조

```
wren-ui/
├── src/
│   ├── pages/                    # Next.js Pages Router
│   │   ├── _app.tsx             # 앱 래퍼
│   │   ├── _document.tsx        # HTML 문서
│   │   ├── index.tsx            # 홈 (/)
│   │   ├── modeling/            # 모델링 페이지
│   │   │   └── index.tsx
│   │   ├── setup/               # 설정 마법사
│   │   │   └── index.tsx
│   │   ├── knowledge/           # 지식 관리
│   │   ├── dashboard/           # 대시보드
│   │   └── api/                 # API 라우트
│   │       └── graphql.ts       # GraphQL 엔드포인트
│   │
│   ├── apollo/
│   │   ├── server/              # GraphQL 서버
│   │   │   ├── schema.ts        # 스키마 정의
│   │   │   ├── resolvers/       # 리졸버
│   │   │   │   ├── askingResolver.ts
│   │   │   │   ├── modelingResolver.ts
│   │   │   │   └── projectResolver.ts
│   │   │   ├── services/        # 비즈니스 로직
│   │   │   │   ├── askingService.ts
│   │   │   │   ├── modelService.ts
│   │   │   │   └── projectService.ts
│   │   │   └── repositories/    # 데이터 접근
│   │   │       ├── projectRepository.ts
│   │   │       ├── modelRepository.ts
│   │   │       └── relationshipRepository.ts
│   │   │
│   │   └── client/              # GraphQL 클라이언트
│   │       ├── graphql/         # 쿼리/뮤테이션 정의
│   │       │   ├── asking.ts
│   │       │   ├── model.ts
│   │       │   └── project.ts
│   │       └── apollo-client.ts
│   │
│   ├── components/              # React 컴포넌트
│   │   ├── sidebar/
│   │   ├── chart/
│   │   ├── diagram/
│   │   ├── table/
│   │   └── learning/
│   │
│   ├── hooks/                   # Custom Hooks
│   │   ├── useAsk.ts
│   │   ├── useModel.ts
│   │   └── useProject.ts
│   │
│   ├── utils/                   # 유틸리티
│   └── styles/                  # 스타일시트 (Less)
│
├── migrations/                  # DB 마이그레이션
├── public/                      # 정적 파일
├── e2e/                        # E2E 테스트 (Playwright)
└── package.json
```

---

## GraphQL API 구조

### 스키마 정의

```graphql
# schema.graphql

type Query {
  # 프로젝트
  projects: [Project!]!
  project(id: Int!): Project

  # 모델
  models(projectId: Int!): [Model!]!
  model(id: Int!): Model

  # 관계
  relations(projectId: Int!): [Relation!]!

  # 대시보드
  dashboards(projectId: Int!): [Dashboard!]!
  dashboard(id: Int!): Dashboard

  # 히스토리
  apiHistories(filter: ApiHistoryFilterInput): ApiHistoryConnection!
}

type Mutation {
  # 프로젝트
  createProject(input: CreateProjectInput!): Project!
  updateProject(id: Int!, input: UpdateProjectInput!): Project!
  deployProject(projectId: Int!): DeployResult!

  # 모델
  submitModelData(projectId: Int!, models: [ModelInput!]!): SubmitResult!
  updateModel(id: Int!, input: UpdateModelInput!): Model!

  # 관계
  createRelation(input: CreateRelationInput!): Relation!
  updateRelation(id: Int!, input: UpdateRelationInput!): Relation!
  deleteRelation(id: Int!): Boolean!

  # Ask
  ask(projectId: Int!, question: String!): AskResult!
  askFollowUp(threadId: String!, question: String!): AskResult!
}

type Subscription {
  askResultUpdated(threadId: String!): AskResult!
}
```

### 주요 타입

```typescript
// types.ts

interface Project {
  id: number;
  name: string;
  dataSource: DataSource;
  onboardingStatus: OnboardingStatus;
  language: ProjectLanguage;
  createdAt: string;
  updatedAt: string;
}

interface Model {
  id: number;
  name: string;
  projectId: number;
  columns: ModelColumn[];
  primaryKey?: string;
  properties: Record<string, any>;
}

interface ModelColumn {
  id: number;
  name: string;
  type: string;
  isCalculated: boolean;
  isHidden: boolean;
  expression?: string;
  properties: Record<string, any>;
}

interface Relation {
  id: number;
  fromModelId: number;
  toModelId: number;
  fromColumnId: number;
  toColumnId: number;
  type: RelationType;  // ONE_TO_ONE | ONE_TO_MANY | MANY_TO_ONE
}

interface AskResult {
  threadId: string;
  queryId: string;
  status: AskStatus;
  sql?: string;
  reasoning?: string;
  error?: string;
}
```

---

## 리졸버 패턴

```typescript
// resolvers/askingResolver.ts

export const askingResolver = {
  Query: {
    async askingTask(_, { threadId }, { dataSources }) {
      return dataSources.askingService.getTask(threadId);
    },
  },

  Mutation: {
    async ask(_, { projectId, question }, { dataSources }) {
      // 1. 태스크 생성
      const task = await dataSources.askingService.createTask({
        projectId,
        question,
      });

      // 2. AI Service 호출
      const result = await dataSources.aiService.ask({
        question,
        projectId,
      });

      // 3. 결과 저장
      await dataSources.askingService.updateTask(task.id, result);

      return {
        threadId: task.threadId,
        queryId: result.queryId,
        status: result.status,
      };
    },

    async askFollowUp(_, { threadId, question }, { dataSources }) {
      const prevTask = await dataSources.askingService.getTask(threadId);

      const result = await dataSources.aiService.askFollowUp({
        question,
        threadId,
        history: prevTask.history,
      });

      return result;
    },
  },
};
```

---

## 서비스 레이어

```typescript
// services/askingService.ts

export class AskingService {
  constructor(
    private repository: AskingRepository,
    private aiClient: AIServiceClient,
    private engineClient: WrenEngineClient
  ) {}

  async ask(projectId: number, question: string): Promise<AskResult> {
    // 1. AI Service에 질문 전송
    const response = await this.aiClient.post('/v1/asks', {
      query: question,
      project_id: projectId,
    });

    // 2. 폴링으로 결과 대기
    const result = await this.pollResult(response.query_id);

    // 3. SQL 검증
    if (result.sql) {
      const validation = await this.engineClient.validate(result.sql);
      if (!validation.valid) {
        // 수정 요청
        return this.correctSql(result.sql, validation.error);
      }
    }

    // 4. 히스토리 저장
    await this.repository.saveHistory({
      projectId,
      question,
      sql: result.sql,
      status: result.status,
    });

    return result;
  }

  private async pollResult(queryId: string): Promise<AskResult> {
    const maxAttempts = 60;
    const interval = 1000;

    for (let i = 0; i < maxAttempts; i++) {
      const result = await this.aiClient.get(`/v1/asks/${queryId}/result`);

      if (result.status === 'finished' || result.status === 'failed') {
        return result;
      }

      await this.delay(interval);
    }

    throw new Error('Timeout waiting for result');
  }
}
```

---

## 리포지토리 레이어

```typescript
// repositories/modelRepository.ts

export class ModelRepository {
  constructor(private knex: Knex) {}

  async findById(id: number): Promise<Model | null> {
    const model = await this.knex('models')
      .where({ id })
      .first();

    if (!model) return null;

    const columns = await this.knex('model_columns')
      .where({ model_id: id });

    return {
      ...model,
      columns,
    };
  }

  async findByProjectId(projectId: number): Promise<Model[]> {
    const models = await this.knex('models')
      .where({ project_id: projectId });

    const modelIds = models.map(m => m.id);

    const columns = await this.knex('model_columns')
      .whereIn('model_id', modelIds);

    return models.map(model => ({
      ...model,
      columns: columns.filter(c => c.model_id === model.id),
    }));
  }

  async create(data: CreateModelInput): Promise<Model> {
    const [id] = await this.knex('models')
      .insert({
        project_id: data.projectId,
        name: data.name,
        primary_key: data.primaryKey,
        properties: JSON.stringify(data.properties),
      });

    if (data.columns) {
      await this.knex('model_columns').insert(
        data.columns.map(col => ({
          model_id: id,
          ...col,
        }))
      );
    }

    return this.findById(id);
  }
}
```

---

## 클라이언트 사용

### Hook 패턴

```typescript
// hooks/useAsk.ts

export function useAsk() {
  const [askMutation, { loading, error }] = useMutation(ASK_MUTATION);
  const [result, setResult] = useState<AskResult | null>(null);

  const ask = useCallback(async (projectId: number, question: string) => {
    const response = await askMutation({
      variables: { projectId, question },
    });

    setResult(response.data.ask);
    return response.data.ask;
  }, [askMutation]);

  return {
    ask,
    loading,
    error,
    result,
  };
}

// 사용 예시
function AskPage() {
  const { ask, loading, result } = useAsk();

  const handleSubmit = async (question: string) => {
    await ask(1, question);
  };

  return (
    <div>
      <Input.Search
        placeholder="질문을 입력하세요"
        onSearch={handleSubmit}
        loading={loading}
      />
      {result && <SQLResult sql={result.sql} />}
    </div>
  );
}
```

---

## UI 컴포넌트

### 차트 컴포넌트 (Vega-Lite)

```tsx
// components/chart/VegaChart.tsx

import { VegaLite } from 'react-vega';

interface VegaChartProps {
  spec: any;
  data: any[];
}

export function VegaChart({ spec, data }: VegaChartProps) {
  const fullSpec = {
    ...spec,
    data: { values: data },
    width: 'container',
    height: 300,
  };

  return (
    <VegaLite
      spec={fullSpec}
      actions={{
        export: true,
        source: false,
        compiled: false,
        editor: false,
      }}
    />
  );
}
```

### 다이어그램 컴포넌트 (React Flow)

```tsx
// components/diagram/ModelDiagram.tsx

import ReactFlow, { Background, Controls } from 'reactflow';

export function ModelDiagram({ models, relations }) {
  const nodes = models.map(model => ({
    id: model.id.toString(),
    type: 'modelNode',
    position: model.position,
    data: { model },
  }));

  const edges = relations.map(rel => ({
    id: rel.id.toString(),
    source: rel.fromModelId.toString(),
    target: rel.toModelId.toString(),
    type: 'relationEdge',
    data: { relation: rel },
  }));

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
    >
      <Background />
      <Controls />
    </ReactFlow>
  );
}
```

---

*다음 글에서는 백엔드 API 구조를 살펴봅니다.*
