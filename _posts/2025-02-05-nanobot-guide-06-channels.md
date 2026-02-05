---
layout: post
title: "Nanobot 완벽 가이드 (6) - Channels 시스템"
date: 2025-02-05
permalink: /nanobot-guide-06-channels/
author: HKUDS
categories: [AI 에이전트, Nanobot]
tags: [Nanobot, Channels, Telegram, WhatsApp, Feishu]
original_url: "https://github.com/HKUDS/nanobot"
excerpt: "Nanobot의 멀티 채널 메시징 시스템을 분석합니다."
---

## Channels 시스템 개요

Nanobot은 다양한 메시징 플랫폼을 통해 AI 어시스턴트와 대화할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Channels 구조                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Telegram   │  │  WhatsApp   │  │   Feishu    │         │
│  │   Channel   │  │   Channel   │  │   Channel   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │      Message Bus      │                      │
│              │  (Inbound/Outbound)   │                      │
│              └───────────────────────┘                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 채널 베이스 클래스

```python
# channels/base.py

from abc import ABC, abstractmethod

class Channel(ABC):
    """채널 베이스 클래스"""

    def __init__(self, config: dict, bus: MessageBus):
        self.config = config
        self.bus = bus
        self._running = False

    @property
    @abstractmethod
    def name(self) -> str:
        """채널 이름"""
        pass

    @abstractmethod
    async def start(self) -> None:
        """채널 시작"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """채널 중지"""
        pass

    @abstractmethod
    async def send_message(self, chat_id: str, content: str) -> None:
        """메시지 전송"""
        pass

    def is_allowed(self, user_id: str) -> bool:
        """사용자 허용 여부 확인"""
        allow_from = self.config.get("allowFrom", [])
        if not allow_from:
            return True  # 제한 없음
        return str(user_id) in [str(x) for x in allow_from]
```

---

## Telegram 채널

가장 쉽게 설정할 수 있는 권장 채널입니다.

### 설정

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "123456:ABC-DEF...",
      "allowFrom": ["123456789"]
    }
  }
}
```

### 구현

```python
# channels/telegram.py

from telegram import Update
from telegram.ext import Application, MessageHandler, filters

class TelegramChannel(Channel):
    @property
    def name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        """Telegram 봇 시작"""
        self.app = Application.builder().token(self.config["token"]).build()

        # 메시지 핸들러 등록
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        self.app.add_handler(
            MessageHandler(filters.VOICE, self._handle_voice)
        )

        # 아웃바운드 메시지 구독
        asyncio.create_task(self._subscribe_outbound())

        # 봇 시작
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

        self._running = True
        logger.info("Telegram channel started")

    async def stop(self) -> None:
        """Telegram 봇 중지"""
        if self._running:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self._running = False

    async def _handle_message(self, update: Update, context) -> None:
        """텍스트 메시지 처리"""
        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)

        # 허용된 사용자인지 확인
        if not self.is_allowed(user_id):
            logger.warning(f"Unauthorized user: {user_id}")
            return

        # 메시지 버스로 전송
        await self.bus.publish_inbound(InboundMessage(
            channel="telegram",
            chat_id=chat_id,
            user_id=user_id,
            content=update.message.text,
        ))

    async def _handle_voice(self, update: Update, context) -> None:
        """음성 메시지 처리 (Whisper 전사)"""
        # 음성 파일 다운로드
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)

        # 임시 파일로 저장
        path = f"/tmp/{voice.file_id}.ogg"
        await file.download_to_drive(path)

        # Whisper로 전사 (Groq API 사용)
        text = await self._transcribe(path)

        if text:
            await self.bus.publish_inbound(InboundMessage(
                channel="telegram",
                chat_id=str(update.effective_chat.id),
                user_id=str(update.effective_user.id),
                content=text,
            ))

    async def send_message(self, chat_id: str, content: str) -> None:
        """메시지 전송"""
        # 긴 메시지 분할 (Telegram 제한: 4096자)
        max_length = 4096

        for i in range(0, len(content), max_length):
            chunk = content[i:i + max_length]
            await self.app.bot.send_message(
                chat_id=int(chat_id),
                text=chunk,
                parse_mode="Markdown",
            )

    async def _subscribe_outbound(self) -> None:
        """아웃바운드 메시지 구독"""
        async for message in self.bus.subscribe_outbound("telegram"):
            await self.send_message(message.chat_id, message.content)
```

### User ID 확인 방법

Telegram에서 `@userinfobot`에게 메시지를 보내면 본인의 User ID를 확인할 수 있습니다.

---

## WhatsApp 채널

QR 코드 스캔으로 연결하는 채널입니다.

### 요구사항

- Node.js 18 이상
- QR 코드 스캔을 위한 WhatsApp 앱

### 설정

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

### 아키텍처

WhatsApp 채널은 Node.js 브릿지를 사용합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    WhatsApp 아키텍처                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     HTTP      ┌─────────────────────┐     │
│  │   Python    │◄────────────►│    Node.js Bridge    │     │
│  │   Channel   │               │  (whatsapp-web.js)   │     │
│  └─────────────┘               └──────────┬──────────┘     │
│                                           │                 │
│                                           ▼                 │
│                                ┌─────────────────────┐     │
│                                │    WhatsApp Web     │     │
│                                └─────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Node.js 브릿지

```javascript
// channels/whatsapp/bridge.js

const { Client, LocalAuth } = require('whatsapp-web.js');
const express = require('express');
const qrcode = require('qrcode-terminal');

const app = express();
app.use(express.json());

const client = new Client({
    authStrategy: new LocalAuth({
        dataPath: process.env.SESSION_PATH || '~/.nanobot/whatsapp-session'
    }),
    puppeteer: {
        headless: true,
        args: ['--no-sandbox']
    }
});

// QR 코드 표시
client.on('qr', (qr) => {
    qrcode.generate(qr, { small: true });
    console.log('Scan the QR code above with WhatsApp');
});

// 연결 완료
client.on('ready', () => {
    console.log('WhatsApp client is ready!');
});

// 메시지 수신
client.on('message', async (msg) => {
    // Python 서버로 전달
    await fetch('http://localhost:18791/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            from: msg.from,
            body: msg.body,
            timestamp: msg.timestamp,
        })
    });
});

// 메시지 전송 API
app.post('/send', async (req, res) => {
    const { to, message } = req.body;
    await client.sendMessage(to, message);
    res.json({ success: true });
});

client.initialize();
app.listen(18792);
```

### Python 채널

```python
# channels/whatsapp.py

class WhatsAppChannel(Channel):
    @property
    def name(self) -> str:
        return "whatsapp"

    async def start(self) -> None:
        """WhatsApp 채널 시작"""
        # Node.js 브릿지 프로세스 시작
        self.bridge_process = await asyncio.create_subprocess_exec(
            "node",
            str(Path(__file__).parent / "whatsapp" / "bridge.js"),
            env={
                **os.environ,
                "SESSION_PATH": str(self.workspace / "whatsapp-session")
            }
        )

        # 웹훅 서버 시작
        self.webhook_app = web.Application()
        self.webhook_app.router.add_post('/message', self._handle_webhook)

        runner = web.AppRunner(self.webhook_app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 18791)
        await site.start()

        # 아웃바운드 구독
        asyncio.create_task(self._subscribe_outbound())

        self._running = True
        logger.info("WhatsApp channel started")

    async def _handle_webhook(self, request) -> web.Response:
        """Node.js 브릿지에서 메시지 수신"""
        data = await request.json()

        phone = data["from"].replace("@c.us", "")

        if not self.is_allowed(phone):
            return web.Response(text="unauthorized")

        await self.bus.publish_inbound(InboundMessage(
            channel="whatsapp",
            chat_id=data["from"],
            user_id=phone,
            content=data["body"],
        ))

        return web.Response(text="ok")

    async def send_message(self, chat_id: str, content: str) -> None:
        """Node.js 브릿지를 통해 메시지 전송"""
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://localhost:18792/send",
                json={"to": chat_id, "message": content}
            )
```

### 연결 방법

```bash
# 터미널 1: QR 코드 스캔
nanobot channels login

# 터미널 2: 게이트웨이 실행
nanobot gateway
```

---

## Feishu 채널

중국에서 널리 사용되는 Feishu(飞书) 메신저 연동입니다.

### 특징

- **WebSocket 롱 커넥션** 사용
- **퍼블릭 IP 불필요** (웹훅 불필요)
- 비즈니스 메시징에 적합

### 설정

```json
{
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "cli_xxx",
      "appSecret": "xxx",
      "encryptKey": "",
      "verificationToken": "",
      "allowFrom": []
    }
  }
}
```

### Feishu 앱 설정

1. [Feishu Open Platform](https://open.feishu.cn/app) 접속
2. 새 앱 생성
3. **Bot** 기능 활성화
4. 권한 추가: `im:message`
5. 이벤트 구독: `im.message.receive_v1` (Long Connection 모드)
6. App ID와 App Secret 복사

### 구현

```python
# channels/feishu.py

import lark_oapi as lark
from lark_oapi.api.im.v1 import *

class FeishuChannel(Channel):
    @property
    def name(self) -> str:
        return "feishu"

    async def start(self) -> None:
        """Feishu 채널 시작 (WebSocket)"""

        # Lark 클라이언트 생성
        self.client = lark.Client.builder() \
            .app_id(self.config["appId"]) \
            .app_secret(self.config["appSecret"]) \
            .build()

        # WebSocket 이벤트 핸들러
        event_handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self._handle_message) \
            .build()

        # WebSocket 클라이언트
        self.ws_client = lark.ws.Client(
            self.config["appId"],
            self.config["appSecret"],
            event_handler=event_handler,
            log_level=lark.LogLevel.DEBUG,
        )

        # 연결 시작
        asyncio.create_task(self._run_websocket())
        asyncio.create_task(self._subscribe_outbound())

        self._running = True
        logger.info("Feishu channel started")

    async def _run_websocket(self) -> None:
        """WebSocket 연결 유지"""
        self.ws_client.start()

    def _handle_message(self, data: P2ImMessageReceiveV1) -> None:
        """메시지 수신 핸들러"""
        event = data.event
        message = event.message
        sender = event.sender

        # 봇 메시지 무시
        if sender.sender_type == "bot":
            return

        user_id = sender.sender_id.user_id
        chat_id = message.chat_id

        # 메시지 내용 파싱
        content = json.loads(message.content)
        text = content.get("text", "")

        if not self.is_allowed(user_id):
            return

        # 비동기 처리를 위해 태스크 생성
        asyncio.create_task(self.bus.publish_inbound(InboundMessage(
            channel="feishu",
            chat_id=chat_id,
            user_id=user_id,
            content=text,
        )))

    async def send_message(self, chat_id: str, content: str) -> None:
        """Feishu로 메시지 전송"""
        request = CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("text")
                .content(json.dumps({"text": content}))
                .build()) \
            .build()

        response = await asyncio.to_thread(
            self.client.im.v1.message.create,
            request
        )

        if not response.success():
            logger.error(f"Feishu send failed: {response.msg}")
```

---

## 게이트웨이

모든 채널을 통합 관리하는 게이트웨이입니다.

```python
# gateway.py

class Gateway:
    """멀티 채널 게이트웨이"""

    def __init__(self, config: Config, bus: MessageBus):
        self.config = config
        self.bus = bus
        self.channels: list[Channel] = []

    async def start(self) -> None:
        """활성화된 모든 채널 시작"""
        channel_config = self.config.get("channels", {})

        # Telegram
        if channel_config.get("telegram", {}).get("enabled"):
            channel = TelegramChannel(
                config=channel_config["telegram"],
                bus=self.bus
            )
            self.channels.append(channel)

        # WhatsApp
        if channel_config.get("whatsapp", {}).get("enabled"):
            channel = WhatsAppChannel(
                config=channel_config["whatsapp"],
                bus=self.bus
            )
            self.channels.append(channel)

        # Feishu
        if channel_config.get("feishu", {}).get("enabled"):
            channel = FeishuChannel(
                config=channel_config["feishu"],
                bus=self.bus
            )
            self.channels.append(channel)

        # 모든 채널 시작
        for channel in self.channels:
            await channel.start()
            logger.info(f"Channel {channel.name} started")

    async def stop(self) -> None:
        """모든 채널 중지"""
        for channel in self.channels:
            await channel.stop()
```

### CLI 명령어

```bash
# 게이트웨이 시작
nanobot gateway

# 채널 상태 확인
nanobot channels status

# WhatsApp 로그인
nanobot channels login
```

---

## 채널 비교

| 특징 | Telegram | WhatsApp | Feishu |
|------|----------|----------|--------|
| **설정 난이도** | Easy | Medium | Medium |
| **요구사항** | Bot Token | Node.js + QR | App 자격증명 |
| **퍼블릭 IP** | 불필요 | 불필요 | 불필요 |
| **메시지 제한** | 4096자 | 없음 | 없음 |
| **음성 지원** | ✅ | ❌ | ❌ |
| **그룹 지원** | ✅ | ✅ | ✅ |

---

## 보안 고려사항

### allowFrom 설정

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "...",
      "allowFrom": ["123456789", "987654321"]
    }
  }
}
```

- 빈 배열 `[]`: 모든 사용자 허용 (주의!)
- ID 목록: 지정된 사용자만 허용

### 토큰 보안

- API 토큰은 절대 코드에 하드코딩하지 마세요
- `~/.nanobot/config.json`의 권한을 `600`으로 설정
- 환경 변수 사용도 고려

```bash
chmod 600 ~/.nanobot/config.json
```

---

*다음 글에서는 Skills 시스템을 분석합니다.*
