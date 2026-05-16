"""
Open WebUI Channels Platform Adapter for Hermes Agent.

Connects to Open WebUI via Socket.IO (real-time WebSocket) — the
officially supported method for bot integrations.  Follows the
same pattern as github.com/open-webui/bot.

Every @mention creates a new thread.  Messages in that thread
become the session context.  Because this is Socket.IO, messages
arrive in real-time — no polling.

Configuration (config.yaml):
    gateway:
      platforms:
        openwebui:
          enabled: true
          extra:
            url: http://openwebui:3000
            api_key: "sk-..."
            channel_name: "#hermes"

Env vars:
    OPENWEBUI_URL, OPENWEBUI_API_KEY, OPENWEBUI_CHANNEL_NAME
    HERMES_API_URL (default: http://127.0.0.1:8642/v1)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import aiohttp
import socketio

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.config import PlatformConfig, Platform

logger = logging.getLogger(__name__)

BOT_NAME = "hermes"


# ── Thread state ───────────────────────────────────────────────────────

class ThreadState:
    """Conversation history for one channel thread."""

    def __init__(self, parent_id: str):
        self.parent_id: str = parent_id
        self.history: List[Dict[str, str]] = []
        self.pending_run_id: Optional[str] = None
        self.pending_session_key: Optional[str] = None
        self.last_reply_id: Optional[str] = None

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})


# ── Adapter ────────────────────────────────────────────────────────────

class OpenWebUIAdapter(BasePlatformAdapter):

    def __init__(self, config: PlatformConfig, **kwargs):
        platform = Platform("openwebui")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        self.base_url: str = (
            os.getenv("OPENWEBUI_URL") or extra.get("url", "")
        ).rstrip("/")
        self.api_key: str = (
            os.getenv("OPENWEBUI_API_KEY") or extra.get("api_key", "")
        )
        self.channel_name: str = (
            os.getenv("OPENWEBUI_CHANNEL_NAME") or extra.get("channel_name", "")
        )

        # Hermes API server
        self.hermes_api_url: str = (
            os.getenv("HERMES_API_URL")
            or extra.get("hermes_api_url", "http://127.0.0.1:8642/v1")
        ).rstrip("/")
        self.hermes_api_key: str = (
            os.getenv("HERMES_API_KEY") or extra.get("hermes_api_key", "")
        )

        # Runtime state
        self._bot_user_id: Optional[str] = None
        self._bot_user_name: str = ""
        self._channel_id: Optional[str] = None

        # Socket.IO
        self._sio: Optional[socketio.AsyncClient] = None
        self._connected_event = asyncio.Event()
        self._http: Optional[aiohttp.ClientSession] = None

        # Threads: thread_parent_id -> ThreadState
        self._threads: Dict[str, ThreadState] = {}

        self.home_channel: str = (
            os.getenv("OPENWEBUI_HOME_CHANNEL") or self.channel_name
        )

    @property
    def name(self) -> str:
        return "Open WebUI"

    # ── HTTP helpers (for REST calls like auth) ──

    def _ow_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _ow_get(self, path: str) -> Any:
        if not self._http:
            return None
        async with self._http.get(
            f"{self.base_url}{path}", headers=self._ow_headers()
        ) as r:
            if r.status >= 400:
                logger.warning("GET %s -> %s", path, r.status)
                return None
            try:
                return await r.json()
            except Exception:
                return None

    # ── Lifecycle ──

    async def connect(self) -> bool:
        if not self.base_url or not self.api_key:
            self._set_fatal_error(
                "config_missing",
                "OPENWEBUI_URL and OPENWEBUI_API_KEY required",
                retryable=False,
            )
            return False

        self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(30))

        # Auth check
        me = await self._ow_get("/api/v1/auths/")
        if not isinstance(me, dict) or not me.get("id"):
            logger.error("Open WebUI: auth failed at %s/api/v1/auths/", self.base_url)
            self._set_fatal_error("auth_failed", "Check API key", retryable=True)
            await self._http.close()
            self._http = None
            return False

        self._bot_user_id = me.get("id", "")
        self._bot_user_name = (me.get("name") or "").lower()
        logger.info("Open WebUI: authed as %s (id=%s)", me.get("name"), self._bot_user_id)

        # Resolve channel ID from name
        ch_id = await self._resolve_channel_id_by_api(me.get("id"))
        if ch_id:
            self._channel_id = ch_id
            logger.info("Open WebUI: resolved channel %s = %s", self.channel_name, ch_id)
        else:
            logger.warning(
                "Open WebUI: channel %r not found via API; will discover via socket",
                self.channel_name,
            )

        # Connect Socket.IO
        self._sio = socketio.AsyncClient(logger=False, engineio_logger=False)
        self._register_handlers()

        try:
            await self._sio.connect(
                self.base_url,
                socketio_path="/ws/socket.io",
                transports=["websocket"],
                auth={"token": self.api_key},
            )
            logger.info("Open WebUI: Socket.IO connected")
        except Exception as e:
            logger.error("Open WebUI: Socket.IO connect failed: %s", e)
            self._set_fatal_error("socket_connect", str(e), retryable=True)
            await self._http.close()
            self._http = None
            return False

        # Join channels
        def join_callback(data):
            logger.debug("Open WebUI: joined user channels: %s", data)

        await self._sio.emit(
            "user-join",
            {"auth": {"token": self.api_key}},
            callback=join_callback,
        )

        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._sio and self._sio.connected:
            await self._sio.disconnect()
        if self._http:
            await self._http.close()

    # ── Channel resolution via REST ──

    async def _resolve_channel_id_by_api(self, user_id: str) -> Optional[str]:
        """Try public REST endpoints for channel discovery."""
        # Try the users/channels endpoint
        channels = await self._ow_get(f"/api/v1/channels/users/{user_id}")
        if isinstance(channels, list):
            search = self.channel_name.lstrip("#").lower()
            for ch in channels:
                ch_name = str(ch.get("name", "")).strip().lower()
                ch_id = str(ch.get("id", ""))
                if ch_name == search:
                    return ch_id
        # Try direct lookup
        for path in ("/api/v1/channels/", "/api/v1/channels"):
            clist = await self._ow_get(path)
            if isinstance(clist, list):
                search = self.channel_name.lstrip("#").lower()
                for ch in clist:
                    ch_name = str(ch.get("name", "")).strip().lower()
                    ch_id = str(ch.get("id", ""))
                    if ch_name == search:
                        return ch_id
        return None

    # ── Socket.IO handlers ──

    def _register_handlers(self) -> None:
        sio = self._sio

        @sio.on("connect")
        async def on_connect():
            logger.info("Open WebUI: socket connected")

        @sio.on("disconnect")
        async def on_disconnect():
            logger.info("Open WebUI: socket disconnected")

        @sio.on("channel-events")
        async def on_channel_events(data: dict):
            await self._handle_channel_event(data)

    async def _handle_channel_event(self, data: dict) -> None:
        """Process an incoming channel event."""
        # Ignore events without proper structure
        event_data = data.get("data", {})
        if not isinstance(event_data, dict):
            return

        event_type = event_data.get("type")
        if event_type != "message":
            return

        # Extract message info
        msg_data = event_data.get("data", {})
        if not isinstance(msg_data, dict):
            return

        ch_id = data.get("channel_id", "")
        msg_id = msg_data.get("id", "")
        content = msg_data.get("content", "")
        user = data.get("user", {})
        user_id = user.get("id", "") if isinstance(user, dict) else ""
        user_name = user.get("name", "") if isinstance(user, dict) else ""
        parent_id = msg_data.get("parent_id")

        # Ignore own messages
        if user_id == self._bot_user_id:
            return

        if not content or not content.strip():
            return

        logger.debug("channel msg from %s: %s", user_name, content[:60])

        # Determine thread context
        if parent_id:
            thread_id = parent_id
        else:
            top_handles = {f"@{BOT_NAME}", f"@{self._bot_user_name}", f"@{user_name}"}
            content_lower = content.lower().strip()
            if not any(content_lower.startswith(h) or content_lower.startswith(h + " ") for h in top_handles):
                return  # not addressed to us
            thread_id = msg_id  # this message becomes the thread parent

        # Dispatch
        asyncio.create_task(
            self._process_message(content.strip(), thread_id, user_name, ch_id)
        )

    # ── Message processing ──

    def _get_or_create_thread(self, thread_id: str) -> ThreadState:
        if thread_id not in self._threads:
            self._threads[thread_id] = ThreadState(thread_id)
        return self._threads[thread_id]

    APPROVE_WORDS = frozenset({"approve", "approved", "yes", "allow", "once"})
    DENY_WORDS = frozenset({"deny", "denied", "no", "reject"})

    def _map_approval_choice(self, text: str) -> Optional[str]:
        cleaned = text.strip().lower().rstrip(".!").strip()
        if cleaned in self.DENY_WORDS:
            return "deny"
        if cleaned == "always":
            return "always"
        if cleaned == "session":
            return "session"
        if cleaned in self.APPROVE_WORDS:
            return "once"
        return None

    def _session_key(self, thread_id: str) -> str:
        return f"owui:{self._channel_id}:{thread_id}"

    async def _process_message(
        self, content: str, thread_id: str, user_name: str, ch_id: str
    ) -> None:
        thread = self._get_or_create_thread(thread_id)

        # ── Approval response? ──
        if thread.pending_run_id:
            choice = self._map_approval_choice(content)
            if choice:
                await self._resolve_approval(thread, choice)
                return

        # ── Normal ──
        thread.add_user_message(content)
        await self._run_hermes(thread, content)

    async def _resolve_approval(self, thread: ThreadState, choice: str) -> None:
        run_id = thread.pending_run_id
        if not run_id:
            return
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(30)) as s:
                async with s.post(
                    f"{self.hermes_api_url}/runs/{run_id}/approval",
                    headers=self._hermes_headers(),
                    json={"choice": choice},
                ) as r:
                    if r.status < 400:
                        await self._send_socket(
                            thread.parent_id,
                            f"✅ **{choice}**",
                        )
        except Exception as e:
            logger.error("approval error: %s", e)
        thread.pending_run_id = None
        thread.pending_session_key = None

    def _hermes_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.hermes_api_key:
            h["Authorization"] = f"Bearer {self.hermes_api_key}"
        return h

    async def _run_hermes(self, thread: ThreadState, message: str) -> None:
        """Submit to Hermes via /v1/runs and post the response as a thread reply."""
        session_id = self._session_key(thread.parent_id)
        parent_id = thread.parent_id

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(600)) as http:
                # Start run
                async with http.post(
                    f"{self.hermes_api_url}/runs",
                    headers=self._hermes_headers(),
                    json={
                        "input": message,
                        "conversation_history": list(thread.history[:-1]),
                        "session_id": session_id,
                    },
                ) as r:
                    if r.status >= 400:
                        err = await r.text()
                        await self._send_socket(parent_id, f"⚠️ Hermes error: {err[:200]}")
                        return
                    run_data = await r.json()
                    run_id = run_data["run_id"]

                # Consume SSE
                accumulated = ""
                async with http.get(
                    f"{self.hermes_api_url}/runs/{run_id}/events",
                    headers=self._hermes_headers(),
                ) as sse:
                    buf = ""
                    async for raw in sse.content:
                        buf += raw.decode("utf-8")
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip()
                            if not line or line.startswith(":") or not line.startswith("data: "):
                                continue
                            try:
                                event = json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue
                            ev = event.get("event", "")

                            if ev == "message.delta":
                                d = event.get("delta", "")
                                if d:
                                    accumulated += d

                            elif ev == "approval.request":
                                prompt = event.get("message", event.get("preview", "Approve?"))
                                thread.pending_run_id = run_id
                                thread.pending_session_key = session_id
                                await self._send_socket(
                                    parent_id,
                                    f"⚠️ **Approval Required**\n```\n{prompt[:500]}\n```\n"
                                    f"Reply **`approve`** or **`deny`**.",
                                )
                                return

                            elif ev == "run.completed":
                                output = event.get("output", "")
                                if output and output not in accumulated:
                                    accumulated = output
                                if accumulated:
                                    thread.add_assistant_message(accumulated)
                                    await self._send_socket(parent_id, accumulated)
                                else:
                                    await self._send_socket(parent_id, "(no response)")
                                return

                            elif ev == "run.failed":
                                err = event.get("error", "Unknown")
                                await self._send_socket(parent_id, f"⚠️ **Error:** {err}")
                                return

                            elif ev == "run.cancelled":
                                await self._send_socket(parent_id, "*Cancelled.*")
                                return

                if accumulated:
                    thread.add_assistant_message(accumulated)
                    await self._send_socket(parent_id, accumulated)
                else:
                    await self._send_socket(parent_id, "(no response)")

        except aiohttp.ClientConnectorError:
            await self._send_socket(parent_id, f"⚠️ Cannot reach Hermes at `{self.hermes_api_url}`.")
        except Exception as e:
            logger.exception("process error")
            await self._send_socket(parent_id, f"⚠️ **Error:** {e}")

    # ── Socket send ──

    async def _send_socket(self, parent_id: str, text: str) -> None:
        """Post a message in the thread via Socket.IO."""
        if not self._sio or not self._sio.connected or not self._channel_id:
            logger.warning("Open WebUI: cannot send (socket disconnected)")
            return
        try:
            await self._sio.emit("channel-message", {
                "channel_id": self._channel_id,
                "data": {
                    "content": text,
                    "data": {"files": []},
                    "parent_id": parent_id,
                },
            })
        except Exception as e:
            logger.error("send socket error: %s", e)

    # ── BasePlatformAdapter send (for send_message tool / cron) ──

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        target = chat_id or self._channel_id
        if not target:
            return SendResult(success=False, error="No target")
        try:
            body = {"content": content, "data": {"files": []}}
            if reply_to:
                body["parent_id"] = reply_to
            if self._sio and self._sio.connected:
                await self._sio.emit("channel-message", {
                    "channel_id": self._channel_id,
                    "data": body,
                })
            return SendResult(success=True)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "Open WebUI", "type": "channel"}


# ── Plugin registration ────────────────────────────────────────────────

def check_requirements() -> bool:
    try:
        import socketio  # noqa: F401
        return True
    except ImportError:
        return False


def validate_config(config) -> bool:
    return True


def is_connected(adapter) -> bool:
    return getattr(adapter, "_is_connected", False)


def _env_enablement() -> Optional[dict]:
    url = os.getenv("OPENWEBUI_URL")
    key = os.getenv("OPENWEBUI_API_KEY")
    if not url or not key:
        return None
    extra = {"url": url, "api_key": key}
    ch = os.getenv("OPENWEBUI_CHANNEL_NAME")
    if ch:
        extra["channel_name"] = ch
        extra["home_channel"] = ch
    if os.getenv("HERMES_API_URL"):
        extra["hermes_api_url"] = os.getenv("HERMES_API_URL")
    if os.getenv("HERMES_API_KEY"):
        extra["hermes_api_key"] = os.getenv("HERMES_API_KEY")
    return extra


def register(ctx):
    ctx.register_platform(
        name="openwebui",
        label="Open WebUI",
        adapter_factory=lambda cfg: OpenWebUIAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["OPENWEBUI_URL", "OPENWEBUI_API_KEY"],
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="OPENWEBUI_HOME_CHANNEL",
        emoji="🌐",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Open WebUI Channels over WebSocket. "
            "Users @mention you to start a thread. Reply in the "
            "thread to continue the conversation. "
            "For approvals, describe the operation and the user can "
            "reply with 'approve' or 'deny'."
        ),
    )
