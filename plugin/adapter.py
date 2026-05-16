"""
Open WebUI Channels Platform Adapter for Hermes Agent.

Architecture:
  Every conversation lives in a channel *thread*.  When a user @mentions
  the bot in a top-level message, Hermes replies as a thread reply —
  that thread becomes the session.  Subsequent messages in the thread
  (no @mention needed) are continuations of that session.

  This gives us:
  - Clean session boundaries (one thread = one conversation)
  - Mid-response interjection (new message in the thread → background task)
  - Natural context grouping (the thread IS the conversation history)
  - Approval flow via message exchange in the thread

Config (config.yaml):
    gateway:
      platforms:
        openwebui:
          enabled: true
          extra:
            url: http://openwebui:3000
            api_key: "sk-..."
            channel: "#hermes"
            poll_interval: 3

Env vars:
    OPENWEBUI_URL, OPENWEBUI_API_KEY, OPENWEBUI_CHANNEL_NAME
    HERMES_API_URL (default: http://127.0.0.1:8642/v1)
    HERMES_API_KEY
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import aiohttp

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.config import PlatformConfig, Platform

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 3
BOT_NAME = "hermes"  # Used for @mention detection


# ── Per-thread state ───────────────────────────────────────────────────

class ThreadState:
    """
    Session state for one channel thread.

    The thread's parent message ID is the stable key.  History is
    accumulated here so the Hermes runs get the full conversation.
    """

    def __init__(self, parent_id: str):
        self.parent_id: str = parent_id
        self.history: List[Dict[str, str]] = []
        self.last_reply_id: Optional[str] = None
        # Approval flow
        self.pending_run_id: Optional[str] = None
        self.pending_session_key: Optional[str] = None

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})


# ── Open WebUI message ─────────────────────────────────────────────────

class OWMessage:
    """A message from the Open WebUI Channels API."""

    def __init__(self, raw: dict):
        self.id: str = raw.get("id", "")
        self.content: str = raw.get("content", "")
        self.parent_id: Optional[str] = raw.get("parent_id")  # None = top-level
        self.user_id: str = ""
        self.user_name: str = ""
        self.created_at: int = raw.get("created_at", 0)
        user = raw.get("user")
        if isinstance(user, dict):
            self.user_id = str(user.get("id", ""))
            self.user_name = str(user.get("name", ""))

    def is_top_level(self) -> bool:
        return not self.parent_id

    def mentions_bot(self, bot_user_id: str) -> bool:
        """Check if content @mentions the bot user."""
        if not self.content:
            return False
        lower = self.content.lower()
        # Match @User Name, @username, or BOT_NAME
        if f"@{BOT_NAME}" in lower:
            return True
        if bot_user_id and f"@{bot_user_id}" in lower:
            return True
        return False

    def __repr__(self) -> str:
        parent = f" -> {self.parent_id}" if self.parent_id else ""
        return (
            f"<OWMessage {self.id}{parent} "
            f"{self.user_name}:{self.content[:50]!r}>"
        )


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
            os.getenv("OPENWEBUI_CHANNEL_NAME") or extra.get("channel", "")
        )
        raw_interval = (
            os.getenv("OPENWEBUI_POLL_INTERVAL")
            or extra.get("poll_interval", DEFAULT_POLL_INTERVAL)
        )
        try:
            self.poll_interval = max(1, int(raw_interval))
        except (TypeError, ValueError):
            self.poll_interval = DEFAULT_POLL_INTERVAL

        # Hermes API server
        self.hermes_api_url: str = (
            os.getenv("HERMES_API_URL")
            or extra.get("hermes_api_url", "http://127.0.0.1:8642/v1")
        ).rstrip("/")
        self.hermes_api_key: str = (
            os.getenv("HERMES_API_KEY") or extra.get("hermes_api_key", "")
        )

        # Runtime state
        self._channel_id: Optional[str] = None
        self._bot_user_id: Optional[str] = None
        self._bot_user_name: str = ""
        self._poll_task: Optional[asyncio.Task] = None
        self._http: Optional[aiohttp.ClientSession] = None

        # Thread state: thread_parent_id → ThreadState
        self._threads: Dict[str, ThreadState] = {}
        # Channel watermark: last-seen message ID
        self._last_seen: Optional[str] = None

        self.home_channel: str = (
            os.getenv("OPENWEBUI_HOME_CHANNEL") or self.channel_name
        )

    @property
    def name(self) -> str:
        return "Open WebUI"

    # ── HTTP ──

    def _ow_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _hermes_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.hermes_api_key:
            h["Authorization"] = f"Bearer {self.hermes_api_key}"
        return h

    def _api_url(self, path: str) -> str:
        return f"{self.base_url}/api{path}"

    async def _ow_get(self, path: str) -> Any:
        async with self._http.get(
            self._api_url(path), headers=self._ow_headers()
        ) as r:
            if r.status >= 400:
                text = await r.text()
                logger.warning("GET %s → %s: %s", path, r.status, text[:200])
                return None
            try:
                return await r.json()
            except Exception:
                text = await r.text()
                logger.warning("GET %s → 200 but non-JSON response (%s...)", path, text[:80])
                return None

    async def _ow_post(self, path: str, body: dict = None) -> Any:
        async with self._http.post(
            self._api_url(path), headers=self._ow_headers(), json=body or {}
        ) as r:
            if r.status >= 400:
                text = await r.text()
                logger.error("POST %s → %s: %s", path, r.status, text[:200])
                return None
            return await r.json()

    # ── Lifecycle ──

    async def connect(self) -> bool:
        if not self.base_url or not self.api_key or not self.channel_name:
            self._set_fatal_error(
                "config_missing",
                "OPENWEBUI_URL, OPENWEBUI_API_KEY, OPENWEBUI_CHANNEL_NAME required",
                retryable=False,
            )
            return False

        self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60))

        # ── Auth: try known endpoints ──
        me = None
        for auth_path in ("/v1/auths/", "/auths/", "/v1/auth/"):
            me = await self._ow_get(auth_path)
            if isinstance(me, dict) and me.get("id"):
                logger.info("auth endpoint found: %s", auth_path)
                break

        if not isinstance(me, dict) or not me.get("id"):
            logger.error(
                "Open WebUI: auth failed — tried /v1/auths/, /auths/, /v1/auth/.\n"
                "  Check that your API key is valid and has permissions.\n"
                "  URL: %s\n  Key starts with: %s...",
                self.base_url, self.api_key[:12] if len(self.api_key) > 12 else "(empty)",
            )
            self._set_fatal_error("auth_failed", "Cannot auth. Check API key.", retryable=True)
            await self._http.close()
            self._http = None
            return False

        self._bot_user_id = me.get("id", "")
        self._bot_user_name = (me.get("name") or "").lower()
        logger.info(
            "Open WebUI: authed as %r (id=%s name=%s)",
            me.get("name"), self._bot_user_id, self._bot_user_name,
        )

        # Resolve channel
        self._channel_id = await self._resolve_channel()
        if not self._channel_id:
            self._set_fatal_error(
                "channel_not_found", f"Channel {self.channel_name!r} not found",
                retryable=True,
            )
            await self._http.close()
            self._http = None
            return False
        logger.info("Open WebUI: channel %s = %s", self.channel_name, self._channel_id)

        # Catch up to latest message
        try:
            msgs = await self._ow_get(f"/channels/{self._channel_id}/messages?limit=5")
            if isinstance(msgs, list) and msgs:
                latest = max(
                    (OWMessage(m) for m in msgs),
                    key=lambda m: m.created_at or 0,
                    default=None,
                )
                if latest:
                    self._last_seen = latest.id
        except Exception as e:
            logger.warning("initial catch-up: %s", e)

        self._poll_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._poll_task = None
        if self._http:
            await self._http.close()
            self._http = None

    async def _resolve_channel(self) -> Optional[str]:
        name = self.channel_name.strip()
        if name.startswith("ch_"):
            for prefix in ("/v1/channels/", "/channels/"):
                ch = await self._ow_get(f"{prefix}{name}")
                if isinstance(ch, dict) and ch.get("id"):
                    return ch["id"]
        # List channels — try both path variants
        for prefix in ("/v1/channels", "/channels/"):
            channels = await self._ow_get(prefix)
            if isinstance(channels, list):
                break
        if not isinstance(channels, list):
            return None
        search = name.lstrip("#").lower()
        for ch in channels:
            ch_name = str(ch.get("name", "")).strip().lower()
            ch_id = str(ch.get("id", ""))
            if ch_name == search or ch_id == name:
                return ch_id
        return None

    # ── Polling ──

    async def _poll_loop(self) -> None:
        while self._is_connected:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("poll error: %s", e, exc_info=True)
            await asyncio.sleep(self.poll_interval)

    async def _poll_once(self) -> None:
        """Fetch new messages and dispatch to the right thread."""
        msgs = await self._ow_get(
            f"/channels/{self._channel_id}/messages?limit=20"
        )
        if not isinstance(msgs, list):
            return

        for raw in msgs:
            msg = OWMessage(raw)
            if self._last_seen and msg.id <= self._last_seen:
                continue
            self._last_seen = msg.id

            # Skip own messages
            if self._bot_user_id and msg.user_id == self._bot_user_id:
                continue

            content = msg.content.strip()
            if not content:
                continue

            # ── Determine which thread this belongs to ──
            if msg.is_top_level():
                # Top-level message → check for @mention to start a thread
                if not msg.mentions_bot(self._bot_user_id):
                    continue  # ignore unaddressed top-level messages
                # This starts a new thread. parent_id = this message
                thread_id = msg.id
                logger.info("new thread %s from %s", thread_id, msg.user_name)
            else:
                # Reply in an existing thread
                thread_id = msg.parent_id
                logger.debug("thread reply %s in %s", msg.id, thread_id)

            # Dispatch to background task
            asyncio.create_task(
                self._handle_message(content, msg, thread_id)
            )

    # ── Message handling ──

    async def _handle_message(
        self, content: str, msg: OWMessage, thread_id: str
    ) -> None:
        """Process one message in its thread context."""
        thread = self._get_or_create_thread(thread_id)

        # ── Approval response? ──
        if thread.pending_run_id:
            choice = self._parse_approval_choice(content)
            if choice:
                await self._resolve_approval(thread, choice, thread_id)
                return

        # ── Normal message ──
        thread.add_user_message(content)
        await self._run_hermes(thread, content, thread_id)

    def _get_or_create_thread(self, thread_id: str) -> ThreadState:
        if thread_id not in self._threads:
            self._threads[thread_id] = ThreadState(thread_id)
        return self._threads[thread_id]

    # ── Approval ──

    APPROVE_WORDS = frozenset({"approve", "approved", "yes", "allow", "once"})
    DENY_WORDS = frozenset({"deny", "denied", "no", "reject"})

    def _parse_approval_choice(self, text: str) -> Optional[str]:
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

    async def _resolve_approval(
        self, thread: ThreadState, choice: str, thread_id: str
    ) -> None:
        run_id = thread.pending_run_id
        if not run_id:
            return
        session_key = thread.pending_session_key
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(30)) as s:
                async with s.post(
                    f"{self.hermes_api_url}/runs/{run_id}/approval",
                    headers=self._hermes_headers(),
                    json={"choice": choice},
                ) as r:
                    if r.status < 400:
                        await self._post_thread_reply(
                            thread_id, f"✅ **{choice}** — continuing…"
                        )
                    else:
                        await self._post_thread_reply(
                            thread_id, "⚠️ Approval submission failed."
                        )
        except Exception as e:
            logger.error("approval error: %s", e)
        thread.pending_run_id = None
        thread.pending_session_key = None

    # ── Hermes execution ──

    def _session_key(self, thread_id: str) -> str:
        return f"owui:{self._channel_id}:{thread_id}"

    async def _run_hermes(
        self, thread: ThreadState, message: str, thread_id: str
    ) -> None:
        """Submit a message to Hermes via /v1/runs and post the response."""
        session_id = self._session_key(thread_id)

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(600)
            ) as hermes:

                # Start run
                async with hermes.post(
                    f"{self.hermes_api_url}/runs",
                    headers=self._hermes_headers(),
                    json={
                        "input": message,
                        "conversation_history": list(thread.history[:-1]),
                        "session_id": session_id,
                    },
                ) as r:
                    if r.status >= 400:
                        err_body = await r.text()
                        await self._post_thread_reply(
                            thread_id,
                            f"⚠️ **Hermes error** ({r.status}): {err_body[:300]}",
                        )
                        return
                    run_data = await r.json()
                    run_id = run_data["run_id"]

                # Consume SSE events
                accumulated = ""
                async with hermes.get(
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
                                await self._post_thread_reply(
                                    thread_id,
                                    f"⚠️ **Approval Required**\n```\n{prompt[:500]}\n```\n"
                                    f"Reply **`approve`** or **`deny`**.",
                                )
                                return  # agent blocks; next user msg resolves

                            elif ev == "run.completed":
                                output = event.get("output", "")
                                if output and output not in accumulated:
                                    accumulated = output
                                if accumulated:
                                    thread.add_assistant_message(accumulated)
                                    await self._post_thread_reply(thread_id, accumulated)
                                else:
                                    await self._post_thread_reply(
                                        thread_id, "(no response)"
                                    )
                                return

                            elif ev == "run.failed":
                                err = event.get("error", "Unknown error")
                                await self._post_thread_reply(
                                    thread_id, f"⚠️ **Error:** {err}"
                                )
                                return

                            elif ev == "run.cancelled":
                                await self._post_thread_reply(
                                    thread_id, "*Cancelled.*"
                                )
                                return

                # SSE ended without terminal event
                if accumulated:
                    thread.add_assistant_message(accumulated)
                    await self._post_thread_reply(thread_id, accumulated)
                else:
                    await self._post_thread_reply(thread_id, "(no response)")

        except asyncio.TimeoutError:
            await self._post_thread_reply(thread_id, "⚠️ **Hermes timed out.**")
        except aiohttp.ClientConnectorError:
            await self._post_thread_reply(
                thread_id,
                f"⚠️ **Cannot connect** to Hermes at `{self.hermes_api_url}`.",
            )
        except Exception as e:
            logger.exception("processing error")
            await self._post_thread_reply(thread_id, f"⚠️ **Error:** {e}")

    # ── Sending ──

    async def _post_thread_reply(self, thread_parent_id: str, text: str) -> None:
        """
        Post a reply in a thread.

        The reply is addressed to the thread's parent message.
        This keeps all Hermes responses in the same thread.
        """
        if not self._http or not self._channel_id:
            return
        body = {"content": text, "data": {"files": []}, "parent_id": thread_parent_id}
        await self._ow_post(f"/channels/{self._channel_id}/messages/post", body)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Send a message via the send_message tool or cron delivery.

        chat_id can be:
        - A thread parent ID (prefixed with "thread:" → reply in that thread)
        - A channel ID (top-level post)
        - Empty → uses the home channel as a top-level post
        """
        target = chat_id or self._channel_id
        if not target:
            return SendResult(success=False, error="No target configured")
        try:
            body: Dict[str, Any] = {"content": content, "data": {"files": []}}

            # If target is "thread:PARENT_ID", post as a thread reply
            if target.startswith("thread:") and self._channel_id:
                parent_id = target[7:]
                body["parent_id"] = parent_id
                await self._ow_post(
                    f"/channels/{self._channel_id}/messages/post", body
                )
            else:
                await self._ow_post(
                    f"/channels/{target}/messages/post", body
                )
            return SendResult(success=True)
        except Exception as e:
            logger.error("send error: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str) -> None:
        pass  # not supported by Open WebUI Channels API

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        if not self._http or not self._channel_id:
            return {"name": "Open WebUI", "type": "channel"}
        ch = await self._ow_get(f"/channels/{chat_id or self._channel_id}")
        if isinstance(ch, dict):
            return {
                "name": ch.get("name", "Open WebUI"),
                "type": "channel",
                "chat_id": chat_id,
            }
        return {"name": "Open WebUI", "type": "channel", "chat_id": chat_id}


# ── Plugin registration ────────────────────────────────────────────────

def check_requirements() -> bool:
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False


def validate_config(config) -> bool:
    """Validate required config. Returns True if we have enough to attempt a connection."""
    # We accept the config here and let connect() report detailed failures.
    # This avoids false negatives when env vars haven't been surfaced yet.
    return True


def is_connected(adapter) -> bool:
    return getattr(adapter, "_is_connected", False)


async def interactive_setup(ctx) -> Optional[dict]:
    from hermes_cli.setup_prompts import prompt_text, prompt_optional
    url = await prompt_text("Open WebUI URL", "http://localhost:3000")
    api_key = await prompt_text("API key", password=True)
    channel = await prompt_text("Channel name", "#hermes")
    poll = await prompt_optional("Poll interval (seconds)", "3")
    result = {"url": url, "api_key": api_key, "channel": channel}
    if poll:
        result["poll_interval"] = int(poll)
    return result


def _env_enablement() -> Optional[dict]:
    url = os.getenv("OPENWEBUI_URL")
    key = os.getenv("OPENWEBUI_API_KEY")
    ch = os.getenv("OPENWEBUI_CHANNEL_NAME")
    if not url and not key and not ch:
        return None
    extra = {}
    if url:
        extra["url"] = url
    if key:
        extra["api_key"] = key
    if ch:
        extra["channel"] = ch
        extra["home_channel"] = ch
    if os.getenv("HERMES_API_URL"):
        extra["hermes_api_url"] = os.getenv("HERMES_API_URL")
    if os.getenv("HERMES_API_KEY"):
        extra["hermes_api_key"] = os.getenv("HERMES_API_KEY")
    return extra


async def _standalone_send(config, chat_id: str, text: str, reply_to: str = None) -> dict:
    extra = (getattr(config, "extra", {}) if not isinstance(config, dict) else config.get("extra", {})) or {}
    base_url = (extra.get("url") or "").rstrip("/")
    api_key = extra.get("api_key") or ""
    target = chat_id or extra.get("channel", "")
    if not base_url or not api_key or not target:
        return {"success": False, "error": "Missing config"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/api/channels/", headers=headers) as r:
            if r.status >= 400:
                return {"success": False, "error": await r.text()}
            chans = await r.json()
        ch_id = None
        search = target.lstrip("#").lower()
        if isinstance(chans, list):
            for ch in chans:
                if str(ch.get("name", "")).strip().lower() == search:
                    ch_id = ch.get("id")
                    break
        if not ch_id:
            return {"success": False, "error": f"Channel {target!r} not found"}
        body = {"content": text, "data": {"files": []}}
        if reply_to:
            body["parent_id"] = reply_to
        async with session.post(
            f"{base_url}/api/channels/{ch_id}/messages/post",
            headers=headers, json=body,
        ) as r:
            return {"success": r.status < 400, "error": await r.text() if r.status >= 400 else ""}


def register(ctx):
    ctx.register_platform(
        name="openwebui",
        label="Open WebUI",
        adapter_factory=lambda cfg: OpenWebUIAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["OPENWEBUI_URL", "OPENWEBUI_API_KEY", "OPENWEBUI_CHANNEL_NAME"],
        install_hint="No extra packages (aiohttp ships with gateway)",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="OPENWEBUI_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        emoji="🌐",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Open WebUI Channels. "
            "You operate in message threads.  Each thread is one "
            "conversation — users @mention you to start a new thread. "
            "Once a thread exists, users can reply without @mentioning. "
            "For approvals, describe the operation and the user can "
            "reply with 'approve' or 'deny' in the same thread."
        ),
    )
