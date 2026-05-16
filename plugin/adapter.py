"""
Open WebUI Channels Platform Adapter for Hermes Agent.

Connects to an Open WebUI instance via its REST API, watches a
designated channel for new messages, and relays them to the Hermes
agent for processing.  Messages are independent API calls, so you
can interject mid-response — just like Discord or Telegram.

Configuration (config.yaml):
    gateway:
      platforms:
        openwebui:
          enabled: true
          extra:
            url: http://openwebui:3000
            api_key: "sk-..."
            channel: "#hermes"
            poll_interval: 3

Or via environment variables:
    OPENWEBUI_URL, OPENWEBUI_API_KEY, OPENWEBUI_CHANNEL_NAME
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import aiohttp

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    MessageEvent,
    MessageType,
)
from gateway.config import PlatformConfig, Platform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_POLL_INTERVAL = 3  # seconds


def _strip_json_block(text: str) -> str:
    """Remove leading/trailing ```json … ``` fences if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ---------------------------------------------------------------------------
# Message envelope from Open WebUI Channel API
# ---------------------------------------------------------------------------

class OWMessage:
    """Minimal parsed representation of a channel message."""

    def __init__(self, raw: dict):
        self.id: str = raw.get("id", "")
        self.content: str = raw.get("content", "")
        self.user_id: str = ""
        self.user_name: str = ""
        self.created_at: int = raw.get("created_at", 0)

        user = raw.get("user")
        if isinstance(user, dict):
            self.user_id = str(user.get("id", ""))
            self.user_name = str(user.get("name", ""))

    @property
    def is_bot(self) -> bool:
        """Return True if this message was sent by the bot user itself."""
        return False  # overridden per-account after auth

    def __repr__(self) -> str:
        return (
            f"<OWMessage id={self.id} user={self.user_name} "
            f"content={self.content[:60]!r}>"
        )


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class OpenWebUIAdapter(BasePlatformAdapter):
    """
    Poll-based Open WebUI Channels adapter.

    The bot authenticates as an Open WebUI user (via API key), joins a
    channel, and polls for new messages.  Each human message triggers an
    agent run; the response is posted back to the same channel.
    """

    def __init__(self, config: PlatformConfig, **kwargs):
        platform = Platform("openwebui")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        # ── Connection settings ──
        self.base_url: str = (
            os.getenv("OPENWEBUI_URL") or extra.get("url", "")
        ).rstrip("/")
        self.api_key: str = (
            os.getenv("OPENWEBUI_API_KEY") or extra.get("api_key", "")
        )
        self.channel_name: str = (
            os.getenv("OPENWEBUI_CHANNEL_NAME") or extra.get("channel", "")
        )

        # ── Polling ──
        raw_interval = (
            os.getenv("OPENWEBUI_POLL_INTERVAL") or extra.get("poll_interval", DEFAULT_POLL_INTERVAL)
        )
        try:
            self.poll_interval = max(1, int(raw_interval))
        except (TypeError, ValueError):
            self.poll_interval = DEFAULT_POLL_INTERVAL

        # ── Runtime state ──
        self._channel_id: Optional[str] = None
        self._bot_user_id: Optional[str] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._http: Optional[aiohttp.ClientSession] = None
        # Last message ID we've seen per channel
        self._last_seen: Optional[str] = None
        # Pending approval session keys → bot messages we posted
        self._pending_approvals: Dict[str, str] = {}  # session_key → channel+msg info

        # The gateway sets this from env/config
        self.home_channel: str = (
            os.getenv("OPENWEBUI_HOME_CHANNEL") or self.channel_name
        )

    @property
    def name(self) -> str:
        return "Open WebUI"

    # ── Header builder ──

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _api(self, path: str) -> str:
        """Build a full API URL.  The Channels API lives under /api."""
        return f"{self.base_url}/api{path}"

    # ── API calls ──

    async def _get(self, path: str) -> Any:
        async with self._http.get(self._api(path), headers=self._headers()) as r:
            if r.status >= 400:
                text = await r.text()
                logger.error("GET %s → %s: %s", path, r.status, text[:200])
                return None
            return await r.json()

    async def _post(self, path: str, body: dict = None) -> Any:
        async with self._http.post(
            self._api(path), headers=self._headers(), json=body or {}
        ) as r:
            if r.status >= 400:
                text = await r.text()
                logger.error("POST %s → %s: %s", path, r.status, text[:200])
                return None
            return await r.json()

    # ── Connection lifecycle ──

    async def connect(self) -> bool:
        """Authenticate, resolve the channel, and start polling."""
        if not self.base_url or not self.api_key or not self.channel_name:
            logger.error(
                "Open WebUI: missing required config: url, api_key, channel"
            )
            self._set_fatal_error(
                "config_missing",
                "OPENWEBUI_URL, OPENWEBUI_API_KEY, and "
                "OPENWEBUI_CHANNEL_NAME must be set",
                retryable=False,
            )
            return False

        self._http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

        # ── Verify auth ──
        try:
            me = await self._get("/auth/")
            if me is None:
                logger.error("Open WebUI: auth check failed")
                self._set_fatal_error(
                    "auth_failed",
                    "Cannot authenticate with Open WebUI. "
                    "Check your API key.",
                    retryable=True,
                )
                await self._http.close()
                self._http = None
                return False
            self._bot_user_id = me.get("id") if isinstance(me, dict) else me.get("id")
            logger.info("Open WebUI: authenticated as user %s", self._bot_user_id)
        except Exception as e:
            logger.error("Open WebUI: auth error: %s", e)
            self._set_fatal_error("auth_failed", str(e), retryable=True)
            await self._http.close()
            self._http = None
            return False

        # ── Resolve channel ──
        self._channel_id = await self._resolve_channel()
        if not self._channel_id:
            logger.error(
                "Open WebUI: channel %r not found or not accessible",
                self.channel_name,
            )
            self._set_fatal_error(
                "channel_not_found",
                f"Channel {self.channel_name!r} not found. "
                f"Make sure the bot user is a member.",
                retryable=True,
            )
            await self._http.close()
            self._http = None
            return False

        # ── Fetch initial messages to set last_seen ──
        try:
            msgs = await self._get(
                f"/channels/{self._channel_id}/messages?limit=5"
            )
            if isinstance(msgs, list) and msgs:
                # Set last_seen to the most recent message so we don't
                # reprocess old messages on startup.
                # We pick the highest ID or latest created_at.
                latest = max(
                    (OWMessage(m) for m in msgs),
                    key=lambda m: m.created_at or 0,
                    default=None,
                )
                if latest:
                    self._last_seen = latest.id
                    logger.info(
                        "Open WebUI: caught up (last_seen=%s)", self._last_seen
                    )
        except Exception as e:
            logger.warning("Open WebUI: initial fetch failed: %s", e)

        # ── Start polling loop ──
        self._poll_task = asyncio.create_task(self._poll_loop())

        self._mark_connected()
        logger.info(
            "Open WebUI: connected to %s channel %s (%s)",
            self.base_url,
            self.channel_name,
            self._channel_id,
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._poll_task = None
        if self._http is not None:
            await self._http.close()
            self._http = None
        logger.info("Open WebUI: disconnected")

    # ── Channel resolution ──

    async def _resolve_channel(self) -> Optional[str]:
        """Find the channel ID by name or accept a direct ID."""
        name = self.channel_name.strip()

        # Try as a direct channel ID first (looks like "ch_...")
        if name.startswith("ch_"):
            channel = await self._get(f"/channels/{name}")
            if isinstance(channel, dict) and channel.get("id"):
                return channel["id"]

        # List channels and match by name
        channels = await self._get("/channels/")
        if not isinstance(channels, list):
            return None

        # Normalise: strip leading # for matching
        search = name.lstrip("#").lower()
        for ch in channels:
            ch_name = str(ch.get("name", "")).strip().lower()
            ch_id = str(ch.get("id", ""))
            if ch_name == search or ch_id == name:
                return ch_id

        # Allow joining by exact channel name if we didn't find it
        # (some Open WebUI versions create channels differently)
        logger.warning(
            "Open WebUI: channel %r not found among %d channels",
            name,
            len(channels),
        )
        return None

    # ── Polling loop ──

    async def _poll_loop(self) -> None:
        """Poll the channel for new messages."""
        while self._is_connected:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Open WebUI: poll error: %s", e, exc_info=True)
            await asyncio.sleep(self.poll_interval)

    async def _poll_once(self) -> None:
        """Fetch messages and process new ones."""
        msgs = await self._get(
            f"/channels/{self._channel_id}/messages?limit=10"
        )
        if not isinstance(msgs, list):
            return

        # Process in chronological order
        for raw in msgs:
            msg = OWMessage(raw)

            # Skip messages we've already seen
            if self._last_seen and msg.id <= self._last_seen:
                continue

            self._last_seen = msg.id

            # Skip bot's own messages
            if self._bot_user_id and msg.user_id == self._bot_user_id:
                continue

            # Skip empty or whitespace-only content
            content = msg.content.strip()
            if not content:
                continue

            # ── Route message ──
            logger.info(
                "Open WebUI: new message from %s: %s",
                msg.user_name,
                content[:80],
            )

            # Check if it's an approval response
            if self._is_approval_response(content, msg.user_id):
                await self._handle_approval_response(content, msg, msg.user_id)
            else:
                # Process as a normal message
                await self._process_message(content, msg, msg.user_id)

    # ── Approval helpers ──

    APPROVE_WORDS = frozenset(
        {"approve", "approved", "yes", "allow", "once"}
    )
    DENY_WORDS = frozenset({"deny", "denied", "no", "reject"})

    def _is_approval_response(self, text: str, user_id: str) -> bool:
        """Heuristic: is this user responding to an approval prompt?"""
        cleaned = text.strip().lower().rstrip(".!").strip()
        if cleaned not in self.APPROVE_WORDS and cleaned not in self.DENY_WORDS:
            return False
        # We'll check if there's a pending approval session key.
        # The BasePlatformAdapter tracks these via register_gateway_notify.
        try:
            from tools.approval import has_pending_approval
            return has_pending_approval(self._session_key_for_user(user_id))
        except Exception:
            return False

    def _session_key_for_user(self, user_id: str) -> str:
        """Derive a stable session/approval key for a user."""
        return f"owui:{self._channel_id}:{user_id}"

    async def _handle_approval_response(
        self, text: str, msg: OWMessage, user_id: str
    ) -> None:
        """Resolve a pending approval."""
        cleaned = text.strip().lower().rstrip(".!").strip()
        if cleaned in self.DENY_WORDS:
            choice = "deny"
        elif cleaned in {"always"}:
            choice = "always"
        elif cleaned in {"session"}:
            choice = "session"
        else:
            choice = "once"

        session_key = self._session_key_for_user(user_id)
        try:
            from tools.approval import resolve_gateway_approval
            resolved = resolve_gateway_approval(session_key, choice)
            if resolved > 0:
                await self.post(channel_id=self._channel_id, text=f"✅ **{choice}**")
            else:
                await self.post(
                    channel_id=self._channel_id,
                    text="⚠️ No pending approval found for that session.",
                )
        except Exception as e:
            logger.error("Approval resolution error: %s", e)
            await self.post(
                channel_id=self._channel_id,
                text=f"⚠️ Approval error: {e}",
            )

    # ── Message processing ──

    async def _process_message(
        self, text: str, msg: OWMessage, user_id: str
    ) -> None:
        """Send a message to the Hermes agent and post the response."""
        try:
            from gateway.run import (
                _resolve_runtime_agent_kwargs,
                _resolve_gateway_model,
                _load_gateway_config,
                GatewayRunner,
            )
            from hermes_cli.tools_config import _get_platform_tools
            from tools.approval import (
                register_gateway_notify,
                unregister_gateway_notify,
                set_current_session_key,
                reset_current_session_key,
            )
            from gateway.session_context import (
                set_session_vars,
                clear_session_vars,
            )

            session_key = self._session_key_for_user(user_id)

            # Build agent
            runtime_kwargs = _resolve_runtime_agent_kwargs()
            reasoning_config = GatewayRunner._load_reasoning_config()
            model = _resolve_gateway_model()
            user_config = _load_gateway_config()
            enabled_toolsets = sorted(
                _get_platform_tools(user_config, "openwebui")
            )
            fallback_model = GatewayRunner._load_fallback_model()

            # We run the agent in a thread executor so approvals block
            # on an Event that we can resolve from the poll loop.
            from run_agent import AIAgent

            agent = AIAgent(
                model=model,
                **runtime_kwargs,
                max_iterations=90,
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=enabled_toolsets,
                platform="openwebui",
                session_id=f"owui-{uuid.uuid4().hex[:12]}",
                fallback_model=fallback_model,
                reasoning_config=reasoning_config,
                gateway_session_key=session_key,
            )

            # Register approval notify so the agent can surface
            # approval requests
            async def _on_approval(data: dict):
                prompt = data.get("message", data.get("preview", "Approve?"))
                await self.post(
                    channel_id=self._channel_id,
                    text=f"⚠️ **Approval Required**\n```\n{prompt}\n```\nReply with **`approve`** or **`deny`**",
                )

            # We need a synchronous callback for the agent's thread
            def _approval_cb(data: dict):
                """Called from the agent thread when approval is needed."""
                prompt = data.get("message", data.get("preview", "Approve?"))
                # Schedule the post on the event loop
                asyncio.run_coroutine_threadsafe(
                    self.post(
                        channel_id=self._channel_id,
                        text=f"⚠️ **Approval Required**\n```\n{prompt[:500]}\n```\nReply with **`approve`** or **`deny`**",
                    ),
                    self._event_loop,
                )

            # Run in executor thread
            loop = asyncio.get_running_loop()
            self._event_loop = loop

            def _run():
                from gateway.session_context import set_session_vars, clear_session_vars

                approval_token = None
                session_tokens = []
                try:
                    approval_token = set_current_session_key(session_key)
                    session_tokens = set_session_vars(
                        platform="openwebui",
                        session_key=session_key,
                    )
                    register_gateway_notify(session_key, _approval_cb)
                    result = agent.run_conversation(
                        user_message=text,
                        task_id=session_key,
                    )
                finally:
                    try:
                        unregister_gateway_notify(session_key)
                    except Exception:
                        pass
                    finally:
                        if approval_token is not None:
                            try:
                                reset_current_session_key(approval_token)
                            except Exception:
                                pass
                        if session_tokens:
                            try:
                                clear_session_vars(session_tokens)
                            except Exception:
                                pass
                return result

            result = await loop.run_in_executor(None, _run)
            response = result.get("final_response", "") if isinstance(result, dict) else ""
            if not response:
                response = "(Hermes returned no response)"

            await self.post(channel_id=self._channel_id, text=response)

        except Exception as e:
            logger.error(
                "Open WebUI: processing error: %s", e, exc_info=True
            )
            await self.post(
                channel_id=self._channel_id,
                text=f"⚠️ Error: {e}",
            )

    # ── Send (for send_message tool and cron delivery) ──

    async def post(
        self,
        channel_id: str,
        text: str,
    ) -> None:
        """Post a message to a channel."""
        if not self._http:
            return
        await self._post(
            f"/channels/{channel_id}/messages/post",
            {"content": text, "data": {"files": []}},
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message (implements BasePlatformAdapter interface)."""
        target = chat_id or self._channel_id
        if not target:
            return SendResult(
                success=False,
                error="No channel configured",
            )
        try:
            body: Dict[str, Any] = {
                "content": content,
                "data": {"files": []},
            }
            if reply_to:
                body["reply_to_id"] = reply_to

            await self._post(
                f"/channels/{target}/messages/post", body
            )
            return SendResult(success=True)
        except Exception as e:
            logger.error("Open WebUI send error: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str) -> None:
        """Typing indicator — not supported by Open WebUI Channels API."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return channel info."""
        if not self._http:
            return {"name": "Open WebUI", "type": "channel"}
        channel = await self._get(f"/channels/{chat_id}")
        if isinstance(channel, dict):
            return {
                "name": channel.get("name", "Open WebUI"),
                "type": "channel",
                "chat_id": chat_id,
            }
        return {"name": "Open WebUI", "type": "channel", "chat_id": chat_id}


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """aiohttp is required — always available in the gateway."""
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False


def validate_config(config: dict) -> Optional[str]:
    """Validate required config fields."""
    extra = config.get("extra", {}) or {}
    missing = []
    for key, env_var in [
        ("url", "OPENWEBUI_URL"),
        ("api_key", "OPENWEBUI_API_KEY"),
        ("channel", "OPENWEBUI_CHANNEL_NAME"),
    ]:
        if not extra.get(key) and not os.getenv(env_var):
            missing.append(env_var)
    if missing:
        return f"Missing required env vars: {', '.join(missing)}"
    return None


def is_connected(adapter) -> bool:
    """Return True if the adapter is connected."""
    return adapter._is_connected if hasattr(adapter, "_is_connected") else False


async def interactive_setup(ctx) -> Optional[dict]:
    """CLI wizard stub — collects env vars via hermes gateway setup."""
    from hermes_cli.setup_prompts import prompt_text, prompt_optional

    url = await prompt_text("Open WebUI URL", "http://localhost:3000")
    api_key = await prompt_text("API key", password=True)
    channel = await prompt_text("Channel name", "#hermes")
    poll_interval = await prompt_optional("Poll interval (seconds)", "3")

    result = {
        "url": url,
        "api_key": api_key,
        "channel": channel,
    }
    if poll_interval:
        result["poll_interval"] = int(poll_interval)
    return result


def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra from env vars before adapter init."""
    url = os.getenv("OPENWEBUI_URL")
    key = os.getenv("OPENWEBUI_API_KEY")
    channel = os.getenv("OPENWEBUI_CHANNEL_NAME")
    if not url and not key and not channel:
        return None
    extra = {}
    if url:
        extra["url"] = url
    if key:
        extra["api_key"] = key
    if channel:
        extra["channel"] = channel
        extra["home_channel"] = channel
    return extra


async def _standalone_send(config: dict, chat_id: str, text: str, reply_to: str = None) -> dict:
    """
    Out-of-process cron delivery.  Called when cron runs separately
    from the gateway and needs to POST a message.
    """
    extra = config.get("extra", {}) or {}
    base_url = (extra.get("url") or "").rstrip("/")
    api_key = extra.get("api_key") or ""
    target = chat_id or extra.get("channel", "")

    if not base_url or not api_key or not target:
        return {"success": False, "error": "Missing config"}

    # Resolve channel name to ID
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        # List channels
        async with session.get(
            f"{base_url}/api/channels/", headers=headers
        ) as r:
            if r.status >= 400:
                return {"success": False, "error": await r.text()}
            channels = await r.json()

        ch_id = None
        search = target.lstrip("#").lower()
        if isinstance(channels, list):
            for ch in channels:
                if str(ch.get("name", "")).strip().lower() == search:
                    ch_id = ch.get("id")
                    break

        if not ch_id:
            return {"success": False, "error": f"Channel {target!r} not found"}

        # Post message
        async with session.post(
            f"{base_url}/api/channels/{ch_id}/messages/post",
            headers=headers,
            json={"content": text, "data": {"files": []}},
        ) as r:
            if r.status >= 400:
                return {"success": False, "error": await r.text()}
            return {"success": True}


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx):
    """Register the Open WebUI platform with the Hermes plugin system."""
    ctx.register_platform(
        name="openwebui",
        label="Open WebUI",
        adapter_factory=lambda cfg: OpenWebUIAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[
            "OPENWEBUI_URL",
            "OPENWEBUI_API_KEY",
            "OPENWEBUI_CHANNEL_NAME",
        ],
        install_hint="No extra packages needed (aiohttp shipped with gateway)",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="OPENWEBUI_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        emoji="🌐",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Open WebUI Channels. "
            "Open WebUI supports markdown formatting in messages. "
            "Users can send you messages in the channel and you respond "
            "in the same channel. Each message is independent — users "
            "can interject with new messages while you're processing. "
            "For approvals, describe the operation and tell the user "
            "to reply with 'approve' or 'deny'."
        ),
    )
