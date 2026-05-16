"""
title: Hermes Agent
author: Hannah / Nous Research
version: 2.0.0
required_open_webui_version: 0.3.0
description: >
  Open WebUI adapter for Hermes Agent with streaming, tool visualisation,
  slash commands, and approval handling. Connects to Hermes Agent's
  built-in API server.

  **Setup:**
  1. Hermes API server must be running (hermes gateway run → port 8642)
  2. Set valve(s) below with URL and optional API key
  3. Select "Hermes Agent" as your model in Open WebUI

  **Features:**
  - Streaming responses via /v1/runs SSE
  - Slash commands /help, /model, /reset, /approve, /deny etc.
  - Tool progress shown as clean markdown status lines
  - Interactive approval dialogs via confirmation popups
  - Session continuity across turns
"""

from pydantic import BaseModel, Field
import aiohttp
import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Optional

try:
    from fastapi.responses import StreamingResponse
except ImportError:
    StreamingResponse = None

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

AGENT_ID = "hermes-agent"
AGENT_NAME = "Hermes Agent"
APPROVAL_KEYWORDS = frozenset({"approve", "approved", "yes", "allow", "once"})
DENY_KEYWORDS = frozenset({"deny", "denied", "no", "reject"})


# ── Help text ──────────────────────────────────────────────────────────

HELP_TEXT = (
    "**Hermes Agent – Available Commands**\n\n"
    "Type any of these as your message:\n\n"
    "| Command | Description |\n"
    "|---------|-------------|\n"
    "| `/reset` or `/new` | Start a fresh session |\n"
    "| `/retry` | Resend last message |\n"
    "| `/undo` | Remove last exchange |\n"
    "| `/yolo` | Toggle approval bypass |\n"
    "| `/compress` | Manually compress context |\n"
    "| `/approve` | Approve pending operation |\n"
    "| `/deny` | Deny pending operation |\n"
    "| `/stop` | Stop background processes |\n"
    "| `/model` | Show/change model |\n"
    "| `/title` | Name the session |\n"
    "| `/status` | Show session info |\n"
    "| `/usage` | Show token usage |\n"
    "| `/plugins` | List plugins |\n"
    "| `/skills` | Search/install skills |\n"
    "| `/cron` | Manage cron jobs |\n"
    "| `/profile` | Show active profile |\n"
    "| `/help` or `/commands` | This help |\n\n"
    "**Approvals**: Hermes may ask for confirmation before running "
    "dangerous commands. A dialog will pop up — click OK/Cancel, "
    "or type `approve` / `deny` in your next message.\n\n"
    "**Commands during a run**: Tools stream their progress inline. "
    "Wait for the current response to finish before sending your next command."
)


# ── Pipe –  v2 ─────────────────────────────────────────────────────────

class Pipe:
    """
    Open WebUI Pipe — Hermes Agent via /v1/runs SSE.

    Yields plain markdown for tool progress (not HTML <details> blocks,
    which Open WebUI renders as raw text).  Approval prompts use
    __event_call__ for an interactive dialog, falling back to a
    text prompt.
    """

    class Valves(BaseModel):
        HERMES_API_URL: str = Field(
            default="http://127.0.0.1:8642/v1",
            description="Hermes Agent API server base URL (e.g. http://host:8642/v1)",
        )
        HERMES_API_KEY: str = Field(
            default="",
            description="API key — must match API_SERVER_KEY in Hermes .env. "
            "Leave empty for local-only access.",
        )
        STREAMING: bool = Field(
            default=True, description="Enable streaming responses",
        )
        APPROVAL_TIMEOUT: int = Field(
            default=300,
            description="Max seconds to wait for user approval response",
        )
        pass

    class UserValves(BaseModel):
        pass

    def __init__(self):
        self.type = "pipe"
        self.id = AGENT_ID
        self.name = AGENT_NAME

        self.valves = self.Valves(
            **{
                "HERMES_API_URL": os.getenv("HERMES_API_URL", "http://127.0.0.1:8642/v1"),
                "HERMES_API_KEY": os.getenv("HERMES_API_KEY", ""),
                "STREAMING": os.getenv("HERMES_STREAMING", "true").lower() in ("true", "1"),
                "APPROVAL_TIMEOUT": int(os.getenv("HERMES_APPROVAL_TIMEOUT", "300")),
            }
        )
        self.user_valves = self.UserValves()

        # Per-user state — lives in memory for the lifetime of the
        # gateway process.  Survives the invocations of pipe().
        self._session_map: dict[str, str] = {}
        self._pending_approvals: dict[str, str] = {}       # user_id → run_id
        self._pending_prompts: dict[str, str] = {}         # user_id → approval prompt text

    # ── Public API ─────────────────────────────────────────────────────

    async def pipes(self) -> list[dict]:
        return [{"id": AGENT_ID, "name": AGENT_NAME}]

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
        __event_call__: Optional[Callable] = None,
    ) -> Any:
        user_id = __user__.get("id", "anonymous") if __user__ else "anonymous"
        messages = body.get("messages", [])
        stream = body.get("stream", self.valves.STREAMING)

        if not messages:
            return {"error": {"detail": "No messages"}}

        last_msg = messages[-1]
        text = self._extract_text(last_msg.get("content", ""))

        # ── 1. Help ──
        if text.strip().lower() in ("/help", "/commands", "help"):
            return await self._respond(HELP_TEXT, stream)

        # ── 2. Pending approval response ──
        if user_id in self._pending_approvals:
            return await self._resolve_approval(text, user_id, stream, __event_emitter__)

        # ── 3. Everything else — run Hermes ──
        history = self._build_history(messages[:-1])
        session_id = self._session_id(user_id, messages)

        if stream:
            return StreamingResponse(
                self._stream(
                    text, history, session_id, user_id,
                    __event_emitter__, __event_call__,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            result = await self._blocking(text, history, session_id, user_id)
            return result

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Normalise multimodal OpenAI content to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    t = str(item.get("type", "")).strip().lower()
                    if t in ("text", "input_text", "output_text"):
                        parts.append(str(item.get("text", "")))
                    elif t in ("image_url", "image"):
                        parts.append("[Image]")
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(p.strip() for p in parts if p.strip())
        return str(content)

    @staticmethod
    def _build_history(messages: list[dict]) -> list[dict]:
        hist = []
        for msg in messages:
            content = Pipe._extract_text(msg.get("content", ""))
            if content:
                hist.append({"role": msg.get("role", "user"), "content": content})
        return hist

    def _session_id(self, user_id: str, messages: list[dict]) -> str:
        cached = self._session_map.get(user_id)
        if cached:
            return cached
        for msg in messages:
            if msg.get("role") == "user":
                content = self._extract_text(msg.get("content", ""))
                if content.strip():
                    digest = hashlib.sha256(content.strip()[:200].encode()).hexdigest()[:16]
                    sid = f"owui-{digest}"
                    self._session_map[user_id] = sid
                    return sid
        sid = f"owui-{user_id}-{uuid.uuid4().hex[:8]}"
        self._session_map[user_id] = sid
        return sid

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.valves.HERMES_API_KEY:
            h["Authorization"] = f"Bearer {self.valves.HERMES_API_KEY}"
        return h

    async def _respond(self, text: str, stream: bool) -> Any:
        """Return a single text response (helper shortcut)."""
        if stream:
            return StreamingResponse(
                self._chunked_text(text),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
            }]
        }

    async def _chunked_text(self, text: str) -> AsyncGenerator[str, None]:
        """Simple streaming generator for a single text block."""
        yield self._sse_chunk({"role": "assistant"}, finish=None)
        for i in range(0, len(text), 80):
            yield self._sse_chunk({"content": text[i:i+80]}, finish=None)
        yield self._sse_chunk({}, finish="stop")

    # ── Approval ───────────────────────────────────────────────────────

    async def _resolve_approval(
        self, user_input: str, user_id: str,
        stream: bool, __event_emitter__: Optional[Callable],
    ) -> Any:
        run_id = self._pending_approvals.pop(user_id, None)
        prompt = self._pending_prompts.pop(user_id, "")
        if not run_id:
            return await self._respond("No pending approval found.", stream)

        text = user_input.strip().lower().rstrip(".!")
        if text in DENY_KEYWORDS:
            choice = "deny"
        elif text in {"always"}:
            choice = "always"
        elif text in {"session"}:
            choice = "session"
        else:
            choice = "once"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(30)) as s:
                async with s.post(
                    f"{self.valves.HERMES_API_URL}/runs/{run_id}/approval",
                    headers=self._headers(),
                    json={"choice": choice},
                ) as r:
                    if r.status >= 400:
                        err = await r.text()
                        msg = (
                            f"⚠️ **Approval failed** ({choice}): {err[:300]}\n\n"
                            f"Run `{run_id}` may have timed out."
                        )
                    else:
                        msg = (
                            f"✅ **{choice}** — the agent will continue. "
                            f"Wait for the next response…"
                        )

                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "notification",
                            "data": {
                                "type": "success" if choice != "deny" else "info",
                                "content": msg,
                            },
                        })
                    return await self._respond(msg, stream)
        except Exception as e:
            return await self._respond(f"⚠️ Error submitting approval: {e}", stream)

    # ── Blocking (non-streaming) ────────────────────────────────────────

    async def _blocking(
        self, message: str, history: list[dict],
        session_id: str, user_id: str,
    ) -> dict:
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": message})
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(600)) as s:
                async with s.post(
                    f"{self.valves.HERMES_API_URL}/chat/completions",
                    headers=self._headers(),
                    json={"messages": messages, "stream": False},
                ) as r:
                    if r.status >= 400:
                        err = await r.text()
                        content = f"⚠️ Hermes error ({r.status}): {err[:500]}"
                    else:
                        data = await r.json()
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                            or "(Hermes returned no response)"
                        )
                    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}
        except asyncio.TimeoutError:
            return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "⚠️ Hermes timed out."}}]}
        except Exception as e:
            return {"choices": [{"index": 0, "message": {"role": "assistant", "content": f"⚠️ Cannot reach Hermes: {e}"}}]}

    # ── Streaming via /v1/runs ─────────────────────────────────────────

    def _sse_chunk(self, delta: dict, finish: Optional[str]) -> str:
        """Build an OpenAI chat.completion.chunk SSE string."""
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": AGENT_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    async def _stream(
        self, message: str, history: list[dict], session_id: str,
        user_id: str,
        __event_emitter__: Optional[Callable],
        __event_call__: Optional[Callable],
    ) -> AsyncGenerator[str, None]:
        """
        Async generator — each yield is one SSE chunk.

        Uses /v1/runs for lifecycle events and approval handling.
        All tool/approval progress is rendered as **plain markdown**
        so Open WebUI displays it correctly (no raw <details> HTML).
        """

        # ── helpers ──
        def _text(delta_text: str) -> str:
            return self._sse_chunk({"content": delta_text}, None)

        def _done() -> str:
            return self._sse_chunk({}, "stop")

        def _tool_line(emoji: str, name: str, detail: str = "") -> str:
            d = f" — {detail}" if detail else ""
            return _text(f"> {emoji} **{name}**{d}  \n")

        # ── Start ──
        yield self._sse_chunk({"role": "assistant"}, None)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(600)) as session:

            # 1. Create run
            try:
                async with session.post(
                    f"{self.valves.HERMES_API_URL}/runs",
                    headers=self._headers(),
                    json={
                        "input": message,
                        "conversation_history": history,
                        "session_id": session_id,
                    },
                ) as r:
                    if r.status >= 400:
                        yield _text(f"⚠️ **Error starting run** ({r.status})")
                        yield _done()
                        return
                    run = await r.json()
                    run_id = run["run_id"]
            except aiohttp.ClientConnectorError:
                yield _text(f"⚠️ **Cannot connect** — is Hermes running?\n`{self.valves.HERMES_API_URL}`")
                yield _done()
                return

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Hermes is working…", "done": False},
                })

            # 2. Consume SSE event stream
            accumulated = ""
            async with session.get(
                f"{self.valves.HERMES_API_URL}/runs/{run_id}/events",
                headers=self._headers(),
            ) as sse:
                buf = ""

                async for raw in sse.content:
                    buf += raw.decode("utf-8")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data: "):
                            continue

                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        ev = event.get("event", "")

                        # ── Text delta ──
                        if ev == "message.delta":
                            d = event.get("delta", "")
                            if d:
                                accumulated += d
                                yield _text(d)

                        # ── Tool started ──
                        elif ev == "tool.started":
                            name = event.get("tool", "tool")
                            preview = event.get("preview", "")
                            yield _tool_line("🔧", name, preview[:200])

                        # ── Tool completed ──
                        elif ev == "tool.completed":
                            name = event.get("tool", "tool")
                            dur = event.get("duration", 0)
                            err = event.get("error", False)
                            emoji = "❌" if err else "✅"
                            yield _tool_line(emoji, name, f"({dur}s)")

                        # ── Reasoning ──
                        elif ev == "reasoning.available":
                            text = event.get("text", "")
                            if text:
                                yield _text(f"\n> 💭 *{text[:500].strip()}*\n")

                        # ── Approval request ──
                        elif ev == "approval.request":
                            self._pending_approvals[user_id] = run_id
                            prompt_msg = event.get(
                                "message",
                                event.get("preview", "Approve this action?"),
                            )
                            self._pending_prompts[user_id] = prompt_msg

                            # Try interactive dialog
                            if __event_call__:
                                try:
                                    ok = await asyncio.wait_for(
                                        __event_call__({
                                            "type": "confirmation",
                                            "data": {
                                                "title": "⚠️ Hermes Approval Required",
                                                "message": (
                                                    f"Hermes wants:\n\n"
                                                    f"```\n{prompt_msg[:1000]}\n```\n\n"
                                                    f"**Approve?**"
                                                ),
                                            },
                                        }),
                                        timeout=self.valves.APPROVAL_TIMEOUT,
                                    )
                                    choice = "once" if ok else "deny"
                                except asyncio.TimeoutError:
                                    yield _text(
                                        "\n\n⏰ **Approval timed out** — no response.\n"
                                    )
                                    del self._pending_approvals[user_id]
                                    del self._pending_prompts[user_id]
                                    choice = "deny"
                            else:
                                # Fallback: emit a text prompt
                                yield _text(
                                    f"\n\n⚠️ **Approval Required**\n\n"
                                    f"```\n{prompt_msg[:1000]}\n```\n\n"
                                    f"Reply with **`approve`** or **`deny`**.\n"
                                )
                                # We can't wait here without __event_call__,
                                # so close the stream.  Next user message
                                # will hit _resolve_approval.
                                yield _done()
                                return

                            # Submit the decision
                            try:
                                async with session.post(
                                    f"{self.valves.HERMES_API_URL}/runs/{run_id}/approval",
                                    headers=self._headers(),
                                    json={"choice": choice},
                                ) as ar:
                                    if ar.status >= 400:
                                        yield _text(f"\n⚠️ Approval submit failed ({ar.status}).\n")
                            except Exception as e:
                                yield _text(f"\n⚠️ Approval error: {e}\n")

                            self._pending_approvals.pop(user_id, None)
                            self._pending_prompts.pop(user_id, None)
                            yield _text(f"*({choice})*  \n")

                        # ── Run completed ──
                        elif ev == "run.completed":
                            output = event.get("output", "")
                            if output and output not in accumulated:
                                yield _text(output)
                            yield _done()
                            if __event_emitter__:
                                await __event_emitter__({
                                    "type": "status",
                                    "data": {"description": "Done", "done": True},
                                })
                            return

                        # ── Run failed ──
                        elif ev == "run.failed":
                            err = event.get("error", "Unknown error")
                            yield _text(f"\n⚠️ **Agent Error:** {err}")
                            yield _done()
                            return

                        # ── Run cancelled ──
                        elif ev == "run.cancelled":
                            yield _text("\n*Run cancelled.*\n")
                            yield _done()
                            return

                # SSE closed without a terminal event
                if accumulated:
                    yield _done()
                else:
                    yield _text("*(Hermes returned no response)*")
                    yield _done()

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": "Done", "done": True},
            })
