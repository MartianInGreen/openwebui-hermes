# openwebui-hermes

Two ways to connect Hermes Agent to Open WebUI:

## Hermes Plugin (Recommended)

A Hermes gateway platform adapter that watches an Open WebUI channel
and processes messages asynchronously through Hermes' agent loop.

**Architecture:** Hermes polls the Open WebUI Channels REST API,
picks up new human messages, runs them through the agent, and posts
responses back as the bot user. Each message is an independent turn
so you can interject mid-response.

**Requires:** Hermes Gateway running with API server enabled.

```
~/.hermes/plugins/openwebui/
├── plugin.yaml      — env var definitions & metadata
├── __init__.py      — exports register()
└── adapter.py       — the OpenWebUIAdapter class
```

### Setup

1. **Create a bot user** in Open WebUI and generate an API key
2. **Create a channel** (or use existing) and add the bot as a member
3. **Set env vars** or add to `~/.hermes/config.yaml`:

```bash
OPENWEBUI_URL=http://your-tailscale-address:3000
OPENWEBUI_API_KEY=sk-your-api-key-here
OPENWEBUI_CHANNEL_NAME=#hermes
```

4. **Restart the gateway**:

```bash
hermes gateway restart
```

The adapter auto-discovers the channel by name and begins polling.

### How it works

- Polls the channel every 3 seconds for new messages
- Skips the bot's own messages (no echo loops)
- Each message runs through a fresh Hermes agent session
- Approval prompts are posted to the channel — reply `approve`/`deny`
- Full toolset, slash commands, and model config work (same as Discord)

---

## Open WebUI Pipe Function (Alternative)

A Pipe Function that makes Hermes appear as a model in Open WebUI's
model selector dropdown. Less capable than the plugin (no interjection,
HTML tool rendering issues) but simpler to set up.

Requires the Hermes API server to be running on port 8642.

**Install:** Admin Panel → Workspace → Functions → Import
`pipe-function/hermes_openwebui_adapter.py`

---

## License

MIT — see [LICENSE](LICENSE)
