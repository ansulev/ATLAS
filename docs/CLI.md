# ATLAS CLI Guide

The ATLAS CLI launches the inference stack and drops you into an interactive
coding session. The canonical chat client is the native Bubbletea TUI
(`atlas tui`, introduced in PC-062). Plain `atlas` in an interactive
terminal launches the same TUI; pipe mode falls through to the built-in
`/solve` REPL.

---

## Launching

```bash
cd /path/to/your/project
atlas              # interactive: launches the TUI
atlas tui          # explicit form
echo "fix bug" | atlas   # pipe mode: routes through /solve
```

The top-level `atlas` binary also dispatches to non-TUI subcommands:

| Subcommand | Purpose |
|---|---|
| `atlas init` | First-run wizard: probes hardware, picks a model, writes `.env` + `secrets/api-keys.json`. |
| `atlas tier` | Hardware probe + tier classification (NVIDIA / AMD / Apple Silicon detection). |
| `atlas doctor` | Install diagnostic. GPU runtime, container health, endpoint reachability. |
| `atlas model list \| install \| verify \| remove` | Model registry operations. |
| `atlas lens check \| build` | Geometric Lens compat probe + per-model training (PC-057 / PC-058 — see below). |

`atlas` does the right thing automatically:

1. **Locates the `atlas-tui` binary** on `$PATH` or in `~/.local/bin`.
2. **Builds from source** in `tui/` if the binary is missing and Go
   1.24+ is available. (~10 s on first run.)
3. **Ensures atlas-proxy is running** via `_ensure_proxy()`. If the
   proxy's `/workspace` bind-mount doesn't already cover your current
   directory, the wrapper force-recreates the proxy container with the
   correct mount (~5 s) so tool calls can read and write your files.
4. **Execs the TUI** with `--proxy http://localhost:8090` and a debug
   log path under `~/.cache/atlas-tui/debug.log`.

```bash
atlas tui                                # default proxy at localhost:8090
atlas tui --proxy http://other-host:8090 # remote proxy
atlas tui --log /tmp/atlas-tui.log       # custom debug-log path
ATLAS_TUI_LOG=off atlas tui              # disable debug logging
```

If the binary is missing **and** Go is unavailable, the launcher prints
install instructions and exits.

---

## Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ Header                                                           │
│   ATLAS TUI · status · cwd · permission mode                     │
├──────────────────────────────────────┬───────────────────────────┤
│ Pipeline                             │                           │
│   live stage table from /events      │  Files                    │
├──────────────────────────────────────┤   workspace tree (depth 2)│
│ Chat                                 │   modified files marked   │
│   user + agent messages              │                           │
│   tool calls and results             │                           │
│   live LLM token stream              │                           │
├──────────────────────────────────────┤                           │
│ Events                               │                           │
│   raw typed-envelope log             │                           │
├──────────────────────────────────────┴───────────────────────────┤
│ Stats   stage · turn · ctx % · session · tools · events          │
├──────────────────────────────────────────────────────────────────┤
│ Message   chat (default) · ! bash · / command · ? help           │
└──────────────────────────────────────────────────────────────────┘
```

The **Files** sidebar appears when the terminal is ≥90 columns wide;
below that, the remaining panes stack vertically. **Pipeline**,
**Events**, and **Files** can each be hidden with `/hide <pane>`. See
[Panes](#panes) for what each region renders in detail.

---

## Input modes

The message box has three modes, distinguished by border color and the
hint row above it:

| First char | Mode | Border | Behavior |
|---|---|---|---|
| _(none)_ | chat | cyan | Sent to `/v1/agent` as a normal message |
| `!` | bash | red | Run as `bash -lc <cmd>` in the working dir; output appears as a system row |
| `/` | command | purple | Slash command; dropdown row above input shows matching commands |

Switching modes is just typing the trigger character — the border flips
and the hint row appears immediately. Backspace past the trigger char
to return to chat mode.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Send message / run bash command / fire slash command |
| `Shift+Enter` | Insert a newline (multi-line input) |
| `Ctrl+L` | Clear chat history |
| `Ctrl+T` | Cycle permission mode (default → accept-edits → yolo) |
| `Ctrl+R` | Re-send the last message |
| `Ctrl+C` | First press cancels the in-flight turn; second press exits |
| `Ctrl+D` | Exit immediately |
| `PgUp` / `PgDn` | Scroll chat by 10 rows |
| `Mouse wheel` | Scroll chat by 3 rows |
| `Ctrl+Home` | Jump to top of chat |
| `Ctrl+End` | Jump to bottom (resume auto-follow) |

Bracketed paste is enabled by default — pasted code arrives as a single
input event, so newlines in pasted text don't trigger a premature send.

### Copying text from the TUI

Mouse capture is on by default. Drag-highlight inside any pane (chat,
events, pipeline, files); on release, the highlighted text is auto-copied
to your clipboard and a transient toast (`✓ copied N chars from <pane>`)
appears in the header for ~2.5s. OSC52 fallback covers SSH sessions. No
chat row gets pushed for the copy — it's pure overlay UX.

If your terminal handles selection itself, you can also:

1. **Hold Shift (Linux/Windows) or Option (macOS)** while dragging.
2. **`/mouse off`** to disable capture for the rest of the session;
   wheel-scroll stops working but native terminal select returns.
   `/mouse on` re-enables.

For programmatic copy of recent chat output use `/copy [N]` (defaults to
the last message; pass an integer for the last N messages).

---

## Slash commands

| Command | Description |
|---|---|
| `/help` | Show in-TUI help with the full keymap and command list |
| `/add <path>` | Add a file to the agent's working context (path-only — agent reads on demand) |
| `/drop <path>` | Remove a file from the working context |
| `/context` | List files currently in context |
| `/diff [path]` | Show `git diff` (optionally for a specific path) |
| `/commit [msg]` | Stage all changes and create a commit (default msg if omitted) |
| `/undo` | `git reset --soft HEAD~1` — revert the last commit, keep changes |
| `/run <cmd>` | Run a shell command in the working dir; output appears in chat |
| `/clear` | Clear chat history (session token counter is preserved) |
| `/compact` | Ask the agent to summarize the conversation in 3-4 sentences |
| `/hide <pane>` | Hide a pane: `files`, `pipeline`, `events`, or `all` |
| `/show <pane>` | Show a pane (or `all`) |
| `/mouse on\|off` | Toggle mouse capture (off lets you copy text) |
| `/copy [N]` | Copy the last N chat messages (default 1) to clipboard via OSC52 |
| `/yank [N]` | Alias for `/copy` |
| `/quit` | Exit (same as `Ctrl+D`) |

The `/add /drop /context` set is TUI-side state — file paths are
appended to outgoing messages as a hint
(`[atlas-tui context: foo.go, bar.go]`) so the agent can `read_file`
them on demand. No file content is sent eagerly.

---

## Panes

### Files

Workspace tree to depth 2. Skips noisy directories (`.git`,
`node_modules`, `__pycache__`, `.venv`, `venv`, `dist`, `build`,
`target`, `.next`, `.nuxt`, `.idea`, `.vscode`, `.cache`,
`.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `__MACOSX`). Capped at
500 entries — overflow renders as `(+N more)`. Files modified by the
agent during the session are highlighted bold orange with a `●` prefix;
folders are bold cyan with `▸`. Re-scans every 4 s and immediately
after every `write_file`/`edit_file`/`delete_file` tool result.

### Pipeline

Live stage table fed by atlas-proxy's `/events` typed-envelope stream.
Each stage row shows an icon (⚙ running, ✓ done, ✗ failed), name,
status, duration, and a one-line detail. Stage names are emitted by the
proxy:

- `agent` — the whole `/v1/agent` turn
- `llm` — each LLM call (per turn)
- `tool` — each tool invocation
- `v3` — overall V3 pipeline (only when V3 fires for a write/edit)
- `v3:<phase>` — V3 sub-phases. `v3:plan` fires once per turn before
  the agent loop (see plan-mode rows in [Chat](#chat) below).
  Write/edit-triggered V3 adds `probe`, `plansearch`, `divsampling`,
  `sandbox_test`, `s_star`, etc.

### Chat

User and agent messages, tool calls and results, and live LLM token
streaming. Visual hierarchy:

- **Bright** (outputs the user cares about): user messages (`you`),
  finished assistant text (`agent`), executed tool calls (`→ tool`)
  and their results (`✓ tool` / `✗ tool` with elapsed time).
- **Dim grey italic** (machine internals): turn separators
  (`── turn N · ctx=K msgs ──`), LLM-call rows (`· llm · …`), V3
  internal LLM rows (`· v3 · …`, violet tint), planner rows
  (`plan` meta — see below), and other system metadata (mode
  changes, errors, V3 stage progress).

Plan-mode rows (when the planner ran for this turn — see
[ARCHITECTURE.md § Plan Mode](ARCHITECTURE.md#plan-mode-per-turn-pre-flight)
for mechanics):

- `plan` rows from `v3_plan` events — planner progress
  (`generating 3 candidate plans`, `candidate 1/3 (temp=0.3)`,
  `candidate 1 score=0.80`, `plan 1 won (score=0.80)`).
- Multi-line `plan_loaded` row — the full step list with glyphs
  (☐ unsatisfied, ✓ satisfied, ⚐ verify-step). A revision appends a
  new row tagged `plan rev N` and replaces the internal plan state,
  so subsequent `plan_adherence` rows count against the revised steps.
- `plan` adherence one-liners — `✓ s2 satisfied · edit_file (1/3)`
  fires when a tool call matches an unsatisfied step. Off-plan
  calls are silent (they only update internal state).
- `Plan revising (rev 1): <reason>` — the agent went off-plan past
  the threshold; the next `plan_loaded` replaces the plan.

During an LLM call the dim row fills in token-by-token. For
`write_file` calls, partial JSON is unescaped on the fly so you see
actual indented HTML/code being generated. Display caps at the last 80
lines so very long generations don't churn the renderer.

A "thinking…" spinner with rotating verbs (Pondering, Cogitating,
Brewing, Conjuring, Synthesizing, Mulling, …) sits at the bottom of the
chat box during a turn. Word changes every ~3 s.

### Events

Compact log of the raw `/events` envelope stream — one line per event
with timestamp, type, stage, and a short summary. Useful for debugging
the proxy↔TUI protocol.

### Stats

One-line strip below the events pane:

- **Active stage** (`● llm`, `● v3:probe`)
- **Turn counter** (`turn:1`)
- **Context utilization** (`ctx:8.5k/32k (26%)`) — color-coded ≥50% (orange),
  ≥80% (red). Updates live during decode.
- **Session-wide token count** (`session:9.5k`)
- **Tool counters** (`tools:3✓/0✗`)
- **Event counter** (`events:42`)

---

## Permission modes

Cycle with `Ctrl+T`:

| Mode | Behavior |
|---|---|
| `default` | Read tools and surgical edits (`edit_file`, `ast_edit`) auto-allow; `write_file`, `delete_file`, `run_command`, and `stop_background` require user approval |
| `accept-edits` | As above + `write_file` auto-allow; `delete_file`, `run_command`, and `stop_background` still confirm |
| `yolo` | Auto-allow everything |

The exact gate is `Destructive: true` on the tool definition in
`proxy/tools.go`; `accept-edits` additionally auto-approves
`write_file` and `edit_file` (the latter is already non-destructive).

The current mode shows in the header. Approval prompts appear in chat
as `permission_request` rows.

---

## Cancelling a turn

Each `/v1/agent` POST is tagged with a `session_id`. On `Ctrl+C` the TUI
cancels the local `context.Context` (closing the TCP connection) **and**
POSTs `/cancel` with the same `session_id` as defense-in-depth, in case
a reverse proxy buffers the disconnect. The proxy's agent loop watches
`ctx.Done()` and exits at the next turn boundary. The cancel propagates
through to llama-server (PC-036).

---

## Debug log

The TUI mirrors every event it receives to an append-only log so you
can review what happened after the fact (alt-screen makes copying out
of the live view impractical).

```bash
tail -f ~/.cache/atlas-tui/debug.log
```

Each line is a JSON-tagged record:
`HH:MM:SS.mmm category:subject {fields}`. Categories are `session`,
`user` (input events), `turn` (turn lifecycle), `chat` (every
chatStreamMsg type except `llm_token` to keep the file readable), `event`
(every typed envelope), and `slash` (slash command dispatch + result).

Override the path via `--log <path>` or `$ATLAS_TUI_LOG`. Set
`ATLAS_TUI_LOG=off` to disable.

---

## Workspace alignment

The proxy executes file operations against `/workspace` inside its
container, which is bind-mounted to a directory on host disk (set in
`docker-compose.yml`). For tool calls to land in your project, that
mount has to point at the directory you're working in.

`atlas tui` aligns this automatically:

1. On startup, `_ensure_proxy()` checks whether the proxy's existing
   `/workspace` mount covers `os.getcwd()`.
2. If not, it force-recreates the `atlas-proxy` container with
   `ATLAS_PROJECT_DIR=$(pwd)` so the bind mount points at your cwd.
   This takes ~5 s.
3. The proxy itself overrides any `working_dir` field in `/v1/agent`
   requests with the container-internal `/workspace` path, so the
   agent's `read_file`/`write_file` calls always resolve correctly.

If you write code from one shell and `atlas tui` is running in another
that's pointing at a different directory, restart the TUI in the right
cwd to re-align.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ATLAS_PROXY_URL` | `http://localhost:8090` | Default `--proxy` value |
| `ATLAS_TUI_LOG` | `~/.cache/atlas-tui/debug.log` | TUI debug log path; set `off` to disable |
| `ATLAS_TUI_STARTUP_NOTE` | _(unset)_ | Initial system message inserted at startup (used by the Python wrapper to surface workspace warnings) |
| `ATLAS_TUI_MOUSE` | `on` | Mouse capture at startup; `off` skips `WithMouseCellMotion` so native terminal select works without modifiers. Mid-session toggle via `/mouse on\|off`. Also exposed as the `--mouse` flag. |
| `GLAMOUR_STYLE` | `dark` | Markdown rendering style for assistant text |
| `ATLAS_AUTO_WORKSPACE` | `1` | Set `0` to disable auto-realign of the proxy's bind mount |

See [CONFIGURATION.md](CONFIGURATION.md) for the full set of variables
that affect the proxy and inference stack.

---

## Stack overview

The TUI is one of several services. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the full picture; the short version:

| Service | Port | Role |
|---|---|---|
| llama-server | 8080 | Local LLM inference (llama.cpp / Qwen3.5-9B-Q6_K) |
| atlas-proxy | 8090 | Agent loop, tool execution, V3 routing, SSE event broker |
| v3-service | 8070 | V3 pipeline (PlanSearch, DivSampling, build verification, repair) |
| geometric-lens | 8099 | C(x)/G(x) energy scoring |
| sandbox | 30820 | Isolated code execution for V3 verification |

`atlas tui` only needs `atlas-proxy` reachable; the proxy fans out to
the other services internally.

---

## atlas lens (PC-057 / PC-058)

Geometric Lens compat probe + per-model training. Lets you bring a non-default GGUF and either verify it'll score with the existing C(x) artifacts or train fresh ones for it.

### `atlas lens check`

Cheap pre-flight against the running llama-server. No training, no model download — just probes `/embedding` and `/props` to confirm the model is Lens-compatible.

```bash
atlas lens check                       # probe whatever llama-server has loaded
atlas lens check Qwen3.5-9B-Q6_K       # probe a registry entry by name
atlas lens check /path/to/model.gguf   # probe an arbitrary file
atlas lens check --json                # machine-readable for scripts / CI
```

Verdict + exit code:

| Verdict | Exit | Meaning |
|---|---|---|
| `compat` | 0 | Artifacts exist and accept this model's embedding dim. Ready to score. |
| `needs-build` | 1 | Model loads but no cost_field.pt at the right dim. Run `atlas lens build`. |
| `incompatible` | 2 | Can't probe — llama-server unreachable, `/embedding` silent, etc. |

Reports the model's embedding dim, layer count, PC-202 hidden-states-patch status, the artifact dir it checked, and the artifact's own input dim. JSON mode produces a stable shape (`verdict`, `reason`, `probe.*`, `artifact_dir`, `artifact_dim`, `matched_model`, `exit_code`).

### `atlas lens build`

Trains a fresh `cost_field.pt` for whichever model llama-server has loaded. Wraps `geometric-lens/geometric_lens/training.py:train_cost_field` (contrastive ranking loss, ~200 epochs, ~30 s on CPU for a few-hundred-sample set).

```bash
atlas lens build --samples path/to/labeled.json    # required: labeled training data
atlas lens build --samples ... --epochs 400        # tune training
atlas lens build --samples ... --force             # retrain even if compat artifact exists
atlas lens build --samples ... --dry-run           # extract embeddings, skip training + save
```

**Sample format** — JSON array (or JSONL) of `{"text": "...", "label": 0|1}` where `label=1` means the snippet represents *passing* / correct code and `label=0` means *failing*. Pull the canonical training set (V3 ablation traces with pass/fail labels) from `huggingface.co/datasets/itigges22/ATLAS`.

Minimums: at least 50 samples with both classes present (build refuses below this — a too-small C(x) actively mis-ranks). Test AUC below 0.70 emits a warning suggesting more samples or epochs.

After a successful build:
1. `cost_field.pt` lands in the artifact dir (default `geometric-lens/geometric_lens/models/`, override with `--artifact-dir`).
2. Re-run `atlas lens check` — should now report `compat`.
3. Update the model's registry entry to `lens_status="supported"` so `atlas doctor` and `atlas tier` surface that the model has artifacts. (Registry write-back automation is PC-059 — for now this is a one-line manual edit to `atlas/cli/commands/model_registry.py`.)

### Prereqs

Both subcommands require a running `llama-server`. `atlas lens check` reuses the same URL resolution as the lens service (`ATLAS_LLAMA_URL` → `LLAMA_EMBED_URL` → `LLAMA_URL` → `http://localhost:8080`). The PC-202 hidden-states patch (baked into `inference/Dockerfile.v31` and `Dockerfile.rocm`) is required for G(x) metric-tensor training but not for C(x) — `check` reports its presence as informational.

---

## Troubleshooting

### TUI renders, but the file pane is empty

You're probably running from a directory the proxy's `/workspace` mount
doesn't cover. Check:

```bash
docker inspect atlas-atlas-proxy-1 --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}'
```

The output should match your `pwd`. If not, exit and restart `atlas tui`
from the right directory; the wrapper auto-realigns on launch.

### "atlas-tui binary not found and Go is not available"

Install Go 1.24+ from [https://go.dev/dl/](https://go.dev/dl/), or
build manually:

```bash
cd tui
go build -o ~/.local/bin/atlas-tui .
```

### Wheel scroll doesn't work in tmux

tmux intercepts mouse events. Either enable mouse passthrough in tmux
(`set -g mouse on`) or use `PgUp`/`PgDn` instead.

### V3 doesn't fire on small files

By design: V3 only fires for files that look like meaningful code. The
trigger rule (see `classifyFileTier` in `proxy/tools.go`):

- Config files by name (`package.json`, `tsconfig.json`, `Dockerfile`, …)
  → always T1 (direct write).
- Data extensions (`.json`, `.yaml`, `.toml`, `.csv`, `.xml`, `.env`),
  style files (`.css`, `.scss`, `.less`), prose (`.md`, `.txt`, `.rst`),
  and shell scripts (`.sh`, `.bash`) → always T1.
- Anything under 10 lines → T1 (nothing for V3 to meaningfully
  diversify on).
- ≥10 lines and either (a) `hasLogicIndicators` returns true (2+ matches
  across 9 pattern families — function/method, control flow, error
  handling, Flask/FastAPI, Express/Node, React state, validation,
  database, JSX) or (b) the extension is in the code/markup set
  (`.py`, `.go`, `.rs`, `.ts`, `.tsx`, `.js`, `.jsx`, `.c`, `.cpp`,
  `.cc`, `.h`, `.hpp`, `.java`, `.kt`, `.swift`, `.rb`, `.php`,
  `.vue`, `.svelte`, `.html`, `.htm`) → T2 (V3 pipeline).
- Unknown extensions → T1.

### "encoding prompt…" lingers for >30 s

Llama.cpp doesn't flush HTTP response headers until the first decoded
token, so "header time" = "prompt eval time". Long conversation
histories (8K+ tokens) can take ~60 s of prompt eval before the first
token arrives. As of May 2026 the proxy runs with no
`ResponseHeaderTimeout` so long V3 chains (which can spend several
minutes before the first SSE frame) complete instead of being killed
mid-flight. If "encoding prompt…" sits for many minutes, the prompt is
genuinely too big — `/compact` to summarize.

---

## Building a non-TUI client

`atlas-proxy`'s `/v1/agent`, `/events`, and `/cancel` endpoints are the
public client contract. Anything that speaks SSE can be a chat client.
See [API.md § Building a non-TUI client](API.md#building-a-non-tui-client) for the protocol and a minimal Python example. PC-063 tracks the full spec writeup.
