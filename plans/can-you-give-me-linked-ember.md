# Recipe: a Gemma agent REPL (llama.cpp `llama-server`), adapted from `agent/`

## Context
You want an agent REPL for Gemma, modeled on this repo's `mimo-agent`. The agent in `agent/` is small (about 590 lines, stdlib only) and mostly independent of the model: it uses a generic OpenAI-compatible streaming client, a tool loop and seven tools. Only a few lines tie it to MiMo. This recipe covers what to copy, what to change and how to check it works against `llama-server`. **This repo doesn't change.**

## 1. Serve Gemma with tool calling
```sh
llama-server -hf ggml-org/gemma-3-12b-it-GGUF --jinja -c 32768 --port 8080
```
- **`--jinja` is required.** Without it, llama.cpp ignores `tools` and never returns `tool_calls`. With it, llama.cpp renders the chat template from the GGUF. Gemma 3's template has no native tool tokens, so llama.cpp falls back to its generic JSON tool-call format. It also fills in the missing support for the system role and the tool role. If your Gemma version ships a template with native tool support, it uses that instead.
- Use 12B or 27B. Smaller Gemmas often answer in prose when they should emit a tool call.
- `-c` sets the context window. The loop never compacts, so `/reset` is how you recover when the conversation fills it.
- Sanity check: `curl localhost:8080/health` should return 200. `llama-server` serves `/health`, so the existing `healthy()` helper works as is.

## 2. Copy the package
Copy these files into a new project as package `gemma_agent/`:
- `client.py`, `tools.py`, `loop.py`, `prompt.py`, `__main__.py`, `__init__.py`
- `tests/fake_server.py`, `tests/test_tools.py`, `tests/test_loop.py`

Then change the `from agent import ...` lines to `from gemma_agent import ...`.

## 3. Files you keep unchanged
- **`client.py`**: a generic OpenAI SSE client. `llama-server` streams `tool_calls` deltas with `index`, `id`, `function.name` and argument fragments, which is exactly what it accumulates. `reasoning_content` stays empty for Gemma, which is fine.
- **`tools.py`**: has nothing model-specific.

## 4. `loop.py`: three edits
1. **`Agent.request()`**: remove `enable_thinking` and `chat_template_kwargs` (MiMo's thinking switch; Gemma's template doesn't have one). Optionally add sampling settings, e.g. `"temperature": 0.3`. A lower temperature helps a smaller model emit valid tool JSON.
2. **`thinking` parameter**: remove it from `__init__`, or keep it as a no-op. Your choice; removing it is cleaner.
3. **`SERVER_ERROR`**: remove the regex and its branch in `step()`. It matches a MiMo-server convention (`\n[server error: …]` appended to content). `llama-server` reports errors as HTTP errors or SSE `error` events, which `client.chat` already turns into `RuntimeError`.

Leave the rest alone: `run`, `run_tools`, `parse_args`, the stop-reason mapping, keeping assistant turns verbatim in history, and Ctrl-C handling.

## 5. `prompt.py`
Keep the prompt short. Optionally add one line for weaker tool-callers: "Call tools by emitting a tool call, never by describing it in prose."

## 6. `__main__.py`: remove the MiMo server management
- Remove `ensure_server`, `REPO`, the `--no-start` flag and the `mimo-server-ctl` call. Replace them with: `if not healthy(a.base_url): print("start llama-server --jinja first"); return 1`.
- `DEFAULT_BASE = "http://127.0.0.1:8080/v1"`, `--model` default `"gemma"` (`llama-server` ignores the name).
- Remove `--thinking` (or keep it if you kept the parameter in step 4).
- Rename the `mimo-agent` strings: prog name, banner, `[mimo-agent]` prefixes.
- Keep `interactive()`, `headless()`, `call_line`, `HELP` and `/reset`, `/exit`, `/help` unchanged.

## 7. Launcher
Run it with `python -m gemma_agent` from the project root. You can also copy `scripts/mimo-agent`, which puts the repo on `sys.path` without using `PYTHONPATH`, and change `runpy.run_module("agent"...)` to `"gemma_agent"`.

## 8. Optional: adjust for Gemma's context size
`tools.MAX_BYTES = 50_000` is roughly 12k tokens per tool result, which is a lot in a 32k context. Drop it to about 20_000, and drop `MAX_LINES` to match.

## Verification
- **Unit tests** (`python -m pytest gemma_agent/tests`) need some test changes first:
  - Remove the `enable_thinking` assertions in `test_loop.py` (lines ~49, 73, 137).
  - Remove the `[server error: …]` test (~90).
  - Remove the auto-start and launcher-symlink tests (~176–192).
  - `fake_server.py` and `test_tools.py` work unchanged.
- **End to end:**
  1. Start `llama-server --jinja`.
  2. In a scratch directory, run `python -m gemma_agent -p "list the files here and read README.md"`. The stderr should show `> ls` and `> read` lines.
  3. Run interactively, ask for an edit, and check that Ctrl-C aborts a turn mid-stream.
- **If tool calls show up as plain text instead of `tool_calls`:**
  - Check that `--jinja` is set.
  - Try a bigger Gemma, or narrow the tools with `--tools read,edit,bash`.
  - Lower the temperature.
