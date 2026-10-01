# Coding-agent eval tasks

Six small, self-contained tasks for checking that the local model works as a coding agent. Each task gives the agent a prompt and a small repo. Hidden checks then decide pass or fail.

| Task | Category | What the agent has to do |
|---|---|---|
| `bugfix-report` | bug fix | Find why month-end expenses go missing and fix the root cause |
| `feature-json-output` | multi-file feature | Add `--format json` to a CLI without changing the text output |
| `fix-failing-suite` | debug | Make a failing test suite pass without touching the tests |
| `refactor-dedupe` | refactor | Extract duplicated parsing into a shared module, keeping behaviour identical |
| `ops-log-report` | scripting/ops | Summarise rotated (partly gzipped) access logs in an exact format |
| `ops-organise-files` | scripting/ops | Sort files into folders by the date in their names, with `--dry-run` |

Each `tasks/<name>/` directory contains:

- `task.json`: the title, the prompt, the tools the agent may use, and any protected files it must not change
- `repo/`: the starting code, copied into a scratch git repo for each run
- `check/`: hidden tests, run against the agent's result and never shown to the agent
- `solution/`: a reference solution; `scripts/tests/test_eval_tasks.py` checks that it passes and that the untouched repo fails

## Running

With the server running (see the main [README](../README.md)):

```bash
scripts/eval-task list                    # list the tasks
scripts/eval-task run all --repeats 3     # run the agent headless on every task, then check
scripts/eval-task setup bugfix-report     # or set up a scratch dir and drive the agent yourself
scripts/eval-task check DIR               # run the hidden checks on a work dir
scripts/eval-task transcript DIR          # render a headless run's transcript as Markdown
```

Scratch work dirs go to `~/scratch/evals` (override with `EVAL_SCRATCH`). Run records are appended to `~/.local/state/evals/runs.jsonl` (override with `EVAL_RESULTS`).
