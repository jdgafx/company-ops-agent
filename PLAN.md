# PLAN: company-ops-agent (SK.2 showcase project 1)

Goal: a Python agent service. An LLM agent answers and acts on questions about a fictional company's data by calling tools in a loop it decides for itself (no fixed retrieve-then-answer pipeline). Tools live in one shared registry; the agent calls it in process, and the project's own MCP server exposes the same registry to external MCP clients (deviation from the first draft, which had the agent call the MCP server as a client; decided 2026-10-17 to keep the loop simple). The service has a FastAPI API, an eval suite that gates CI, pytest tests, Docker, and a free-tier cloud deploy.

Done = the SK.2 done-when list: public repo URL; CI green including the eval gate; live deploy URL answers; README with diagram and eval table; evidence file reviews/evidence/portfolio-lead-SK2-agent-service.md.
Do-not-touch: no paid cloud spend; no secrets in git; no change to the 13 existing apps.

## Data
- Fictional company "Lumen Outdoor Co." (invented). Files in `data/`: customers.json, orders.json, inventory.json, tickets.json. No real client data.
- Read tools: search_customers, get_customer, get_order, list_orders(status), check_inventory(sku).
- Write tool: create_ticket (adds to tickets.json in a copy-on-write store). Write tools are gated: the agent must pass `confirm=true` after it states what it will do; eval checks that writes happen only when asked.

## Steps (each with proof)
1. Scaffold: pyproject, src layout `src/ops_agent/`, .gitignore, .env.example, empty tests dir. Proof: `pip install -e .[dev]` works; `pytest` runs (0 tests, exit 0 with a smoke test).
2. Data + store: JSON loader, typed models (pydantic), store with read API and ticket write. Proof: unit tests for every read tool and create_ticket (pytest green).
3. Tools layer: plain Python functions with JSON schemas, one registry used by both the agent and the MCP server. Proof: tests call each tool through the registry; unknown tool and bad args return structured errors (no crash).
4. MCP server: `ops_agent.mcp_server` using the official `mcp` Python SDK, exposing the registry over stdio and streamable HTTP. Proof: a test starts the server in-process and lists tools and calls one through an MCP client.
5. Agent loop: Anthropic SDK tool-use loop. Before writing it, load the `claude-api` skill and confirm model id, tool-use params and whether `temperature` is sent (never from memory). Model id from config (`claude-haiku-5-5` default). Agent connects to the project MCP server as its tool source. Hard cap on steps (8) and on tokens; returns answer + trace (tool calls, args, results, latency, token counts). Proof: tests with a scripted fake model client cover: direct answer, two-tool chain, step cap hit, tool error recovery.
6. FastAPI: `POST /v1/ask` (question) -> answer + trace; `GET /healthz`. Input validation at the boundary (length limit, no empty question). Proof: pytest with TestClient; 422 on bad input; 200 with trace on a scripted run.
7. Eval suite: `evals/tasks.jsonl` (about 25 tasks: lookups, multi-step lookups, a write with confirm, a refusal case where data is absent, a prompt-injection case inside a ticket text). Checks are programmatic (expected tool called, expected facts in answer, no forbidden write). Runner prints pass rate, mean cost and latency, writes `evals/results.json`. Gate: exit 1 if pass rate < threshold in `evals/gate.json` (absolute, set in step 8 from measured variance). Proof: runner run locally with a scripted model shows the gate fails on a deliberately broken tool and passes on the good one.
8. Live eval: same runner with the real model. Run it 3 times to measure variance before committing a baseline. The live gate uses an absolute threshold with margin (set from the measured minimum, not a delta). The scripted-model eval is the deterministic regression check. Proof: 3 results files with pass rates, spend per run recorded. Gate proof: one deliberate break pushed to a throwaway branch gives a red CI run; the main branch gives a green run.
9. Docker: slim image, non-root user, healthcheck. Proof: `docker build` and `docker run` then `curl /healthz` returns 200 (run locally, not deploy).
10. CI: GitHub Actions workflow `ci.yml`: lint (ruff), pytest, eval gate (replay of recorded live answers: no key, no spend; fails below the gate in evals/gate.json), docker build. Proof: green run URL.
11. Deploy: free tier only. BLOCKED until Chris chooses a host and provides an account (see Blockers).
12. README: architecture diagram (mermaid), how to run, tests, eval table (from results.json), live demo link (after step 11), limits, what it does not do. No gap statements; no claims about Chris beyond verified facts.
13. Evidence file and independent review. Reviewer (sonnet) reads the diff and reports only; implementer fixes.

## Model routing for this build
- Code edits: sonnet helpers, one file = one editor. Reviews: sonnet reviewer, reports only.
- Lookups and log checks: haiku.
- Live eval runs are paid API calls: keep each run to the task set only; no loops of re-runs.

## Abuse and spend limits (required before step 11)
- Per-IP rate limit on /v1/ask.
- Global daily cap on requests and on tokens; 429 once hit. Counter in a file volume, not memory.
- max_tokens small per call; question length cap (already in step 6); step cap (8).
- Documented in the README limits section. Reviewer checks these specifically.

## Free-tier targets (verify each claim from vendor docs, write wiki/topics page)
- Azure: Azure Container Registry is not free, so push the image to GHCR. Container Apps min replicas 0. Log Analytics within its free ingestion.
- AWS alternative: Lambda container image + Function URL. App Runner is not free. Check ECR free storage.

## Eval key and spend
- Provider: OpenRouter (Anthropic-compatible endpoint), as decided in SM.1. The live key is used only by the local runner (`OPS_AGENT_API_KEY`) for the three live runs. CI holds no key; the server holds no key (BYOK).
- Record spend per live eval run (tokens and USD estimate) in the evidence file.

## Blockers (raised only at deploy, after probing)
- Deploy host. Probe first: does an Azure or AWS account already exist (Azure CLI from its Docker image, device-code login in the cng browser, then `az account list`)? If a subscription exists, deploy is unblocked.
- Block only if a new account or a card is needed. The block lists the ways tried.

## Not in scope
- No changes to the 13 portfolio apps. No real client data. No paid services. No secrets in git.
