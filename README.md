# AgentKavach

[![PyPI](https://img.shields.io/pypi/v/agentkavach.svg)](https://pypi.org/project/agentkavach/)
[![Python](https://img.shields.io/pypi/pyversions/agentkavach.svg)](https://pypi.org/project/agentkavach/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Hard budget limits for LLM agents. AgentKavach wraps your OpenAI, Anthropic, Google, or Mistral client, tracks spend in real time, and stops an agent once it reaches its budget, so a runaway loop or retry storm cannot quietly run up your API bill.

```bash
pip install agentkavach
```

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Budgets](#budgets)
- [Guardrails](#guardrails)
- [Alerts and the kill switch](#alerts-and-the-kill-switch)
- [Providers](#providers)
- [Security and privacy](#security-and-privacy)
- [How it works](#how-it-works)
- [Open core](#open-core)
- [Documentation](#documentation)
- [License](#license)

## Overview

An AI agent can spend money on every API call. When one gets stuck in a loop or retries aggressively, the cost climbs faster than a human can react, and most tooling only reports the damage after it happens.

AgentKavach is a circuit breaker for that problem. It sits in front of your LLM client, checks the running spend before each call, raises alerts as usage approaches a limit, and refuses further calls once the budget is reached. The check runs locally and in memory, so enforcement does not depend on a network round trip.

You pass both keys explicitly: your AgentKavach key (`api_key`) and your provider key (`llm_key`). The SDK never reads them from the environment. Your provider key is used only inside your process to call the provider directly. It is never read from the environment, never written to disk, and never sent to AgentKavach.

## Installation

```bash
pip install agentkavach
```

AgentKavach supports Python 3.9 and later. OpenAI works out of the box. To use Anthropic, Google, or Mistral, install that provider's own SDK alongside it.

## Quickstart

```python
from agentkavach import AgentKavach, Budget

guard = AgentKavach(
    provider="openai",
    api_key="ak_...",          # your AgentKavach key
    llm_key="sk-...",          # your OpenAI key, used locally and never sent to AgentKavach
    agent_name="research-bot",
    budget=Budget.daily(20),   # a hard 20 USD per day cap
)

response = guard.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize today's headlines."}],
)
```

> Both `api_key` and `llm_key` are required and must be passed explicitly to the constructor. The SDK never reads them from the environment, so supply them yourself, for example `api_key=os.environ["AGENTKAVACH_API_KEY"]`. If either is missing the constructor raises `ValueError` immediately. Your provider key (`llm_key`) stays in your process: it is used only to call the provider directly, is never read from the environment, is never written to disk, and is never sent to AgentKavach.

`guard.create(...)` mirrors the underlying provider client, so you can drop it into existing code with minimal changes. Before each call the engine checks the running spend. As usage crosses the configured thresholds it sends alerts, and once the budget is exhausted the next call raises `BudgetExceededError` instead of reaching the provider.

## Budgets

A budget defines how much an agent may spend over a period.

```python
from agentkavach import Budget

Budget.daily(20)      # 20 USD per calendar day
Budget.monthly(500)   # 500 USD per calendar month
Budget.total(1000)    # 1000 USD lifetime cap

# Share one budget across every agent in an organization:
Budget.org_budget(limit=2000, period="monthly")
```

Per agent budgets cap a single agent. An organization budget aggregates spend across every agent that reports to the same backend, which is useful for a fleet of workers that should share one limit.

## Guardrails

Budgets cap cost. Guardrails cap the shape of a single run, and are passed to the constructor:

```python
guard = AgentKavach(
    provider="openai",
    api_key="ak_...",
    llm_key="sk-...",
    agent_name="research-bot",
    budget=Budget.daily(20),
    max_tokens_per_run=50_000,    # stop the run after 50k tokens
    max_calls_per_run=100,        # stop after 100 API calls
    max_runtime_seconds=300,      # stop after 5 minutes
    detect_loops=True,            # halt on repeated identical calls
)
```

Each guardrail raises a specific, catchable exception (`TokenLimitError`, `CallLimitError`, `RuntimeLimitError`, `LoopDetectedError`) when its limit is reached.

## Alerts and the kill switch

Attach channels that fire as usage crosses thresholds you choose. Email, Slack, PagerDuty, and webhooks are supported.

```python
from agentkavach import AgentKavach, Budget, ChannelType

guard = AgentKavach(
    provider="openai",
    api_key="ak_...",
    llm_key="sk-...",
    agent_name="research-bot",
    budget=Budget.daily(20),
    channels=[
        AgentKavach.channel(ChannelType.EMAIL, threshold=0.5, to="oncall@example.com"),
        AgentKavach.channel(ChannelType.SLACK, threshold=0.8, webhook_url="https://hooks.slack.com/services/..."),
        AgentKavach.channel(ChannelType.PAGERDUTY, threshold=1.0, routing_key="R0..."),
    ],
    on_kill=lambda: print("agent halted"),
)
```

A threshold of `0.8` fires when usage reaches 80 percent of the budget. The optional `on_kill` callback runs once when an agent is halted, which is a convenient place to release resources or page a human.

## Providers

One API across four providers. Set `provider` and pass the matching key as `llm_key`:

| Provider  | `provider` value |
| --------- | ---------------- |
| OpenAI    | `"openai"`       |
| Anthropic | `"anthropic"`    |
| Google    | `"google"`       |
| Mistral   | `"mistral"`      |

## Security and privacy

This SDK handles your provider key, so here is exactly what it does with your data. Every line is open and MIT licensed, so you can verify these claims yourself.

- Your provider key (`llm_key`) is passed explicitly to the constructor. The SDK never reads it from the environment, never writes it to disk, and never sends it to AgentKavach. It is held only in memory, used solely to call the provider directly from your process.
- Your AgentKavach key (`api_key`) is likewise passed explicitly and used only to authenticate spend-tracking requests to the AgentKavach backend. If an api_key is expired or revoked, your LLM calls keep running — the SDK simply stops sending spend data once the backend rejects the key, so a lapsed key never takes your application down.
- For spend tracking and alerts, the SDK reports the following to the AgentKavach backend on each call: agent name, provider, model, input and output token counts, computed cost, duration, timestamp, and a run identifier.
- Prompt and response text is sent only when you opt in with `save_prompts=True`. It is off by default.
- The budget check runs in your process and in memory, so enforcement does not wait on a network call.

## How it works

1. Before a call, the engine reads the running spend for the active budget and compares it to the limit.
2. If the limit is already reached, it raises `BudgetExceededError` and the call never goes out.
3. Otherwise it forwards the call to the provider, then records the actual cost and usage and checks the configured thresholds.
4. If a threshold was crossed, it dispatches the matching alerts.

AgentKavach is designed to fail open. If anything inside the SDK raises an unexpected error, your LLM call still proceeds. Only the budget and guardrail errors are allowed to propagate.

## Open core

The SDK in this repository is open source under the MIT license. The hosted backend and dashboard, which add spend analytics, organization budgets aggregated across processes, and alert delivery, are a separate commercial service at [agentkavach.com](https://agentkavach.com). They are optional and are not required to read, run, or audit this code.

## Documentation

- Full documentation: https://agentkavach.com/public/docs
- Pricing: https://agentkavach.com/public/pricing
- Package on PyPI: https://pypi.org/project/agentkavach/

## License

Released under the [MIT License](LICENSE).
