# Changelog

All notable changes to the AgentKavach Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-06-18

### Changed

- First stable release. The public API (`AgentKavach`, `Budget` for daily, monthly, total, and organization budgets, the alert channels, the per run guardrails, and the exception hierarchy) is now stable under semantic versioning. There are no breaking changes since 0.2.0.

## [0.2.0]

### Removed

- `Budget.shared_budget()` has been removed (breaking). The constructor silently ignored its `name` and `agents` arguments and always collapsed to the organization wide pool. Use `Budget.org_budget(limit, period)` instead.

### Added

- A per channel `dispatch` setting. It defaults to `"backend"`. Set it to `"sdk"` to deliver an alert from inside your own network, for example to an internal or on premises endpoint.
- Per run guardrails: `max_tokens_per_run`, `max_calls_per_run`, `max_runtime_seconds`, and loop detection.

## [0.1.0]

### Added

- Initial release: hard budget limits for OpenAI, Anthropic, Google, and Mistral, budget checks that run in memory before each call, threshold alerts, a kill switch, and usage export built on OpenTelemetry.
