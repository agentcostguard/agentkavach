"""Release smoke test for the built ``agentkavach`` wheel.

Run by ``.github/workflows/publish.yml`` against the installed wheel on every
supported Python version. It verifies the public API imports and constructs.

Note: this deliberately loads NO native code — the provider SDKs (openai,
anthropic, google-genai) and tiktoken are imported lazily, only on a real LLM
call, which this smoke never makes. That makes the smoke fully deterministic:
any failure here is a genuine import/API break, so the workflow can safely
retry it to absorb transient GitHub-runner crashes (e.g. a SIGSEGV / exit 139
executing pure Python) without ever masking a real failure.
"""

from agentkavach import AgentKavach, Budget, BudgetExceededError
from agentkavach.budget import Period

# Public exception + enum are importable.
assert BudgetExceededError is not None
assert Period is not None

# api_key and llm_key are both required and validated at construction, so pass
# dummy placeholders.
cg = AgentKavach(api_key="ak_test_smoke", llm_key="sk-test", budget=Budget.daily(10))
assert cg.spent == 0.0
assert cg.remaining == 10.0

# Multi-provider construction (no real calls, so the provider SDKs stay unloaded).
AgentKavach(
    api_key="ak_test_smoke", provider="anthropic", llm_key="sk-ant-test", budget=Budget.daily(10)
)
AgentKavach(
    api_key="ak_test_smoke", provider="google", llm_key="AIza-test", budget=Budget.daily(10)
)

# Org budget always uses the "__org__" sentinel so the server aggregates spend
# across every agent.
org = Budget.org_budget(limit=100, period="daily")
assert org.shared_name == "__org__"
assert org.is_shared
assert org.limit == 100

print("All smoke tests passed.")
