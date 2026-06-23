"""Manual ds4 smoke test — tool calling + structured output through OSPREY.

Requires a running ds4 server. Override the endpoint via DS4_BASE_URL.
Run:  UV_LINK_MODE=copy DS4_BASE_URL=http://127.0.0.1:8401/v1 uv run python scripts/smoke/ds4_smoke.py
"""

import os

from pydantic import BaseModel, Field

from osprey.models.providers.ds4 import DS4ProviderAdapter
from osprey.models.providers.litellm_adapter import execute_litellm_completion

BASE = os.environ.get("DS4_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("DS4_MODEL", "deepseek-v4-flash")
COMMON = {
    "provider": "ds4",
    "model_id": MODEL,
    "api_key": "EMPTY",
    "base_url": BASE,
    "max_tokens": 400,
    "temperature": 0.0,
}

GET_PV = {
    "type": "function",
    "function": {
        "name": "get_pv",
        "description": "Read the current value of an EPICS process variable (PV).",
        "parameters": {
            "type": "object",
            "properties": {"pv_name": {"type": "string"}},
            "required": ["pv_name"],
        },
    },
}


class TuningDecision(BaseModel):
    action: str = Field(description="one of: increase, decrease, hold")
    variable: str
    step_size: float
    rationale: str


def main():
    ok, msg = DS4ProviderAdapter().check_health(api_key="EMPTY", base_url=BASE)
    print("health:", ok, "-", msg)
    assert ok, msg

    tc = execute_litellm_completion(
        message="What is the current value of SR:DCCT? Use a tool.",
        tools=[GET_PV],
        tool_choice="auto",
        **COMMON,
    )
    print("tool_call:", tc)
    # A tool call comes back as a non-empty list; plain text falls through to "".
    assert tc and isinstance(tc, list), repr(tc)

    res = execute_litellm_completion(
        message="Beam loss rose after the last booster RF phase change. "
        "Propose one corrective step.",
        output_format=TuningDecision,
        **COMMON,
    )
    assert isinstance(res, TuningDecision), repr(res)
    print("structured:", type(res).__name__, "->", res.model_dump())


if __name__ == "__main__":
    main()
