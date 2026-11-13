"""Structured data for an analyst's finished report (opt-in: ``structured_analyst_reports``).

The analyst's tool loop and prose are untouched. Once it has written its report,
one structured call reads that report and fills the analyst's schema; the render
helper puts the fields above the prose as written. When the provider has no
structured output or the call fails, the prose is returned unchanged with no
data, so the run never depends on it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured

Structurer = Callable[[str], tuple[str, dict | None]]


def no_structure(report: str) -> tuple[str, dict | None]:
    return report, None


def report_structurer(llm: Any, schema, render: Callable[[Any, str], str], agent_name: str) -> Structurer:
    structured_llm = bind_structured(llm, schema, agent_name)

    def structure(report: str) -> tuple[str, dict | None]:
        if not report or structured_llm is None:
            return report, None
        prompt = (
            f"Below is the {agent_name}'s finished report. Fill the fields from what the report says. "
            "Do not add facts, numbers or judgements it does not state; leave a field empty instead. "
            f"{NO_EXTERNAL_TOOLS}\n\nReport:\n{report}"
        )
        data = invoke_structured(structured_llm, prompt, agent_name)
        if data is None:
            return report, None
        return render(data, report), data.model_dump(mode="json")

    return structure
