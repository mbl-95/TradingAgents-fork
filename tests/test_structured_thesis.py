"""The structured thesis in the graph: report data, caller tools, the prior thesis and the Valuation Analyst."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.tools import tool

from tests import test_graph_end_to_end as e2e
from tests.test_debate_opening import _REPORTS, _capturing_llm, _investment_state, _risk_state
from tests.test_memory_log import _make_pm_state
from tests.test_structured_agents import _make_rm_state, _structured_rm_llm
from tradingagents.agents import schemas
from tradingagents.agents.analysts.structure import report_structurer
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.graph.setup import merge_tools

PILLAR = {"name": "Demand", "statement": "Orders grow.", "evidence": "Q3 beat.", "status": "intact",
          "kill_criterion": "Two quarters of falling orders."}
DECISION = schemas.PortfolioDecision(
    rating="Hold", executive_summary="s", investment_thesis="t", thesis_pillars=[PILLAR],
    change_driver="no_change", move_assessment="justified")


def _structured_llm(result, captured: dict | None = None):
    structured = MagicMock()
    sink = captured if captured is not None else {}
    structured.invoke.side_effect = lambda p: sink.__setitem__("prompt", p) or result
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# --- the structurer ----------------------------------------------------------

@pytest.mark.unit
def test_structurer_puts_the_fields_above_the_prose_and_returns_the_data():
    data = schemas.ValuationReport(stance="fair", key_gap="g", what_would_change_it="w")
    captured = {}
    structure = report_structurer(_structured_llm(data, captured), schemas.ValuationReport,
                                  schemas.render_valuation_report, "Valuation Analyst")
    report, out = structure("The prose.")
    assert report.startswith("**Valuation Stance**: fair") and report.endswith("\n\nThe prose.")
    assert out == {"stance": "fair", "key_gap": "g", "what_would_change_it": "w"}
    assert "The prose." in captured["prompt"] and "Do not add facts" in captured["prompt"]


@pytest.mark.unit
def test_structurer_keeps_the_prose_when_the_call_fails_or_there_is_nothing():
    failing = MagicMock()
    failing.with_structured_output.return_value.invoke.side_effect = RuntimeError("bad json")
    structure = report_structurer(failing, schemas.MarketReport, schemas.render_market_report, "Market Analyst")
    assert structure("Prose.") == ("Prose.", None)
    assert structure("") == ("", None)
    unsupported = MagicMock()
    unsupported.with_structured_output.side_effect = NotImplementedError
    assert report_structurer(unsupported, schemas.NewsReport, schemas.render_news_report, "News")("P.") == ("P.", None)


# --- caller tools ------------------------------------------------------------

@tool
def get_fundamentals(ticker: str) -> str:
    """A caller's replacement for the engine's company snapshot."""
    return "caller"


@tool
def get_relative_performance(ticker: str) -> str:
    """Stock vs sector and market."""
    return "rel"


@pytest.mark.unit
def test_merge_tools_adds_caller_tools_and_a_same_name_one_replaces_the_default():
    from tradingagents.agents.analysts import fundamentals_analyst, market_analyst

    merged = merge_tools(market_analyst.TOOLS, [get_relative_performance])
    assert [t.name for t in merged][-1] == "get_relative_performance" and len(merged) == len(market_analyst.TOOLS) + 1
    replaced = merge_tools(fundamentals_analyst.TOOLS, [get_fundamentals])
    assert [t.name for t in replaced].count("get_fundamentals") == 1 and get_fundamentals in replaced


@pytest.mark.unit
def test_extra_tools_for_an_analyst_without_a_tool_loop_are_refused(tmp_path, monkeypatch, offline):
    with pytest.raises(ValueError, match="social"):
        e2e._graph(tmp_path, monkeypatch, e2e.ScriptedModel()).graph_setup.setup_graph(
            ("market", "social"), extra_tools={"social": [get_relative_performance]})


# --- prior thesis ------------------------------------------------------------

@pytest.mark.unit
def test_pm_and_rm_see_the_prior_thesis_or_are_told_there_is_none():
    for prior, expected in (("2026-11-02 Hold\n- Demand: orders grow", "Prior thesis (the view this run is compared against):\n2026-11-02 Hold"),
                            ("", "Prior thesis: none was given; this is a first view")):
        captured = {}
        out = create_portfolio_manager(_structured_llm(DECISION, captured))({**_make_pm_state(), "prior_thesis": prior})
        assert expected in captured["prompt"]
        rm_captured = {}
        create_research_manager(_structured_rm_llm(rm_captured))({**_make_rm_state(), "prior_thesis": prior})
        assert expected in rm_captured["prompt"]
    assert out["final_decision_data"]["thesis_pillars"] == [PILLAR]
    assert out["final_decision_data"]["change_driver"] == "no_change"


@pytest.mark.unit
def test_a_free_text_pm_decision_has_no_data():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    llm.invoke.return_value = MagicMock(content="**Rating**: Hold\n\nWait.")
    out = create_portfolio_manager(llm)(_make_pm_state())
    assert out["final_decision_data"] is None and out["final_rating"] == "Hold"


# --- the valuation report in the debates --------------------------------------

DEBATERS = [
    (create_bull_researcher, lambda: _investment_state(""), "Valuation report: "),
    (create_bear_researcher, lambda: _investment_state(""), "Valuation report: "),
    (create_aggressive_debator, lambda: _risk_state(), "Valuation Report: "),
    (create_conservative_debator, lambda: _risk_state(), "Valuation Report: "),
    (create_neutral_debator, lambda: _risk_state(), "Valuation Report: "),
]


@pytest.mark.unit
@pytest.mark.parametrize("factory,state,label", DEBATERS)
def test_debaters_read_the_valuation_report_only_when_it_exists(factory, state, label):
    without, empty, given = {}, {}, {}
    factory(_capturing_llm(without))(state())
    factory(_capturing_llm(empty))({**state(), "valuation_report": "  "})
    factory(_capturing_llm(given))({**state(), "valuation_report": "Undervalued by 15%."})
    assert without["prompt"] == empty["prompt"] and label not in without["prompt"]
    assert f"{label}Undervalued by 15%.\n" in given["prompt"]
    assert _REPORTS  # the shared reports are still there


# --- the whole graph -----------------------------------------------------------

@pytest.fixture
def offline(monkeypatch, tmp_path):
    return e2e.offline.__wrapped__(monkeypatch, tmp_path)


@pytest.fixture
def structured_answers(monkeypatch):
    for schema, value in {
        schemas.MarketReport: schemas.MarketReport(move_scope="sector_wide", trend="down"),
        schemas.NewsReport: schemas.NewsReport(),
        schemas.FundamentalsReport: schemas.FundamentalsReport(fundamentals_changed=True),
        schemas.ValuationReport: schemas.ValuationReport(stance="undervalued", key_gap="g", what_would_change_it="w"),
        schemas.PortfolioDecision: DECISION.model_copy(update={"rating": schemas.PortfolioRating.OVERWEIGHT}),
    }.items():
        monkeypatch.setitem(e2e.STRUCTURED, schema, value)


def _desk_graph(tmp_path, monkeypatch, model, extra_tools=None):
    import copy

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph import trading_graph

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
               memory_log_path=str(tmp_path / "log.md"), structured_analyst_reports=True)
    monkeypatch.setattr(trading_graph, "create_tier_client", lambda config, tier, **k: e2e._Client(model))
    return trading_graph.TradingAgentsGraph(("market", "social", "news", "fundamentals", "valuation"),
                                            config=cfg, extra_tools=extra_tools)


@pytest.mark.unit
def test_a_structured_run_with_the_valuation_analyst_and_caller_tools(tmp_path, monkeypatch, offline, structured_answers):
    calls = []

    @tool
    def get_valuation_snapshot(ticker: str) -> str:
        """Stored fair-value bands."""
        calls.append(ticker)
        return "bands"

    graph = _desk_graph(tmp_path, monkeypatch, e2e.ScriptedModel(structured=True),
                        extra_tools={"valuation": [get_valuation_snapshot], "market": [get_relative_performance]})
    state, signal = graph.propagate("NVDA", e2e.TRADE_DATE, prior_thesis="2026-11-02 Hold")

    assert signal == "Overweight" and calls == ["NVDA"]
    assert state["valuation_report"].startswith("**Valuation Stance**: undervalued")
    assert state["valuation_report_data"]["stance"] == "undervalued"
    assert state["market_report_data"]["move_scope"] == "sector_wide"
    assert state["fundamentals_report_data"]["fundamentals_changed"] is True
    assert state["news_report_data"] == {"company_events": [], "external_themes": []}
    assert state["final_decision_data"]["thesis_pillars"] == [PILLAR]
    assert "**Thesis Pillars**:" in state["final_trade_decision"]
    assert (tmp_path / "results").exists()


@pytest.mark.unit
def test_a_free_text_provider_still_gets_every_report_without_data(tmp_path, monkeypatch, offline):
    graph = _desk_graph(tmp_path, monkeypatch, e2e.ScriptedModel(structured=False))
    state, _ = graph.propagate("NVDA", e2e.TRADE_DATE)
    for key in ("market", "news", "fundamentals", "valuation"):
        assert state[f"{key}_report"].strip() == e2e.TEXT and state[f"{key}_report_data"] is None
    assert state["final_decision_data"] is None


@pytest.mark.unit
def test_a_default_run_is_unchanged_by_the_new_options(tmp_path, monkeypatch, offline):
    """No valuation analyst, no structured reports, no caller tools: reports as before, no data."""
    state, _ = e2e._graph(tmp_path, monkeypatch, e2e.ScriptedModel()).propagate("NVDA", e2e.TRADE_DATE)
    assert state["market_report"] == e2e.TEXT and state["valuation_report"] == ""
    assert state["market_report_data"] is None and state["valuation_report_data"] is None
