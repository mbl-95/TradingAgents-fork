"""The structured thesis: analyst report data, PM pillars and drivers, the trader's entry zone."""

import pytest
from pydantic import ValidationError

from tradingagents.agents.schemas import (
    FundamentalsReport,
    MarketReport,
    NewsReport,
    PortfolioDecision,
    TraderProposal,
    ValuationReport,
    render_fundamentals_report,
    render_market_report,
    render_news_report,
    render_pm_decision,
    render_trader_proposal,
    render_valuation_report,
)

BASE_DECISION = dict(rating="Hold", executive_summary="Wait.", investment_thesis="Balanced.",
                     price_target=None, time_horizon="3-6 months")
BASE_RENDER = (
    "**Rating**: Hold\n\n**Executive Summary**: Wait.\n\n**Investment Thesis**: Balanced.\n\n"
    "**Price Target**: not provided\n\n**Time Horizon**: 3-6 months"
)


def test_a_decision_without_the_new_fields_renders_exactly_as_before():
    assert render_pm_decision(PortfolioDecision(**BASE_DECISION)) == BASE_RENDER


def test_thesis_sections_are_appended_after_every_existing_one():
    decision = PortfolioDecision(**BASE_DECISION, thesis_pillars=[{
        "name": "Demand", "statement": "Data-center demand grows.", "evidence": "Q3 beat.",
        "status": "intact", "kill_criterion": "Two quarters of falling orders.",
    }], change_driver="sector_contagion", move_assessment="overreaction",
        market_concern={"concern": "AI capex debt", "applies_to_company": "partly", "reasoning": "Customers borrow."},
        opportunity_note="Attractive below 160 if orders hold.")
    text = render_pm_decision(decision)
    assert text.startswith(BASE_RENDER + "\n\n**Thesis Pillars**:\n- Demand (intact): Data-center demand grows.")
    order = ["**Thesis Pillars**", "**Change Driver**: sector_contagion", "**Move Assessment**: overreaction",
             "**Market Concern**: AI capex debt Applies to the company: partly.", "**Opportunity**: Attractive"]
    positions = [text.index(s) for s in order]
    assert positions == sorted(positions)


def test_unknown_enum_values_are_rejected():
    with pytest.raises(ValidationError):
        PortfolioDecision(**BASE_DECISION, change_driver="vibes")
    with pytest.raises(ValidationError):
        MarketReport(move_scope="everywhere", trend="up")


def test_trader_entry_zone_only_renders_when_given():
    plain = TraderProposal(action="Buy", reasoning="Trend.", entry_price="$1,234.50")
    assert "Entry Zone" not in render_trader_proposal(plain)
    assert plain.entry_price == 1234.5
    zoned = render_trader_proposal(TraderProposal(action="Buy", reasoning="Trend.", entry_zone_id="D1"))
    assert "**Entry Zone**: D1" in zoned and zoned.endswith("FINAL TRANSACTION PROPOSAL: **BUY**")


def test_market_report_coerces_numbers_and_keeps_a_percent():
    data = MarketReport(move_scope="sector_wide", trend="down", excess_return_1m_vs_sector="-4.2%",
                        key_levels=[{"level": "$180", "kind": "support", "status": "holding"},
                                    {"level": "n/a", "kind": "resistance", "status": "broken"}])
    assert data.excess_return_1m_vs_sector == -4.2
    text = render_market_report(data, "The narrative.")
    assert text.splitlines()[:4] == ["**Move Scope**: sector_wide", "**Excess Return 1M vs Sector**: -4.2%",
                                     "**Trend**: down", "**Key Levels**: support 180 (holding)"]
    assert text.endswith("\n\nThe narrative.")


def test_empty_lists_validate_and_render_as_none():
    assert render_news_report(NewsReport(), "N.").startswith("**Company Events**:\n- none\n\n**External Themes**:\n- none")
    text = render_fundamentals_report(FundamentalsReport(), "F.")
    assert text.startswith("**Fundamentals Changed**: no\n**Key Metrics**:\n- none")


def test_valuation_report_renders_its_three_lines_above_the_narrative():
    data = ValuationReport(stance="undervalued", key_gap="Price 15% under the base case.",
                           what_would_change_it="A guidance cut.")
    assert render_valuation_report(data, "V.") == (
        "**Valuation Stance**: undervalued\n**Key Gap**: Price 15% under the base case.\n"
        "**What Would Change It**: A guidance cut.\n\nV.")
