"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    """Normalise an LLM-written optional numeric field before validation.

    Three shapes show up in practice: a placeholder string ("None", "N/A") in
    place of an omitted value (#1058); a percentage where a price was asked for
    ("15%", #1288); and a human-formatted price ("$1,234.50"). A percentage
    cannot be salvaged into an absolute level -- reading "15%" as 15 would put a
    stop at $15 on a $600 stock -- so it is dropped like a placeholder, leaving
    one bad field to null out instead of failing the whole proposal. A formatted
    price is reduced to its number.

    Anything that is not a single number is dropped the same way. A range
    ("150-160") or a hedge ("around 150") would otherwise reach pydantic, fail
    validation, and discard the whole decision, losing every field the model got
    right along with the price.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _NULLISH_FLOAT or text.endswith("%"):
        return None
    cleaned = text.replace(",", "").lstrip("$€£¥").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(StrEnum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(StrEnum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Conflicting arguments alone are not a "
            "reason to Hold: commit to the stronger side, sized by how "
            "decisively it wins. Choose Hold only when the evidence is still "
            "balanced after weighing, or too thin to support a call."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including sizing guidance relative to a standard allocation. The "
            "research team does not see the caller's holdings; the trader and "
            "portfolio manager apply the actual position."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Optional entry price target as an absolute number in the instrument's "
            "quote currency (e.g. 189.5), never a percentage or a range. Omit it "
            "if you cannot state a specific level."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional stop-loss as an absolute price in the instrument's quote "
            "currency (e.g. 172.0), never a percentage. Convert a percentage "
            "distance to the price level it implies, or omit it."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )
    entry_zone_id: str | None = Field(
        default=None,
        description=(
            "Optional: when the market report names price zones by label (e.g. 'D1' for a demand "
            "zone), the label of the zone the entry sits in. Omit it when the report names no zones "
            "or the entry is not in one."
        ),
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    # Named even when absent, so a reader can tell a level the trader chose not
    # to give from one the schema never asked for.
    for label, value in (("Entry Price", proposal.entry_price),
                         ("Stop Loss", proposal.stop_loss),
                         ("Position Sizing", proposal.position_sizing)):
        parts.extend(["", f"**{label}**: {value if value is not None and value != '' else 'not provided'}"])
    # Only when given: most callers have no zones, and their reports stay as they were.
    if proposal.entry_zone_id:
        parts.extend(["", f"**Entry Zone**: {proposal.entry_zone_id}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class ThesisPillar(BaseModel):
    """One claim the rating rests on."""

    name: str = Field(description="A short name, e.g. 'Data-center demand'.")
    statement: str = Field(description="The claim, in one sentence.")
    evidence: str = Field(description="The evidence from the analysts that supports it now.")
    status: Literal["new", "intact", "weakened", "broken"] = Field(
        description="Against the prior thesis: new when it had no such pillar or there is no prior thesis.",
    )
    kill_criterion: str = Field(description="What observable fact would prove the pillar wrong.")


class ChangeDriver(StrEnum):
    COMPANY_FUNDAMENTALS = "company_fundamentals"
    COMPANY_NEWS = "company_news"
    SECTOR_CONTAGION = "sector_contagion"
    MACRO = "macro"
    SENTIMENT_FLOW = "sentiment_flow"
    TECHNICAL = "technical"
    NO_CHANGE = "no_change"


class MoveAssessment(StrEnum):
    JUSTIFIED = "justified"
    OVERREACTION = "overreaction"
    UNDERREACTION = "underreaction"
    UNCLEAR = "unclear"


class MarketConcern(BaseModel):
    concern: str = Field(description="What worried investors fear.")
    applies_to_company: Literal["yes", "partly", "no"]
    reasoning: str


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate. "
            "Conflicting arguments alone are not a reason to Hold: commit to the "
            "stronger side, sized by how decisively it wins. Choose Hold only "
            "when the evidence is still balanced after weighing, or too thin to "
            "support a call."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    # The thesis in structure. All optional: a provider that drops them still
    # validates, and a decision without them renders exactly as before.
    thesis_pillars: list[ThesisPillar] = Field(
        default_factory=list,
        description="The two to five claims the rating rests on, each with what would prove it wrong.",
    )
    change_driver: ChangeDriver | None = Field(
        default=None,
        description=(
            "The main reason the view differs from the prior thesis; no_change when it does not or "
            "when there is no prior thesis."
        ),
    )
    move_assessment: MoveAssessment | None = Field(
        default=None,
        description="Whether the recent price move is in proportion to what changed in the business.",
    )
    market_concern: MarketConcern | None = Field(
        default=None, description="What worried investors fear, and whether it applies to this company.",
    )
    opportunity_note: str | None = Field(
        default=None,
        description=(
            "Only when move_assessment is overreaction: what would make the lower price attractive, "
            "and what would invalidate that. A generic signal, never advice about a position."
        ),
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    # Named even when absent: a missing line reads as a field nobody asked for,
    # so a reader cannot tell "no target" from "target not reported".
    target = decision.price_target if decision.price_target is not None else "not provided"
    parts.extend(["", f"**Price Target**: {target}"])
    parts.extend(["", f"**Time Horizon**: {decision.time_horizon or 'not provided'}"])
    # Appended after every existing section, and only when given, so the
    # existing headers and their parsers are untouched.
    if decision.thesis_pillars:
        pillars = [f"- {p.name} ({p.status}): {p.statement} Evidence: {p.evidence} "
                   f"Wrong if: {p.kill_criterion}" for p in decision.thesis_pillars]
        parts.extend(["", "**Thesis Pillars**:", *pillars])
    if decision.change_driver is not None:
        parts.extend(["", f"**Change Driver**: {decision.change_driver.value}"])
    if decision.move_assessment is not None:
        parts.extend(["", f"**Move Assessment**: {decision.move_assessment.value}"])
    if decision.market_concern is not None:
        c = decision.market_concern
        parts.extend(["", f"**Market Concern**: {c.concern} Applies to the company: {c.applies_to_company}. "
                          f"{c.reasoning}"])
    if decision.opportunity_note:
        parts.extend(["", f"**Opportunity**: {decision.opportunity_note}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(StrEnum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# ---------------------------------------------------------------------------
# Analyst report data (opt-in: config["structured_analyst_reports"])
# ---------------------------------------------------------------------------
#
# The Market, News, Fundamentals and Valuation analysts keep writing prose in
# their tool loop. With structured reports on, one short structured call reads
# the finished report and fills these fields; the prose is kept as written and
# the render helpers put a header above it. So these schemas carry no narrative
# field: the model is asked only for what prose does not keep consistent from
# run to run. Lists default to empty, so a thin report still validates.


class MoveScope(StrEnum):
    STOCK_SPECIFIC = "stock_specific"
    SECTOR_WIDE = "sector_wide"
    MARKET_WIDE = "market_wide"
    NO_NOTABLE_MOVE = "no_notable_move"


class Trend(StrEnum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


class KeyLevel(BaseModel):
    level: float | None = Field(description="The price level, an absolute number in the quote currency.")
    kind: Literal["support", "resistance", "moving_average"] = Field(description="What the level is.")
    status: Literal["holding", "broken", "reclaimed"] = Field(description="How price is treating it now.")

    @field_validator("level", mode="before")
    @classmethod
    def _level(cls, v):
        return _coerce_optional_float(v)


def _percent(value):
    """A percentage field may arrive as "12.5%"; unlike a price, the sign is the unit, so keep the number."""
    if isinstance(value, str) and value.strip().endswith("%"):
        value = value.strip()[:-1]
    return _coerce_optional_float(value)


class MarketReport(BaseModel):
    """What the Market Analyst's report says about the move and the levels."""

    move_scope: MoveScope = Field(
        description=(
            "Where the recent move comes from, as the report judges it against the sector and the market: "
            "stock_specific, sector_wide, market_wide, or no_notable_move. When the report does not "
            "compare the stock with its sector or the market, use no_notable_move unless it calls the "
            "move company-specific itself."
        ),
    )
    excess_return_1m_vs_sector: float | None = Field(
        default=None,
        description="The stock's one-month return minus its sector's, in percent, only if the report states it.",
    )
    trend: Trend = Field(description="The prevailing trend on the daily chart according to the report.")
    key_levels: list[KeyLevel] = Field(
        default_factory=list, description="The price levels the report's argument relies on, at most five.",
    )

    @field_validator("excess_return_1m_vs_sector", mode="before")
    @classmethod
    def _excess(cls, v):
        return _percent(v)


class CompanyEvent(BaseModel):
    headline: str
    date: str | None = Field(default=None, description="YYYY-MM-DD when the report gives it.")
    source: str | None = None
    category: Literal["earnings", "guidance", "product", "customer", "legal", "management", "capital", "other"]
    direction: Literal["positive", "negative", "neutral"]
    materiality: Literal["low", "medium", "high"]


class ExternalTheme(BaseModel):
    theme: str = Field(description="A sector or macro story, e.g. 'AI infrastructure debt financing'.")
    description: str
    channel: Literal["demand", "funding_cost", "multiple_compression", "supply_chain", "regulation",
                     "sentiment_flow"] = Field(description="How the theme reaches the company.")
    exposure: Literal["direct", "indirect", "none"] = Field(
        description="The company's real exposure, judged from its own business, not from how the stock trades.",
    )
    direction: Literal["positive", "negative", "neutral"]


class NewsReport(BaseModel):
    """The News Analyst's report, split into news about the company and the stories around it."""

    company_events: list[CompanyEvent] = Field(default_factory=list, description="News about the company itself.")
    external_themes: list[ExternalTheme] = Field(
        default_factory=list,
        description="Sector and macro stories that can move this stock without being about it.",
    )


class Metric(BaseModel):
    name: str = Field(description="e.g. revenue growth, gross margin, free cash flow, net debt / EBITDA, forward P/E.")
    value: str = Field(description="The value as the report states it, with its unit.")
    period: str | None = Field(default=None, description="e.g. 'Q3 FY26' or 'TTM'.")
    change_vs_prior: str | None = Field(default=None, description="The change against the prior period, if stated.")


class QualityFlag(BaseModel):
    flag: str = Field(description="e.g. balance sheet, cash conversion, customer concentration, dilution.")
    direction: Literal["improving", "deteriorating", "stable"]


class FundamentalsReport(BaseModel):
    """The handful of numbers the Fundamentals Analyst's report argues from."""

    metrics: list[Metric] = Field(default_factory=list, description="The numbers the report argues from, at most eight.")
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    fundamentals_changed: bool = Field(
        default=False,
        description="True when a reported number changed since the last filing in a way that matters to an investment case.",
    )


class ValuationStance(StrEnum):
    UNDERVALUED = "undervalued"
    FAIR = "fair"
    OVERVALUED = "overvalued"
    UNCLEAR = "unclear"


class ValuationReport(BaseModel):
    """The Valuation Analyst's view in three lines."""

    stance: ValuationStance = Field(
        description="Whether the price is below, at or above what the valuation evidence supports; unclear when the data is too thin.",
    )
    key_gap: str = Field(
        description="The one gap that matters most, e.g. price vs the base case, the multiple vs its own history, guidance vs consensus.",
    )
    what_would_change_it: str = Field(description="What would move the stance, stated as something observable.")


def _bullets(rows: list[str]) -> list[str]:
    return [f"- {r}" for r in rows] if rows else ["- none"]


def render_market_report(data: MarketReport, narrative: str) -> str:
    levels = [f"{k.kind.replace('_', ' ')} {k.level:g} ({k.status})" for k in data.key_levels if k.level is not None]
    excess = (f"{data.excess_return_1m_vs_sector:+.1f}%" if data.excess_return_1m_vs_sector is not None
              else "not stated")
    return "\n".join([
        f"**Move Scope**: {data.move_scope.value}",
        f"**Excess Return 1M vs Sector**: {excess}",
        f"**Trend**: {data.trend.value}",
        f"**Key Levels**: {'; '.join(levels) if levels else 'none'}",
        "",
        narrative,
    ])


def render_news_report(data: NewsReport, narrative: str) -> str:
    events = [f"{e.headline} ({e.category}, {e.direction}, {e.materiality})" for e in data.company_events]
    themes = [f"{t.theme} ({t.channel}, exposure {t.exposure}, {t.direction})" for t in data.external_themes]
    return "\n".join(["**Company Events**:", *_bullets(events), "", "**External Themes**:", *_bullets(themes),
                      "", narrative])


def render_fundamentals_report(data: FundamentalsReport, narrative: str) -> str:
    metrics = [f"{m.name}: {m.value}" + (f" ({m.period})" if m.period else "") for m in data.metrics]
    flags = [f"{f.flag}: {f.direction}" for f in data.quality_flags]
    return "\n".join([
        f"**Fundamentals Changed**: {'yes' if data.fundamentals_changed else 'no'}",
        "**Key Metrics**:", *_bullets(metrics), "", "**Quality Flags**:", *_bullets(flags), "", narrative,
    ])


def render_valuation_report(data: ValuationReport, narrative: str) -> str:
    return "\n".join([
        f"**Valuation Stance**: {data.stance.value}",
        f"**Key Gap**: {data.key_gap}",
        f"**What Would Change It**: {data.what_would_change_it}",
        "",
        narrative,
    ])
