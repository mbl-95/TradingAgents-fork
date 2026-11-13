"""Valuation Analyst (opt-in analyst ``"valuation"``): is the price below, at or above what the evidence supports?

Its default tool is the engine's company snapshot (``get_fundamentals``: multiples,
analyst targets). A caller with richer valuation data (consensus estimates per
year, the multiple's own history, fair-value bands) adds its own tool through
``extra_tools={"valuation": [...]}``; the analyst is told to prefer it.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.structure import no_structure, report_structurer
from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.schemas import ValuationReport, render_valuation_report
from tradingagents.agents.tools import get_fundamentals

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (get_fundamentals,)


def create_valuation_analyst(llm, tools=TOOLS, structured: bool = False):
    """``tools`` replaces the default set; ``structured`` adds the stance as data."""
    structure = report_structurer(llm, ValuationReport, render_valuation_report, "Valuation Analyst") if structured else no_structure

    def valuation_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            "You are a valuation analyst. Judge whether the current price is below, at or above what the "
            "valuation evidence supports. Work from numbers your tools return: consensus estimates and their "
            "growth, analyst price targets and how many analysts stand behind them, the current multiple "
            "against its own history, and fair-value ranges when a tool provides them. When a tool gives "
            "stored valuation bands or a valuation snapshot, start there; use the general company snapshot "
            "to fill gaps. State your stance (undervalued, fair, overvalued, or unclear when the data is "
            "thin), the single gap that matters most (price against the base case, the multiple against its "
            "history, or management guidance against consensus), and what observable fact would change the "
            "stance. Say plainly when coverage is thin. Do not judge the business or the trend; other "
            "analysts do that. Keep the report concise and end with a Markdown table of the numbers you used."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, tools, state["messages"])
        report, data = structure(report)

        return {
            "messages": [result],
            "valuation_report": report,
            "valuation_report_data": data,
        }

    return valuation_analyst_node
