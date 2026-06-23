from typing_extensions import TypedDict
from typing import List, Dict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from PrintGraph import print_graph
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model

load_dotenv()

class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    total_inr: float
    total_eur: float
    conversionCurrency: str

llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
    temperature=0,
    max_tokens=500
)


def calc_total(state: PortfolioState) -> PortfolioState:
    """Node used to calculate the total_usd which is a conversion of the amount from INR"""
    state["total_usd"] = state["amount_usd"] * 1.08
    state["conversionCurrency"] = state["conversionCurrency"].lower()
    return state

def convert_to_inr(state: PortfolioState) -> PortfolioState:
    """Node used to convert the total_usd to total_inr"""
    state["total_inr"] = state["total_usd"] * 90
    return state

def convert_to_eur(state: PortfolioState) -> PortfolioState:
    """Node used to convert the total_usd to total_inr"""
    state["total_eur"] = state["total_usd"] * 0.7
    return state

def selectConversionType(state: PortfolioState) -> str:
    if(state["conversionCurrency"] == "eur"):
        return "eur"
    else:
        return "inr"

if __name__ == '__main__':
    graph = StateGraph(PortfolioState)

    graph.add_node("calcTotal", calc_total)
    graph.set_entry_point("calcTotal")

    graph.add_node("convertToINR", convert_to_inr)
    graph.add_node("convertToEUR", convert_to_eur)

    graph.add_conditional_edges(
        "calcTotal",
        selectConversionType,
        {
            "eur": "convertToEUR",
            "inr": "convertToINR"
        }
    )

    graph.add_edge("convertToINR", END)
    graph.add_edge("convertToEUR", END)


    app = graph.compile()

    response = llm.invoke("Who was the first person to walk on the moon?")

    print(response)

    result = app.invoke({"amount_usd": 10, "conversionCurrency": "INR"})
    print(result)

    result = app.invoke({"amount_usd": 10, "conversionCurrency": "EUR"})

    print(result)

    print_graph(app)