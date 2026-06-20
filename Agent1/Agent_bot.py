from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

load_dotenv()

#sk=

class AgentState(TypedDict):
    messages: List[HumanMessage]
    flag: str

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(response.content)
    return state

if __name__ == '__main__':
    print("AI Agent 1")
    print(os.getenv("OPENAI_API_KEY"))
    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    agent = graph.compile()

    user_input = input("Enter: ")

    while user_input != "exit":
        agent.invoke({"messages": [HumanMessage(content=user_input)]})
        user_input = input("Enter: ")