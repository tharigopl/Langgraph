from typing import Annotated, Dict, TypedDict, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers"""
    return a + b

@tool
def subtract(a: int, b: int):
    """This is a subtraction function that subtracts 2 numbers"""
    return a - b


@tool
def multiply(a: int, b: int):
    """This function will multiply two numbers"""
    return a * b

tools = [add, subtract, multiply]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    """Function which makes the calls to LLM with the query"""
    #system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    #response = model.invoke([system_prompt] + state["messages"])
    response = model.invoke(state["messages"])

    return {"messages": [response]}


def should_condition(state: AgentState) -> AgentState:
    """Conditional edge to check to continue tool calls or exit"""
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"




if __name__ == '__main__':
    print('Agent 3')
    graph = StateGraph(AgentState)

    graph.add_node("agent", model_call)

    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_condition,
        {
            "end": END,
            "continue": "tools"
        }
    )

    graph.add_edge("tools", "agent")

    app = graph.compile()


    def print_stream_values(stream):
        printed = 0
        for s in stream:
            messages = s["messages"]
            for message in messages[printed:]:
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
            printed = len(messages)

    def print_stream_updates(stream):
        printed = 0
        for s in stream:
            for node_name, node_update in s.items():
                if "messages" in node_update:
                    messages = node_update["messages"]
                    for message in messages[printed:]:
                        if isinstance(message, tuple):
                            print(message)
                        else:
                            message.pretty_print()
                    printed = len(messages)

    inputs = {"messages": [("user", "Addiing 40 + 10 + 10 + 10 and multiply by 10 and tell me a joke")]}
    print_stream_values(app.stream(inputs, stream_mode="values"))
    #print_stream_updates(app.stream(inputs, stream_mode="updates"))
