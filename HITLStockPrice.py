from typing_extensions import TypedDict
from typing import List, Dict, Annotated, Sequence, Union
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from PrintGraph import print_graph
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def get_stock_price(ticker: str) -> str:
    """Tool node used to get the price of the stock ticker"""
    dictionary = {
        "AMZN": 100,
        "MSFT": 150,
        "AAPL": 200
    }
    if ticker in dictionary:
        return dictionary[ticker]
    return "Ticker price not found"





def human_input_node(state: AgentState):
    """Pauses the graph and waits for user input via interrupt()"""
    user_input = interrupt("Waiting for user input")
    print("human node: ", user_input)
    return {"messages": [HumanMessage(content=user_input)]}


def agent_node(state: AgentState) -> AgentState:
    """Calls the LLM with the current message history"""
    system_prompt = SystemMessage(content="You are an AI stock trading assistant. Help the user with anything related to stock trading.")
    response = llm.invoke([system_prompt] + list(state["messages"]))

    #print(f'\n🤖 AI in the agent node: {response.content}')
    if response.tool_calls:
        print(f'🛠️ USING TOOLS: {[tc["name"] for tc in response.tool_calls]}')

    return {"messages": [response]}

@traceable
def call_graph(query: Union[TypedDict, str], resume: str) -> dict:
    if resume == "":
        result = app.invoke(query, config=config)
    else:
        result = app.invoke(Command(resume=query), config=config)
    return result

@tool
def buy_stocks(symbol: str, quantity: int, total_price: float) -> str:
    """Buy stocks given the stock symbol and quantity"""

    return {"MSFT": 200.3, "AAPL": 100.4, "AMZN": 150}.get(symbol, 0.0)

tools = [get_stock_price, buy_stocks]

llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

if __name__ == '__main__':
    graph = StateGraph(AgentState)

    graph.add_node("human_input", human_input_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    def route_agent(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"          # tool calls pending → run tools
        return "human_input"        # done → wait for next user input

    graph.add_edge(START, "human_input")
    graph.add_edge("human_input", "agent")
    graph.add_conditional_edges("agent", route_agent, {
        "tools": "tools",
        "human_input": "human_input"
    })
    graph.add_edge("tools", "agent")

    app = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "user_123"}}

    print_graph(app)

    # First invoke — starts the graph, immediately hits interrupt() in human_input_node

    #app.invoke({"messages": []}, config=config)
    call_graph({"messages": []}, "")

    while True:
        user_input = input("\nuser: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # Resume the paused graph by passing the user input via Command(resume=...)
        #result = app.invoke(Command(resume=user_input), config=config)
        result = call_graph(user_input, "resume")


        # Print the last message only if it has content (not a tool call)
        last_message = result["messages"][-1]
        if hasattr(last_message, "content") and last_message.content:
            print(f"\n🤖 AI: {last_message.content}")
