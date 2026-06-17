from typing import Dict, TypedDict
from langgraph.graph import StateGraph
from AgentState import AgentState, compliment_node

# class AState(TypedDict):
#     message: str
#
#
# def compliment_node(state: AState) -> AState:
#     """ Simple node to return compliment message"""
#     # f'{state["message"]}, you \'re doing an amazing job learning LangGraph!'
#     state["message"] = f'{state["message"]}, you \'re doing an amazing job learning LangGraph!'
#     return state


if __name__ == '__main__':
    print("Exercise 1")
    graph = StateGraph(AgentState)

    graph.add_node("complimenter", compliment_node)

    graph.set_entry_point("complimenter")
    graph.set_finish_point("complimenter")

    app = graph.compile()

    result = app.invoke({"name": "Bob"})

    print(result["name"])
