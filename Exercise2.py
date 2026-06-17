import math
from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    values : List
    operation : str
    name : str
    result : str

def operate(state: AgentState) -> AgentState:
    if state["operation"] == '+':
        state["result"] = f'Hi {state["name"]}, your answer is {sum(state["values"])}'
    else:
        state["result"] = f'Hi {state["name"]}, your answer is {math.prod(state["values"])}'

    return state

if __name__ == '__main__':
    print('Exercise 2')

    graph = StateGraph(AgentState)

    graph.add_node("operator", operate)
    graph.set_entry_point("operator")
    graph.set_finish_point("operator")

    app = graph.compile()

    result = app.invoke({"values": [1, 2, 3, 4], "name": "Bob", "operation": "+"})

    print(result)

    result = app.invoke({"values": [1, 2, 3, 4], "name": "Bob", "operation": "*"})
    print(result)