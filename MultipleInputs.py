from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph

class MultipleInputAgentState(TypedDict):
    values: List[int]
    name: str
    result: str

def process_values(state: MultipleInputAgentState)  -> MultipleInputAgentState:
    """ Node to process multiple different inputs"""

    state["result"] = f'Hi there {state["name"]}! Your sum = {sum(state["values"])}'
    return state


if __name__ == '__main__':
    print('Multiple Inputs')
    graph = StateGraph(MultipleInputAgentState)

    graph.add_node("process_values", process_values)

    graph.set_entry_point("process_values")
    graph.set_finish_point("process_values")

    app = graph.compile()

    result = app.invoke({"values": [1,2], "name": "Bob"})

    print(result["result"])
    print(result["values"])
    print(result["name"])