from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from PIL import Image
import io
import matplotlib.pyplot as plt

class AgentState(TypedDict):
    number1: int
    operation: str
    number2: int
    finalNumber: int


def adder(state: AgentState) -> AgentState:
    """ Node to do add two numbers"""
    state["finalNumber"] = state["number1"] + state["number2"]
    return state


def subtractor(state: AgentState) -> AgentState:
    """ Node to do subtract two numbers"""
    state["finalNumber"] = state["number1"] - state["number2"]
    return state


def decide_next_node(state: AgentState) -> AgentState:
    """ Decider node to decide next step in the graph"""
    if state["operation"] == "+":
        return "addition_operation"
    else:
        return "subtraction_operation"


if __name__ == '__main__':
    graph = StateGraph(AgentState)

    graph.add_node("add_node", adder)
    graph.add_node("subtract_node", subtractor)

    graph.add_node("router", lambda state: state)  # pass through function. State is not changed so just assign the
    # input state to output state

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        decide_next_node,
        {
            # Edge: Node
            "addition_operation": "add_node",
            "subtraction_operation": "subtract_node"
        }
    )
    graph.add_edge("add_node", END)
    graph.add_edge("subtract_node", END)

    app = graph.compile()

    image_bytes = app.get_graph().draw_mermaid_png()
    image = Image.open(io.BytesIO(image_bytes))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    initial_state_1 = {"number1": 10, "number2": 10, "operation": "+"}
    result = app.invoke(initial_state_1)
    print(result)
    initial_state_1 = {"number1": 10, "number2": 10, "operation": "*"}
    result = app.invoke(initial_state_1)

    print(result)
