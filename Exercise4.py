from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from PIL import Image
import io
import matplotlib.pyplot as plt

class AgentState(TypedDict):
    number1: int
    number2: int
    number3: int
    number4: int
    finalNumber1: int
    finalNumber2: int
    operation1: str
    operation2: str


def add_node1(state: AgentState) -> AgentState:
    state['finalNumber1'] = state['number1'] + state['number2']
    return state


def add_node2(state: AgentState) -> AgentState:
    state['finalNumber2'] = state['number3'] + state['number4']
    return state


def subtract_node1(state: AgentState) -> AgentState:
    state['finalNumber1'] = state['number1'] - state['number2']
    return state


def subtract_node2(state: AgentState) -> AgentState:
    state['finalNumber2'] = state['number3'] - state['number4']
    return state


def decide_next_node1(state: AgentState):
    if state['operation1'] == "+":
        return "add_node_1"
    else:
        return "subtract_node_1"


def decide_next_node2(state: AgentState):
    if state['operation2'] == "+":
        return "add_node_2"
    else:
        return "subtract_node_2"


if __name__ == '__main__':
    print('Exercise 4')

    graph = StateGraph(AgentState)

    graph.add_node("add_node1", add_node1)
    graph.add_node("add_node2", add_node2)
    graph.add_node("subtract_node1", subtract_node1)
    graph.add_node("subtract_node2", subtract_node2)

    graph.add_node("decide_next_node1", lambda state: state)

    graph.add_edge(START, "decide_next_node1")

    graph.add_conditional_edges(
        "decide_next_node1",
        decide_next_node1,
        {
            "add_node_1": "add_node1",
            "subtract_node_1": "subtract_node1"
        }
    )

    graph.add_node("decide_next_node2", lambda state: state)

    graph.add_edge("add_node1", "decide_next_node2")
    graph.add_edge("subtract_node1", "decide_next_node2")

    graph.add_conditional_edges(
        "decide_next_node2",
        decide_next_node2,
        {
            "add_node_2": "add_node2",
            "subtract_node_2": "subtract_node2"
        }
    )

    graph.add_edge("add_node2", END)
    graph.add_edge("subtract_node2", END)

    app = graph.compile()

    initialinput = AgentState(number1=10, operation1="-", number2=5, number3=7, operation2="+", number4=2)

    result = app.invoke(initialinput)

    print(result)

    # image_bytes = app.get_graph().draw_mermaid_png()
    # image = Image.open(io.BytesIO(image_bytes))
    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()