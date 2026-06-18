from typing import Dict, TypedDict
from langgraph.graph import StateGraph
from PIL import Image
import io
import matplotlib.pyplot as plt


class AgentState(TypedDict):
    name: str
    age: str
    final: str


def first_node(state: AgentState) -> AgentState:
    """ This is the first node of the sequence"""
    state["final"] = f'Hi {state["name"]}!'
    return state


def second_node(state: AgentState) -> AgentState:
    """ This is the second node of the sequence"""
    state["final"] = state["final"] + f' Your are {state["age"]} years old!'
    return state


if __name__ == '__main__':
    print('sequential graph')

    graph = StateGraph(AgentState)

    graph.add_node("first_node", first_node)
    graph.add_node("second_node", second_node)

    graph.set_entry_point("first_node")
    graph.add_edge("first_node", "second_node")
    graph.set_finish_point("second_node")

    app = graph.compile()

    result = app.invoke({"name": "Rosario", "age": "38"})

    print(result)

    # image_bytes = app.get_graph().draw_mermaid_png()
    # image = Image.open(io.BytesIO(image_bytes))
    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()