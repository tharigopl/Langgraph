from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, START, END
import random
from PIL import Image
import io
import matplotlib.pyplot as plt


class AgentState(TypedDict):
    name: str
    numbers: List[int]
    counter: int
    counterMax: int

def greeting_node(state: AgentState) -> AgentState:
    """ Greeting node which says hi to person"""
    state["name"] = f'Hi there, {state["name"]}'
    state["counter"] = 0
    #state["counterMax"] = 0
    return state

def random_node(state: AgentState) -> AgentState:
    """Random node generates number from 0 to 10"""
    state["numbers"].append(random.randint(0, 10))
    state["counter"] += 1

    return state


def should_continue(state: AgentState) -> AgentState:
    """function which decides to end the loop or not"""
    if state["counter"] < state["counterMax"]:
        print("Enter loop", state["counter"])
        return "loop"
    else:
        return "exit"


if __name__ == '__main__':
    print('Looping Graph')

    graph = StateGraph(AgentState)

    graph.add_node("greeting_node", greeting_node)
    graph.add_node("random_node", random_node)

    graph.add_edge(START, "greeting_node")

    graph.add_edge("greeting_node", "random_node")

    graph.add_conditional_edges(
        "random_node", # Source
        should_continue, # Action
        {
            "loop": "random_node",
            "exit": END
        }
    )

    app = graph.compile()

    result = app.invoke({"name":"Rosario", "numbers":[], "counterMax":10})

    image_bytes = app.get_graph().draw_mermaid_png()
    image = Image.open(io.BytesIO(image_bytes))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    print(result)