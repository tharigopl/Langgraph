# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from typing import Dict, TypedDict
from langgraph.graph import StateGraph
from AgentState import AgentState, greeting_node
from IPython.display import Image as IPImage, display
import io
import matplotlib.pyplot as plt
from PIL import Image

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    graph = StateGraph(AgentState)

    # Add a node

    graph.add_node("greeter", greeting_node)

    graph.set_entry_point("greeter")
    graph.set_finish_point("greeter")

    app = graph.compile()

    # display(IPImage(app.get_graph().draw_mermaid_png()))
    #
    # # Read raw image bytes from LangGraph
    # image_bytes = app.get_graph().draw_mermaid_png()
    #
    # # Convert bytes and display via matplotlib
    # image = Image.open(io.BytesIO(image_bytes))
    # plt.imshow(image)
    # plt.axis("off")  # Hide grid axes
    # plt.show()  # Pops up the PyCharm SciView / plot window

    result = app.invoke({"message" : "bob"})

    print(result["message"])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
