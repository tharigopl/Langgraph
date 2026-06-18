from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, START, END
import random
from PIL import Image
import io
import matplotlib.pyplot as plt

class AgentState(TypedDict):
    player_name: str
    guesses: List
    attempts: int
    lower_bound: int
    upper_bound: int
    numberToGuess: int
    maxAttempts: int
    hint: str


def setup(state: AgentState) -> AgentState:
    """ Set up node initializes the state"""
    state['player_name'] = f'Welcome, {state["player_name"]}! The game has begun. I \'m thinking of a number between {state["lower_bound"]} and {state["upper_bound"]}'
    state['attempts'] = 0
    state['guesses'] = []
    state['lower_bound'] = 1
    state['upper_bound'] = 20
    print(state["player_name"])
    return state


def guess(state: AgentState) -> AgentState:
    """Guess node guesses the number"""
    guess = random.randint(state["lower_bound"], state["upper_bound"])
    state["guesses"].append(guess)
    state["attempts"] += 1
    print(f'Attempt {state["attempts"]}: Guessing {guess} (Current range: {state["lower_bound"]} - {state["upper_bound"]})')
    return state


def hint(state:AgentState) -> AgentState:
    """Checks if the guessed number matches and prints information"""

    # get latest guess
    lastGuess = state["guesses"][-1]
    if lastGuess > state["numberToGuess"]:
        state["upper_bound"] = lastGuess - 1
        print(f'Hint: The number {lastGuess} is too high. Try lower!')
    elif lastGuess < state["numberToGuess"]:
        state["lower_bound"] = lastGuess + 1
        print(f'Hint: The number {lastGuess} is too low. Try higher!')
    return state
def should_continue(state:AgentState) -> AgentState:
    """should_continue conditional edge checks if the guessed number is correct or higher or lower"""
    guessedNumber = state["guesses"][-1]

    if state["attempts"] == state["maxAttempts"]:
        print(f'Game over! Reached max attempts {state["maxAttempts"]}')
        return "end"
    elif guessedNumber == state["numberToGuess"]:
        print(f'Success! Correct! You found the number {state["numberToGuess"]} in {state["attempts"]} attempts.')
        return "end"
    else:
        print(f'CONTINUING: {state["attempts"]}/{state["maxAttempts"]} attempts used.')
        return "continue"


if __name__ == '__main__':
    graph = StateGraph(AgentState)

    graph.add_node("setup_node", setup)
    graph.add_node("guess_node", guess)
    graph.add_node("hint_node", hint)

    graph.add_edge(START, "setup_node")
    graph.add_edge("setup_node", "guess_node")
    graph.add_edge("guess_node", "hint_node")

    graph.add_conditional_edges(
        "hint_node",
        should_continue,
        {
            "end": END,
            "continue": "guess_node"
        }
    )

    app = graph.compile()

    # image_bytes = app.get_graph().draw_mermaid_png()
    # image = Image.open(io.BytesIO(image_bytes))
    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()

    input = {"player_name": "Student", "guesses": [], "attempts": 0, "lower_bound": 1, "upper_bound": 20, "maxAttempts": 4, "numberToGuess" : 10}
    result = app.invoke(input)

    print(result)