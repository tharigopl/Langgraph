from typing import Dict, TypedDict
from langgraph.graph import StateGraph


# State Schema
class AgentState(TypedDict):
    message: str
    name : str


def greeting_node(state: AgentState) -> AgentState:
    """Simple node that adds a greeting message to the state"""

    state['message'] = "Hey " + state["message"] + ", how are you doing?"

    return state


def compliment_node(state: AgentState) -> AgentState:
    """ Simple node to return compliment message"""
    # f'{state["message"]}, you \'re doing an amazing job learning LangGraph!'
    state["name"] = f'{state["name"]}, you \'re doing an amazing job learning LangGraph!'
    return state



