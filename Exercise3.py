from typing import Dict, TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    name: str
    age: str
    skills: list
    result: str

def personalize_name(state: AgentState) -> AgentState:
    """ First node to personalize name"""
    state["result"] = f'{state["name"]}, welcome to the system! '
    return state


def describe_age(state: AgentState) -> AgentState:
    """ Second node to describe age"""
    state["result"] = state["result"] + f'You are {state["age"]} years old! '
    return state

def list_skills(state: AgentState) -> AgentState:
    """ Third node to list skills"""
    state["result"] = state["result"] + f'You have skills in: { ", ".join(state["skills"])}.'
    return state

if __name__ == '__main__':
    print('Exercise 3')

    graph = StateGraph(AgentState)

    graph.add_node("personalize_name", personalize_name)
    graph.add_node("describe_age", describe_age)
    graph.add_node("list_skills", list_skills)

    graph.set_entry_point("personalize_name")
    graph.set_finish_point("list_skills")

    graph.add_edge("personalize_name", "describe_age")
    graph.add_edge("describe_age", "list_skills")

    app = graph.compile()

    result = app.invoke({"name": "Rosario", "age": "38", "skills": ["Python", "Machine Learning", "Langgraph"]})

    print(result)

