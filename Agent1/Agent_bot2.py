import os
from typing import Dict, TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f'\n AI: {response.content}')
    return state

if __name__ == '__main__':
    print("AI Agent 2")
    graph = StateGraph(AgentState)

    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    agent = graph.compile()

    conversation_history = []

    with open("logging.txt", "r") as file:
        for line in file:
            msgInHistory = line.strip()
            print(msgInHistory)
            if msgInHistory.startswith('You'):
                conversation_history.append(HumanMessage(msgInHistory))
            elif msgInHistory.startswith('AI'):
                conversation_history.append(AIMessage(msgInHistory))

    user_input = input("Enter: ")

    while user_input != "exit":
        conversation_history.append(HumanMessage(content=user_input))

        result = agent.invoke({"messages": conversation_history})

        conversation_history = result["messages"]
        user_input = input("Enter: ")

    with open("logging.txt", "w+") as file:
        file.write("Your conversation history: \n")

        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f'You: {message.content}\n')
            else:
                file.write(f'AI: {message.content}\n')
        file.write("End of conversation")
    print("Conversation saved in logging.txt file")